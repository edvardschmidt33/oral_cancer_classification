"""SimCLR-style self-supervised pretraining for the 6-channel early-fusion
backbone. After pretraining, the backbone state_dict is saved to
`pretraining.checkpoint_path` and supervised training (when
`pretraining.enabled: true`) loads it before fine-tuning.

Run with:
    python -m src.pretrain --config configs/config_ef.yaml
    python -m src.pretrain --config configs/config_ef.yaml --smoke
"""
import argparse
import os

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from src.augmentations import SimCLRAugmentation
from src.dataset import SimCLRDataset
from src.models import SimCLRModel
from src.utils import set_seed


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def nt_xent_loss(z1, z2, temperature=0.1):
    """Symmetric NT-Xent loss. Assumes z1, z2 are L2-normalized [B, D].
    Positives: (z1[i], z2[i]); negatives: all other 2B-2 entries."""
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                         # [2B, D]
    sim = z @ z.T / temperature                            # [2B, 2B]

    # Mask out self-similarity on the diagonal.
    diag_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(diag_mask, -1e9)

    # Positive index for row i: i+B (mod 2B).
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ]).long()

    return F.cross_entropy(sim, labels)


def build_loader(cfg):
    """Pool both train/ and test/ images (no labels needed for SimCLR)."""
    bf_train = cfg['data']['bf_train_dir']
    fl_train = cfg['data']['fl_train_dir']
    bf_test = cfg['data']['bf_test_dir']
    fl_test = cfg['data']['fl_test_dir']

    train_files = sorted(os.listdir(bf_train))
    test_files = sorted(os.listdir(bf_test))
    train_files = [f for f in train_files if f.lower().endswith('.jpg')]
    test_files = [f for f in test_files if f.lower().endswith('.jpg')]

    aug = SimCLRAugmentation()
    train_ds = SimCLRDataset(train_files, bf_train, fl_train, aug)
    test_ds = SimCLRDataset(test_files, bf_test, fl_test, aug)
    dataset = ConcatDataset([train_ds, test_ds])

    pre = cfg['pretraining']
    loader = DataLoader(
        dataset,
        batch_size=pre['batch_size'],
        shuffle=True,
        num_workers=pre['num_workers'],
        pin_memory=True,
        drop_last=True,                            # NT-Xent needs a fixed B
        persistent_workers=pre['num_workers'] > 0,
    )
    print(f"pretraining pool: train={len(train_ds)} + test={len(test_ds)} = {len(dataset)} cells")
    return loader


def apply_freeze(model, strategy):
    """'partial' = freeze backbone stages 0-1 only; 'none' = everything trainable.
    The 6-channel stem is always trainable since it was newly initialized."""
    for p in model.proj_head.parameters():
        p.requires_grad = True

    for name, p in model.backbone.named_parameters():
        is_stem = name.startswith('stem')
        if strategy == 'partial':
            p.requires_grad = (
                is_stem
                or any(k in name for k in ['stages.2', 'stages.3'])
                or ('norm' in name and 'stages.0' not in name and 'stages.1' not in name)
            )
        elif strategy == 'none':
            p.requires_grad = True
        else:
            raise ValueError(f"unknown pretraining freeze strategy: {strategy!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_ef.yaml')
    parser.add_argument('--smoke', action='store_true',
                        help='Single forward + backward on one batch, then exit.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    pre = cfg.get('pretraining', {})
    if not pre.get('enabled', False) and not args.smoke:
        print(f"pretraining.enabled=False in {args.config}; nothing to do.")
        return

    set_seed(cfg['split']['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    loader = build_loader(cfg)

    model = SimCLRModel(
        backbone_name=cfg['model']['backbone'],
        pretrained=True,
        proj_dim=pre['proj_dim'],
        proj_hidden_dim=pre.get('proj_hidden_dim'),
    ).to(device)
    apply_freeze(model, pre['freeze_strategy'])

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=pre['lr'],
        weight_decay=pre['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=pre['epochs'], eta_min=pre.get('eta_min', 1e-7),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    accum_steps = max(1, int(pre.get('accum_steps', 1)))
    temperature = pre['temperature']

    if args.smoke:
        x1, x2 = next(iter(loader))
        x1, x2 = x1.to(device), x2.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            z1, z2 = model(x1), model(x2)
            loss = nt_xent_loss(z1, z2, temperature=temperature)
        loss.backward()
        print(f"smoke: x1 {tuple(x1.shape)} -> z1 {tuple(z1.shape)}, loss {loss.item():.4f}")
        return

    ckpt_path = pre['checkpoint_path']
    os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)
    epochs = pre['epochs']

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        n_steps = 0
        n_batches_total = len(loader)

        optimizer.zero_grad(set_to_none=True)

        for i, (x1, x2) in enumerate(loader):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                z1 = model(x1)
                z2 = model(x2)
                loss = nt_xent_loss(z1, z2, temperature=temperature)

            loss_to_back = loss / accum_steps
            if use_amp:
                scaler.scale(loss_to_back).backward()
            else:
                loss_to_back.backward()

            is_step = ((i + 1) % accum_steps == 0) or ((i + 1) == n_batches_total)
            if is_step:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_steps += 1

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"epoch {epoch:02d} | lr {lr_now:.2e} | nt_xent {avg_loss:.4f} | steps {n_steps}")

        # Save after every epoch (overwrites) so we always have the latest.
        torch.save(model.backbone.state_dict(), ckpt_path)

    print(f"done. backbone weights at {ckpt_path}")


if __name__ == '__main__':
    main()
