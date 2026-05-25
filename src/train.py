"""Training entrypoint for the gated-fusion model.

Run with:
    python -m src.train --config configs/config.yaml --fold 0
    python -m src.train --config configs/config.yaml --fold 0 --smoke
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.dataset import OralCancerDataset
from src.augmentations import (
    build_bf_color_transform,
    build_fl_color_transform,
    build_shared_geo_transform,
)
from src.models import (
    CrossAttentionFusionModel,
    EarlyFusionConcatModel,
    GatedFusionModel,
)
from src.utils import extract_patient_id, get_patient_splits, set_seed


def build_model(cfg):
    """Return (model, set_freeze, freeze_keys) where freeze_keys is the pair
    of strategy names for warmup / partial-unfreeze phases."""
    model_type = cfg['model'].get('type', 'gated')
    if model_type == 'cross_attention':
        model = CrossAttentionFusionModel(
            backbone_name=cfg['model']['backbone'],
            pretrained=True,
            proj_dim=cfg['model'].get('proj_dim', 64),
            num_heads=cfg['model'].get('num_heads', 4),
            window_size=cfg['model'].get('window_size', 3),
        )
        return model, model.set_freeze_strategy, ('warmup', 'partial')
    if model_type == 'early_fusion_concat':
        model = EarlyFusionConcatModel(
            backbone_name=cfg['model']['backbone'],
            pretrained=True,
        )
        return model, model.set_freeze_strategy, ('warmup', 'partial')
    if model_type == 'gated':
        model = GatedFusionModel(
            model_name=cfg['model']['backbone'],
            pretrained=True,
        )
        return model, model.set_encoders_freeze_strategy, ('full', 'partial')
    raise ValueError(f"unknown model.type: {model_type!r}")


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg, train_idx, val_idx, filenames, labels):
    train_ds = OralCancerDataset(
        filenames=[filenames[i] for i in train_idx],
        labels=[labels[i] for i in train_idx],
        bf_dir=cfg['data']['bf_train_dir'],
        fl_dir=cfg['data']['fl_train_dir'],
        bf_transform=build_bf_color_transform(),
        fl_transform=build_fl_color_transform(),
        geo_transform=build_shared_geo_transform(),
    )
    val_ds = OralCancerDataset(
        filenames=[filenames[i] for i in val_idx],
        labels=[labels[i] for i in val_idx],
        bf_dir=cfg['data']['bf_train_dir'],
        fl_dir=cfg['data']['fl_train_dir'],
    )
    num_workers = cfg['training']['num_workers']
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def mixup_batch(bf, fl, labels, alpha):
    """Same lambda and permutation applied to both modalities — required to
    preserve BF–FL pairing under mixup."""
    if alpha <= 0:
        return bf, fl, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(bf.size(0), device=bf.device)
    bf = lam * bf + (1 - lam) * bf[idx]
    fl = lam * fl + (1 - lam) * fl[idx]
    return bf, fl, labels, labels[idx], lam


def sample_weights(labels_a, labels_b, lam, class_weights, device):
    w = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return lam * w[labels_a.long()] + (1 - lam) * w[labels_b.long()]


def train(model, device, train_loader, optimizer, scaler, cfg, use_amp):
    model.train()
    mixup_alpha = cfg['training']['mixup_alpha']
    class_weights = cfg['training']['class_weights']
    accum_steps = max(1, int(cfg['training'].get('accum_steps', 1)))
    total_loss = 0.0
    n_batches = 0

    # Approximate train-AUC: pair each batch's sigmoid output with the
    # dominant-side MixUp label (lab_a if lam >= 0.5 else lab_b). Cheap and
    # avoids a second forward pass; biased slightly toward 0.5 when lam ~ 0.5.
    all_probs, all_labels = [], []

    optimizer.zero_grad(set_to_none=True)
    n_batches_total = len(train_loader)

    for i, (bf, fl, labels, _) in enumerate(train_loader):
        bf = bf.to(device, non_blocking=True)
        fl = fl.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        bf, fl, lab_a, lab_b, lam = mixup_batch(bf, fl, labels, mixup_alpha)
        mixed_targets = lam * lab_a + (1 - lam) * lab_b
        weights = sample_weights(lab_a, lab_b, lam, class_weights, device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(bf, fl).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(
                logits, mixed_targets, weight=weights
            )

        # Scale down so accumulated gradient matches the average over accum_steps mini-batches.
        loss_to_back = loss / accum_steps

        if use_amp:
            scaler.scale(loss_to_back).backward()
        else:
            loss_to_back.backward()

        # Step on completed accumulation cycles or on the very last batch
        # (so a trailing partial cycle still contributes).
        is_step = ((i + 1) % accum_steps == 0) or ((i + 1) == n_batches_total)
        if is_step:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            dominant = lab_a if lam >= 0.5 else lab_b
            all_probs.append(torch.sigmoid(logits).float().detach().cpu().numpy())
            all_labels.append(dominant.detach().cpu().numpy().astype(int))

        total_loss += loss.item()  # per-mini-batch unscaled loss for reporting
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    probs = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels)
    train_auc = (
        roc_auc_score(labels_np, probs)
        if len(np.unique(labels_np)) > 1 else float('nan')
    )
    return avg_loss, train_auc


@torch.no_grad()
def test(model, device, val_loader):
    """Returns (cell_level_auc, per_patient_summary).

    per_patient_summary is a dict pid -> (true_label, mean_prob, n_cells).
    Labels are constant within a patient (weak supervision), so within-patient
    AUC is undefined — we report mean predicted prob instead.
    """
    model.eval()
    all_probs, all_labels, all_fnames = [], [], []

    for bf, fl, labels, fnames in val_loader:
        bf = bf.to(device, non_blocking=True)
        fl = fl.to(device, non_blocking=True)
        logits = model(bf, fl).squeeze(1)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
        all_fnames.extend(fnames)

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)

    cell_auc = roc_auc_score(labels, probs)

    by_pat = defaultdict(lambda: {'probs': [], 'label': None})
    for f, p, l in zip(all_fnames, probs, labels):
        pid = extract_patient_id(f)
        by_pat[pid]['probs'].append(float(p))
        by_pat[pid]['label'] = int(l)

    per_patient = {
        pid: (d['label'], float(np.mean(d['probs'])), len(d['probs']))
        for pid, d in by_pat.items()
    }
    return cell_auc, per_patient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--smoke', action='store_true',
                        help='Single forward pass on one batch, then exit.')
    parser.add_argument('--wandb', action='store_true',
                        help='Log to Weights & Biases.')
    parser.add_argument('--wandb-project', default='oral-cancer-classification')
    parser.add_argument('--wandb-run-name', default=None,
                        help='Defaults to "fold{N}".')
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['split']['seed'])

    labels_df = pd.read_csv(cfg['data']['labels_csv'])
    filenames = labels_df['Name'].tolist()
    labels = labels_df['Diagnosis'].astype(int).tolist()

    folds = get_patient_splits(
        filenames, labels,
        n_folds=cfg['split']['n_folds'],
        seed=cfg['split']['seed'],
    )
    train_idx, val_idx = folds[args.fold]
    print(f"fold {args.fold}: {len(train_idx)} train cells, {len(val_idx)} val cells")

    train_loader, val_loader = build_dataloaders(cfg, train_idx, val_idx, filenames, labels)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, set_freeze, (freeze_full, freeze_partial) = build_model(cfg)
    model = model.to(device)

    pre = cfg.get('pretraining', {})
    if pre.get('enabled', False) and cfg['model'].get('type') == 'early_fusion_concat':
        ckpt_path = pre.get('checkpoint_path')
        if ckpt_path and os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
            print(f"loaded pretrained backbone from {ckpt_path} "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            print(f"warning: pretraining.enabled=true but checkpoint not found at {ckpt_path!r}; "
                  f"continuing with ImageNet-init backbone")

    if args.smoke:
        bf, fl, _y, _ = next(iter(train_loader))
        bf, fl = bf.to(device), fl.to(device)
        with torch.no_grad():
            logits = model(bf, fl)
        print(f"smoke: bf {tuple(bf.shape)}, fl {tuple(fl.shape)} -> logits {tuple(logits.shape)}")
        return

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f'fold{args.fold}',
            config={**cfg, 'fold': args.fold},
            tags=[f'fold{args.fold}'],
        )

    epochs = cfg['training']['epochs']
    freeze_epochs = cfg['training']['freeze_backbone_epochs']
    use_amp = device.type == 'cuda'

    set_freeze(freeze_full)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=cfg['training']['eta_min']
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    ckpt_dir = cfg['output']['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    best_auc = -1.0
    best_path = os.path.join(ckpt_dir, f'fold{args.fold}_best.pt')
    last_path = os.path.join(ckpt_dir, f'fold{args.fold}_last.pt')

    for epoch in range(epochs):
        if epoch == freeze_epochs:
            print(f"epoch {epoch}: switching backbone to partial unfreeze (stages 2-3 + norms)")
            set_freeze(freeze_partial)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=cfg['training']['lr'],
                weight_decay=cfg['training']['weight_decay'],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch, eta_min=cfg['training']['eta_min']
            )

        train_loss, train_auc = train(model, device, train_loader, optimizer, scaler, cfg, use_amp)
        scheduler.step()
        val_auc, per_patient = test(model, device, val_loader)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"epoch {epoch:02d} | lr {lr_now:.2e} | train_loss {train_loss:.4f} | train_auc {train_auc:.4f} | val_auc {val_auc:.4f}")
        for pid in sorted(per_patient):
            lbl, mp, n = per_patient[pid]
            print(f"    pat_{pid} (label={lbl}, n={n}): mean_prob={mp:.3f}")

        if use_wandb:
            log = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/auc': train_auc,
                'val/auc': val_auc,
                'lr': lr_now,
            }
            for pid, (lbl, mp, n) in per_patient.items():
                log[f'val/pat_{pid}_mean_prob'] = mp
                log[f'val/pat_{pid}_label'] = lbl
            wandb.log(log)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'fold': args.fold,
                'model_state': model.state_dict(),
                'val_auc': val_auc,
                'config': cfg,
            }, best_path)
            print(f"    -> new best (auc={val_auc:.4f}), saved to {best_path}")
            if use_wandb:
                wandb.summary['best_val_auc'] = best_auc
                wandb.summary['best_epoch'] = epoch

        torch.save({
            'epoch': epoch,
            'fold': args.fold,
            'model_state': model.state_dict(),
            'val_auc': val_auc,
            'config': cfg,
        }, last_path)

    print(f"fold {args.fold} done. best val AUC: {best_auc:.4f} (best: {best_path}, last: {last_path})")
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
