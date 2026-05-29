"""Training entrypoint for the early-fusion model.

Run with:
    python -m src.train --config configs/config_ef.yaml --fold 0
    python -m src.train --config configs/config_ef.yaml --fold 0 --smoke
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn import metrics
from torch.utils.data import DataLoader

from src.augmentations import (
    build_bf_color_transform,
    build_fl_color_transform,
    build_shared_geo_transform,
)
from src.dataset import OralCancerDataset
from src.models import EarlyFusionConcatModel
from src.utils import (
    compute_norm_stats,
    extract_patient_id,
    get_patient_splits,
    probe_fl_channels,
    set_seed,
)


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg, train_idx, val_idx, filenames, labels,
                      bf_mean, bf_std, fl_mean, fl_std, fl_channels):
    size = cfg['model']['img_size']
    crop_size = cfg['model'].get('crop_size')
    train_ds = OralCancerDataset(
        filenames=[filenames[i] for i in train_idx],
        labels=[labels[i] for i in train_idx],
        bf_dir=cfg['data']['bf_train_dir'],
        fl_dir=cfg['data']['fl_train_dir'],
        size=size, crop_size=crop_size,
        bf_mean=bf_mean, bf_std=bf_std,
        fl_mean=fl_mean, fl_std=fl_std,
        fl_channels=fl_channels,
        bf_color=build_bf_color_transform(),
        fl_color=build_fl_color_transform(fl_channels=fl_channels),
        geo_transform=build_shared_geo_transform(),
        fl_tensor_blur=True,
    )
    val_ds = OralCancerDataset(
        filenames=[filenames[i] for i in val_idx],
        labels=[labels[i] for i in val_idx],
        bf_dir=cfg['data']['bf_train_dir'],
        fl_dir=cfg['data']['fl_train_dir'],
        size=size, crop_size=crop_size,
        bf_mean=bf_mean, bf_std=bf_std,
        fl_mean=fl_mean, fl_std=fl_std,
        fl_channels=fl_channels,
        bf_color=None, fl_color=None, geo_transform=None,
        fl_tensor_blur=False,
    )
    num_workers = cfg['training']['num_workers']
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return train_loader, val_loader


def mixup(x, y, alpha):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def weighted_bce(logits, y, w_pos, w_neg):
    """Per-sample weighted BCE: weight is keyed by the target side (y > 0.5).
    Matches the formulation in EF_convnext_best.py."""
    w = torch.where(y > 0.5, torch.full_like(y, w_pos), torch.full_like(y, w_neg))
    return F.binary_cross_entropy_with_logits(logits, y, weight=w)


def cell_metrics(y, s, threshold=0.5):
    p = (s > threshold).astype(int)
    return {
        'AUC':  metrics.roc_auc_score(y, s) if len(np.unique(y)) > 1 else float('nan'),
        'F1':   metrics.f1_score(y, p, zero_division=0),
        'Acc':  metrics.accuracy_score(y, p),
        'Prec': metrics.precision_score(y, p, zero_division=0),
        'Rec':  metrics.recall_score(y, p, zero_division=0),
    }


def patient_auc(names, y, s):
    """Aggregate cell scores to patient level (mean) and compute AUC."""
    d = pd.DataFrame({'patient': [extract_patient_id(n) for n in names], 'y': y, 's': s})
    g = d.groupby('patient').agg(y=('y', 'first'), s=('s', 'mean'))
    if g['y'].nunique() < 2:
        return float('nan')
    return metrics.roc_auc_score(g['y'], g['s'])


def run_epoch(model, loader, optimizer, scaler, device, train_mode,
              use_mixup, use_amp, mixup_alpha, w_pos, w_neg):
    model.train(train_mode)
    all_y, all_s, all_names = [], [], []
    total_loss = 0.0
    for batch in loader:
        x, y, names = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        if train_mode and use_mixup:
            x, y_a, y_b, lam = mixup(x, y, alpha=mixup_alpha)

        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x).squeeze(1)
                if train_mode and use_mixup:
                    loss = (lam * weighted_bce(logits, y_a, w_pos, w_neg)
                            + (1 - lam) * weighted_bce(logits, y_b, w_pos, w_neg))
                else:
                    loss = weighted_bce(logits, y, w_pos, w_neg)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        all_s.extend(torch.sigmoid(logits.float()).detach().cpu().tolist())
        all_y.extend(y.detach().cpu().tolist())
        all_names.extend(names)

    avg_loss = total_loss / max(len(loader.dataset), 1)
    return avg_loss, np.array(all_y), np.array(all_s), all_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_ef.yaml')
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    use_amp = device.type == 'cuda'

    labels_df = pd.read_csv(cfg['data']['labels_csv'])
    filenames = labels_df['Name'].tolist()
    labels = labels_df['Diagnosis'].astype(int).tolist()

    fl_channels = probe_fl_channels(cfg['data']['fl_train_dir'])
    print(f"FL channels detected: {fl_channels}")

    folds = get_patient_splits(
        filenames, labels,
        n_folds=cfg['split']['n_folds'],
        seed=cfg['split']['seed'],
    )
    train_idx, val_idx = folds[args.fold]
    print(f"fold {args.fold}: {len(train_idx)} train cells, {len(val_idx)} val cells")

    # Class weights from the actual training distribution (inverse frequency).
    train_labels = np.array([labels[i] for i in train_idx], dtype=int)
    pos_rate = float(train_labels.mean())
    neg_rate = 1.0 - pos_rate
    w_pos, w_neg = float(neg_rate), float(pos_rate)
    print(f"train cancer rate: {pos_rate:.4f} | W_POS={w_pos:.4f}, W_NEG={w_neg:.4f}")

    # Per-dataset normalization stats from the training names of this fold.
    train_names = [filenames[i] for i in train_idx]
    bf_mean, bf_std = compute_norm_stats(
        cfg['data']['bf_train_dir'], train_names,
        modality='BF', sample_size=2000, seed=cfg['split']['seed'],
    )
    fl_mean, fl_std = compute_norm_stats(
        cfg['data']['fl_train_dir'], train_names,
        modality='FL', fl_channels=fl_channels,
        sample_size=2000, seed=cfg['split']['seed'],
    )
    print(f"BF mean: {bf_mean}\nBF std:  {bf_std}")
    print(f"FL mean: {fl_mean}\nFL std:  {fl_std}")

    train_loader, val_loader = build_dataloaders(
        cfg, train_idx, val_idx, filenames, labels,
        bf_mean, bf_std, fl_mean, fl_std, fl_channels,
    )

    total_channels = 3 + fl_channels
    model = EarlyFusionConcatModel(
        backbone_name=cfg['model']['backbone'],
        pretrained=True,
        total_channels=total_channels,
        dropout=cfg['training'].get('dropout', 0.15),
    ).to(device)
    print(f"total params: {sum(p.numel() for p in model.parameters()):,}")

    pre = cfg.get('pretraining', {})
    if pre.get('enabled', False):
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
        x, _y, _n = next(iter(train_loader))
        x = x.to(device)
        with torch.no_grad():
            logits = model(x)
        print(f"smoke: x {tuple(x.shape)} -> logits {tuple(logits.shape)}")
        return

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f'fold{args.fold}',
            config={**cfg, 'fold': args.fold,
                    'bf_mean': bf_mean, 'bf_std': bf_std,
                    'fl_mean': fl_mean, 'fl_std': fl_std,
                    'w_pos': w_pos, 'w_neg': w_neg},
            tags=[f'fold{args.fold}'],
        )

    epochs = cfg['training']['epochs']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=cfg['training']['eta_min']
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    ckpt_dir = cfg['output']['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    best_smoothed = -1.0
    best_path = os.path.join(ckpt_dir, f'fold{args.fold}_best.pt')
    last_path = os.path.join(ckpt_dir, f'fold{args.fold}_last.pt')
    auc_history = []

    mixup_alpha = cfg['training']['mixup_alpha']
    use_mixup = mixup_alpha > 0

    for epoch in range(epochs):
        tr_loss, _, _, _ = run_epoch(
            model, train_loader, optimizer, scaler, device,
            train_mode=True, use_mixup=use_mixup, use_amp=use_amp,
            mixup_alpha=mixup_alpha, w_pos=w_pos, w_neg=w_neg,
        )
        scheduler.step()
        va_loss, y, s, names = run_epoch(
            model, val_loader, optimizer, scaler, device,
            train_mode=False, use_mixup=False, use_amp=use_amp,
            mixup_alpha=0.0, w_pos=w_pos, w_neg=w_neg,
        )

        m = cell_metrics(y, s)
        p_auc = patient_auc(names, y, s)
        auc_history.append(m['AUC'])
        smoothed = float(np.mean(auc_history[-3:]))
        lr_now = optimizer.param_groups[0]['lr']

        print(f"epoch {epoch:02d} | lr {lr_now:.2e} | tr_loss {tr_loss:.4f} | "
              f"va_loss {va_loss:.4f} | cell_AUC={m['AUC']:.4f} (sm {smoothed:.4f}) | "
              f"pat_AUC={p_auc:.4f} | F1={m['F1']:.4f} Acc={m['Acc']:.4f}")

        if use_wandb:
            log = {
                'epoch': epoch,
                'train/loss': tr_loss,
                'val/loss': va_loss,
                'val/cell_auc': m['AUC'],
                'val/cell_auc_smoothed': smoothed,
                'val/patient_auc': p_auc,
                'val/f1': m['F1'],
                'val/acc': m['Acc'],
                'val/prec': m['Prec'],
                'val/rec': m['Rec'],
                'lr': lr_now,
            }
            wandb.log(log)

        ckpt_payload = {
            'epoch': epoch,
            'fold': args.fold,
            'model_state': model.state_dict(),
            'cell_auc': m['AUC'],
            'smoothed_auc': smoothed,
            'patient_auc': p_auc,
            'bf_mean': bf_mean, 'bf_std': bf_std,
            'fl_mean': fl_mean, 'fl_std': fl_std,
            'fl_channels': fl_channels,
            'config': cfg,
        }
        if smoothed > best_smoothed:
            best_smoothed = smoothed
            torch.save(ckpt_payload, best_path)
            print(f"    -> new best smoothed AUC={smoothed:.4f}, saved to {best_path}")
            if use_wandb:
                wandb.summary['best_smoothed_auc'] = best_smoothed
                wandb.summary['best_epoch'] = epoch
        torch.save(ckpt_payload, last_path)

    print(f"fold {args.fold} done. best smoothed val AUC: {best_smoothed:.4f} "
          f"(best: {best_path}, last: {last_path})")
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
