"""Inference: predict on the held-out test set and write submission.csv.

Loads each fold's best checkpoint, runs the test loader, averages sigmoid
probabilities across folds, and writes a CSV with columns matching
sampleSubmission.csv (Name, Diagnosis).

Run with:
    python -m src.inference --config configs/config.yaml --folds 0 1 2
    python -m src.inference --config configs/config.yaml --folds 0 1 2 --ckpt last
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import OralCancerDataset
from src.models import (
    CrossAttentionFusionModel,
    EarlyFusionConcatModel,
    GatedFusionModel,
)


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_model(cfg, pretrained=False):
    model_type = cfg['model'].get('type', 'gated')
    if model_type == 'cross_attention':
        return CrossAttentionFusionModel(
            backbone_name=cfg['model']['backbone'],
            pretrained=pretrained,
            proj_dim=cfg['model'].get('proj_dim', 64),
            num_heads=cfg['model'].get('num_heads', 4),
            window_size=cfg['model'].get('window_size', 3),
        )
    if model_type == 'early_fusion_concat':
        return EarlyFusionConcatModel(
            backbone_name=cfg['model']['backbone'],
            pretrained=pretrained,
        )
    if model_type == 'gated':
        return GatedFusionModel(
            model_name=cfg['model']['backbone'],
            pretrained=pretrained,
        )
    raise ValueError(f"unknown model.type: {model_type!r}")


def build_test_loader(cfg):
    """Filenames come from sampleSubmission.csv so the output ordering matches
    the expected submission ordering exactly."""
    sample = pd.read_csv(cfg['data']['sample_submission_csv'])
    filenames = sample['Name'].tolist()
    ds = OralCancerDataset(
        filenames=filenames,
        labels=None,
        bf_dir=cfg['data']['bf_test_dir'],
        fl_dir=cfg['data']['fl_test_dir'],
    )
    loader = DataLoader(
        ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
    )
    return loader, filenames


def _d4_views(x):
    """Generate the 8 dihedral-group (D4) views of a [B, C, H, W] tensor:
    4 rotations (0/90/180/270) × {identity, horizontal flip}."""
    views = []
    for k in range(4):
        rot = torch.rot90(x, k, dims=(2, 3))
        views.append(rot)
        views.append(torch.flip(rot, dims=(3,)))
    return views


@torch.no_grad()
def predict(model, device, loader, use_tta=False):
    """If `use_tta`, average sigmoid probs over the 8 D4 views (BF and FL
    transformed in lockstep to preserve their pairing)."""
    model.eval()
    probs = []
    for bf, fl, _label, _fname in loader:
        bf = bf.to(device, non_blocking=True)
        fl = fl.to(device, non_blocking=True)

        if use_tta:
            bf_views = _d4_views(bf)
            fl_views = _d4_views(fl)
            acc = None
            for b, f in zip(bf_views, fl_views):
                p = torch.sigmoid(model(b, f).squeeze(1))
                acc = p if acc is None else acc + p
            batch_probs = acc / len(bf_views)
        else:
            batch_probs = torch.sigmoid(model(bf, fl).squeeze(1))

        probs.append(batch_probs.float().cpu().numpy())
    return np.concatenate(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2],
                        help='Fold checkpoint indices to load and average.')
    parser.add_argument('--ckpt', choices=['best', 'last'], default='best',
                        help='Which per-fold checkpoint to load: best-AUC or final-epoch.')
    parser.add_argument('--output', default=None,
                        help='Submission path. Defaults to <submission_dir>/submission_<ckpt>[_tta].csv.')
    parser.add_argument('--tta', dest='tta', action='store_true', default=None,
                        help='Force TTA on (overrides config).')
    parser.add_argument('--no-tta', dest='tta', action='store_false',
                        help='Force TTA off (overrides config).')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_tta = args.tta if args.tta is not None else bool(cfg.get('inference', {}).get('tta', False))
    print(f"TTA: {'on (8-view D4)' if use_tta else 'off'}")

    loader, filenames = build_test_loader(cfg)
    print(f"test set: {len(filenames)} images")

    fold_probs = []
    for fold in args.folds:
        ckpt_path = os.path.join(cfg['output']['checkpoint_dir'], f'fold{fold}_{args.ckpt}.pt')
        print(f"loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        model = build_model(cfg, pretrained=False).to(device)
        model.load_state_dict(ckpt['model_state'])

        probs = predict(model, device, loader, use_tta=use_tta)
        fold_probs.append(probs)
        val_auc = ckpt.get('val_auc', float('nan'))
        print(f"  fold {fold}: val_auc={val_auc:.4f}, predicted {len(probs)} cells")

    avg = np.mean(np.stack(fold_probs, axis=0), axis=0)

    out_dir = cfg['output']['submission_dir']
    os.makedirs(out_dir, exist_ok=True)
    suffix = f'_{args.ckpt}' + ('_tta' if use_tta else '')
    out_path = args.output or os.path.join(out_dir, f'submission{suffix}.csv')
    pd.DataFrame({'Name': filenames, 'Diagnosis': avg}).to_csv(out_path, index=False)
    print(f"wrote {len(filenames)} predictions to {out_path}")


if __name__ == '__main__':
    main()
