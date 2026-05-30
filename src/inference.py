"""Inference: predict on the held-out test set and write submission.csv.

Loads each fold's best checkpoint (which embeds the per-modality normalization
stats and fl_channels), runs the test loader with optional 4-view TTA
(identity + h-flip + v-flip + h+v-flip), averages sigmoid probabilities across
folds, and writes a CSV matching sampleSubmission.csv (Name, Diagnosis).

Run with:
    python -m src.inference --config configs/config_ef.yaml --folds 0 1
    python -m src.inference --config configs/config_ef.yaml --folds 0 1 --ckpt last
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import OralCancerDataset
from src.models import EarlyFusionConcatModel


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_test_loader(cfg, bf_mean, bf_std, fl_mean, fl_std, fl_channels):
    sample = pd.read_csv(cfg['data']['sample_submission_csv'])
    filenames = sample['Name'].tolist()
    ds = OralCancerDataset(
        filenames=filenames,
        labels=None,
        bf_dir=cfg['data']['bf_test_dir'],
        fl_dir=cfg['data']['fl_test_dir'],
        size=cfg['model']['img_size'],
        crop_size=cfg['model'].get('crop_size'),
        bf_mean=bf_mean, bf_std=bf_std,
        fl_mean=fl_mean, fl_std=fl_std,
        fl_channels=fl_channels,
        bf_color=None, fl_color=None, geo_transform=None,
        fl_tensor_blur=False,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
    )
    return loader, filenames


def tta_views(x):
    """4-view TTA on the already-normalised concatenated tensor: identity,
    h-flip, v-flip, both. Same set as EF_convnext_best.py."""
    return [x,
            torch.flip(x, dims=[3]),
            torch.flip(x, dims=[2]),
            torch.flip(x, dims=[2, 3])]


@torch.no_grad()
def predict(model, device, loader, use_tta, use_amp):
    model.eval()
    out = []
    for x, _label, _name in loader:
        x = x.to(device, non_blocking=True)
        views = tta_views(x) if use_tta else [x]
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            probs = torch.zeros(x.size(0), device=device)
            for v in views:
                probs += torch.sigmoid(model(v).squeeze(1).float())
            probs /= len(views)
        out.append(probs.cpu().numpy())
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_ef.yaml')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1],
                        help='Fold checkpoint indices to load and average.')
    parser.add_argument('--ckpt', choices=['best', 'last'], default='best',
                        help='Which per-fold checkpoint to load.')
    parser.add_argument('--output', default=None,
                        help='Submission path. Defaults to <submission_dir>/submission_<ckpt>[_tta].csv.')
    parser.add_argument('--tta', dest='tta', action='store_true', default=None,
                        help='Force TTA on (overrides config).')
    parser.add_argument('--no-tta', dest='tta', action='store_false',
                        help='Force TTA off (overrides config).')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    use_tta = args.tta if args.tta is not None else bool(cfg.get('inference', {}).get('tta', False))
    print(f"TTA: {'on (4-view: flips)' if use_tta else 'off'}")

    run_name = cfg['output'].get('run_name')
    prefix = f'{run_name}_' if run_name else ''
    out_dir = cfg['output']['submission_dir']
    os.makedirs(out_dir, exist_ok=True)

    fold_probs = []
    filenames = None
    for fold in args.folds:
        ckpt_path = os.path.join(cfg['output']['checkpoint_dir'],
                                 f'{prefix}fold{fold}_{args.ckpt}.pt')
        print(f"loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        bf_mean = ckpt['bf_mean']
        bf_std = ckpt['bf_std']
        fl_mean = ckpt['fl_mean']
        fl_std = ckpt['fl_std']
        fl_channels = ckpt['fl_channels']
        total_channels = 3 + fl_channels
        eff_img_size = cfg['model'].get('crop_size') or cfg['model']['img_size']

        loader, filenames = build_test_loader(
            cfg, bf_mean, bf_std, fl_mean, fl_std, fl_channels,
        )

        model = EarlyFusionConcatModel(
            backbone_name=cfg['model']['backbone'],
            pretrained=False,
            total_channels=total_channels,
            dropout=cfg['training'].get('dropout', 0.15),
            img_size=eff_img_size,
            upsample_to=cfg['model'].get('upsample_to'),
        ).to(device)
        model.load_state_dict(ckpt['model_state'])

        probs = predict(model, device, loader, use_tta=use_tta, use_amp=use_amp)
        fold_probs.append(probs)
        sauc = ckpt.get('smoothed_auc', float('nan'))
        cauc = ckpt.get('cell_auc', float('nan'))
        print(f"  fold {fold}: cell_auc={cauc:.4f}, smoothed_auc={sauc:.4f}, "
              f"predicted {len(probs)} cells")

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    avg = np.mean(np.stack(fold_probs, axis=0), axis=0)

    suffix = f'_{args.ckpt}' + ('_tta' if use_tta else '')
    default_name = f'{prefix}submission{suffix}.csv' if prefix else f'submission{suffix}.csv'
    out_path = args.output or os.path.join(out_dir, default_name)
    pd.DataFrame({'Name': filenames, 'Diagnosis': avg}).to_csv(out_path, index=False)
    print(f"wrote {len(filenames)} predictions to {out_path}")
    print(f"score stats: min={avg.min():.4f} max={avg.max():.4f} "
          f"mean={avg.mean():.4f} frac>0.5={(avg > 0.5).mean():.4f}")


if __name__ == '__main__':
    main()
