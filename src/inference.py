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
from src.utils import compute_norm_stats, get_patient_splits, probe_fl_channels, set_seed


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def unwrap_checkpoint(ckpt):
    """Accept either a wrapped payload ({'model_state': ..., 'bf_mean': ...})
    or a raw state_dict (just weights). Returns (payload_dict, model_state)."""
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        return ckpt, ckpt['model_state']
    return {}, ckpt


def resolve_fold_stats(payload, cfg, fold, seed):
    """Read BF/FL norm stats and fl_channels from the checkpoint payload if
    present; otherwise recompute them from the training data using the same
    per-fold split as `train.py`. Lets inference work on raw weight dumps that
    don't embed the stats (e.g. older checkpoints uploaded from Kaggle)."""
    fl_channels = payload.get('fl_channels')
    if fl_channels is None:
        fl_channels = probe_fl_channels(cfg['data']['fl_train_dir'])
        print(f"  fl_channels not in checkpoint -> probed: {fl_channels}")

    stat_keys = ('bf_mean', 'bf_std', 'fl_mean', 'fl_std')
    if all(k in payload for k in stat_keys):
        return (payload['bf_mean'], payload['bf_std'],
                payload['fl_mean'], payload['fl_std'], fl_channels)

    print("  normalization stats missing from checkpoint -> recomputing from labels_csv")
    print(f"  (using current config split: n_folds={cfg['split']['n_folds']}, seed={seed})")
    set_seed(seed)
    labels_df = pd.read_csv(cfg['data']['labels_csv'])
    filenames = labels_df['Name'].tolist()
    labels = labels_df['Diagnosis'].astype(int).tolist()
    folds = get_patient_splits(
        filenames, labels,
        n_folds=cfg['split']['n_folds'],
        seed=seed,
    )
    train_idx, _ = folds[fold]
    train_names = [filenames[i] for i in train_idx]
    bf_mean, bf_std = compute_norm_stats(
        cfg['data']['bf_train_dir'], train_names,
        modality='BF', sample_size=2000, seed=seed,
    )
    fl_mean, fl_std = compute_norm_stats(
        cfg['data']['fl_train_dir'], train_names,
        modality='FL', fl_channels=fl_channels,
        sample_size=2000, seed=seed,
    )
    return bf_mean, bf_std, fl_mean, fl_std, fl_channels


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
        payload, model_state = unwrap_checkpoint(ckpt)
        bf_mean, bf_std, fl_mean, fl_std, fl_channels = resolve_fold_stats(
            payload, cfg, fold, seed=cfg['split']['seed'],
        )
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
        model.load_state_dict(model_state)

        probs = predict(model, device, loader, use_tta=use_tta, use_amp=use_amp)
        fold_probs.append(probs)
        sauc = payload.get('smoothed_auc', float('nan'))
        cauc = payload.get('cell_auc', float('nan'))
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
