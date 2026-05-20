"""Inference: predict on the held-out test set and write submission.csv.

Loads each fold's best checkpoint, runs the test loader, averages sigmoid
probabilities across folds, and writes a CSV with columns matching
sampleSubmission.csv (Name, Diagnosis).

Run with:
    python -m src.inference --config configs/config.yaml --folds 0 1 2
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import OralCancerDataset
from src.models import GatedFusionModel


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


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


@torch.no_grad()
def predict(model, device, loader):
    model.eval()
    probs = []
    for bf, fl, _label, _fname in loader:
        bf = bf.to(device, non_blocking=True)
        fl = fl.to(device, non_blocking=True)
        logits = model(bf, fl).squeeze(1)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2],
                        help='Fold checkpoint indices to load and average.')
    parser.add_argument('--output', default=None,
                        help='Submission path. Defaults to <submission_dir>/submission.csv.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader, filenames = build_test_loader(cfg)
    print(f"test set: {len(filenames)} images")

    fold_probs = []
    for fold in args.folds:
        ckpt_path = os.path.join(cfg['output']['checkpoint_dir'], f'fold{fold}_best.pt')
        print(f"loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        model = GatedFusionModel(
            model_name=cfg['model']['backbone'],
            pretrained=False,  # weights are overwritten by the checkpoint
        ).to(device)
        model.load_state_dict(ckpt['model_state'])

        probs = predict(model, device, loader)
        fold_probs.append(probs)
        val_auc = ckpt.get('val_auc', float('nan'))
        print(f"  fold {fold}: val_auc={val_auc:.4f}, predicted {len(probs)} cells")

    avg = np.mean(np.stack(fold_probs, axis=0), axis=0)

    out_dir = cfg['output']['submission_dir']
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output or os.path.join(out_dir, 'submission.csv')
    pd.DataFrame({'Name': filenames, 'Diagnosis': avg}).to_csv(out_path, index=False)
    print(f"wrote {len(filenames)} predictions to {out_path}")


if __name__ == '__main__':
    main()
