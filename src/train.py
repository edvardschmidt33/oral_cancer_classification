"""Skeleton training entrypoint for the early-fusion baseline.

Only the wiring is in place — no full training loop yet. Run with:
    python -m src.train --config configs/config.yaml --fold 0 --smoke
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import OralCancerDataset
from src.augmentations import (
    build_bf_color_transform,
    build_fl_color_transform,
    build_shared_geo_transform,
)
from src.models import create_early_fusion_model
from src.utils import get_patient_splits, set_seed


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
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
    )
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--smoke', action='store_true',
                        help='Single forward pass on one batch, then exit.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['split']['seed'])

    labels_df = pd.read_csv(cfg['data']['labels_csv'])
    filenames = labels_df['Name'].tolist()
    labels = labels_df['Diagnosis'].astype(int).tolist()

    folds = get_patient_splits(
        filenames, labels,
        n_splits=cfg['split']['n_splits'],
        seed=cfg['split']['seed'],
    )
    train_idx, val_idx = folds[args.fold]
    print(f"fold {args.fold}: {len(train_idx)} train cells, {len(val_idx)} val cells")

    train_loader, val_loader = build_dataloaders(cfg, train_idx, val_idx, filenames, labels)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_early_fusion_model(
        model_name=cfg['model']['backbone'],
        pretrained=True,
        num_classes=1,
    ).to(device)

    if args.smoke:
        x, y, _ = next(iter(train_loader))
        x = x.to(device)
        with torch.no_grad():
            logits = model(x)
        print(f"smoke: input {tuple(x.shape)} -> logits {tuple(logits.shape)}")
        return

    raise NotImplementedError("Full training loop not implemented yet — Phase 1.5 TODO.")


if __name__ == '__main__':
    main()
