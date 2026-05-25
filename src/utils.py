import re
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


PATIENT_RE = re.compile(r'pat_(\d+)')


def extract_patient_id(filename):
    m = PATIENT_RE.search(filename)
    if m is None:
        raise ValueError(f"Could not extract patient id from filename: {filename}")
    return m.group(1)


def get_patient_splits(filenames, labels, n_folds=3, seed=42):
    """StratifiedGroupKFold over cells, grouping by patient.

    - Groups (patients) never cross the train/val boundary.
    - Class balance is kept as even as possible per fold.

    Returns list of (train_indices, val_indices) tuples indexing into `filenames`.
    """
    df = pd.DataFrame({
        'idx': np.arange(len(filenames)),
        'patient': [extract_patient_id(f) for f in filenames],
        'label': np.asarray(labels, dtype=int),
    })

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_col = np.full(len(df), -1, dtype=int)
    splits = list(sgkf.split(df['idx'], df['label'], df['patient']))
    for fold_idx, (_, val_idx) in enumerate(splits):
        fold_col[val_idx] = fold_idx
    assert (fold_col >= 0).all(), "Some rows were not assigned a fold"
    df['fold'] = fold_col

    # --- diagnostics ---
    print(f"Total rows: {len(df)}")
    print(f"Total patients: {df['patient'].nunique()}")
    print(f"Patients per fold: {df.groupby('fold')['patient'].nunique().to_dict()}")
    print(f"Cancer rate (cells) per fold: "
          f"{df.groupby('fold')['label'].mean().round(3).to_dict()}")

    pat = df.groupby('patient').agg(fold=('fold', 'first'), label=('label', 'first'))
    print(f"Cancer patients per fold: "
          f"{pat.groupby('fold')['label'].sum().astype(int).to_dict()}")
    print(f"Total patients per fold:  "
          f"{pat.groupby('fold')['label'].count().to_dict()}")
    for f in range(n_folds):
        sub = pat[pat['fold'] == f]
        if sub['label'].nunique() < 2:
            print(f"  WARNING: fold {f} has only one class at patient level -- "
                  f"its AUC will be meaningless.")

    folds = []
    for fold_idx in range(n_folds):
        val_rows = df[df['fold'] == fold_idx]
        train_rows = df[df['fold'] != fold_idx]
        train_idx = train_rows['idx'].to_numpy()
        val_idx = val_rows['idx'].to_numpy()

        train_pids = sorted(train_rows['patient'].unique().tolist())
        val_pids = sorted(val_rows['patient'].unique().tolist())
        assert set(train_pids).isdisjoint(val_pids), "Patient leak between train and val"

        print(f"[fold {fold_idx}] train patients: {train_pids}")
        print(f"[fold {fold_idx}] val   patients: {val_pids}")
        folds.append((train_idx, val_idx))

    return folds


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
