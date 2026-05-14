import re
import random
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedKFold


PATIENT_RE = re.compile(r'pat_(\d+)')


def extract_patient_id(filename):
    m = PATIENT_RE.search(filename)
    if m is None:
        raise ValueError(f"Could not extract patient id from filename: {filename}")
    return m.group(1)


def get_patient_splits(filenames, labels, n_splits=3, seed=42):
    """Group cells by patient, then stratified k-fold over patients.

    Returns a list of (train_idx, val_idx) tuples indexing into `filenames`.
    Cells from the same patient never cross train/val boundaries.
    """
    patient_to_indices = defaultdict(list)
    for i, fname in enumerate(filenames):
        patient_to_indices[extract_patient_id(fname)].append(i)

    patient_ids = sorted(patient_to_indices.keys())
    patient_labels = np.array([labels[patient_to_indices[pid][0]] for pid in patient_ids])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (tr_pat, va_pat) in enumerate(skf.split(patient_ids, patient_labels)):
        train_pids = {patient_ids[i] for i in tr_pat}
        val_pids = {patient_ids[i] for i in va_pat}

        train_idx = [i for pid in train_pids for i in patient_to_indices[pid]]
        val_idx = [i for pid in val_pids for i in patient_to_indices[pid]]

        assert set(train_idx).isdisjoint(val_idx), "Patient leak between train and val"
        folds.append((np.array(train_idx), np.array(val_idx)))

        print(f"[fold {fold_idx}] train patients: {sorted(train_pids)}")
        print(f"[fold {fold_idx}] val   patients: {sorted(val_pids)}")
    return folds


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
