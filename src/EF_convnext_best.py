import os
from pathlib import Path

# Kaggle mounts every attached dataset under /kaggle/input/<slug>/
for entry in sorted(Path('/kaggle/input').iterdir()):
    print(entry)
    for sub in sorted(entry.iterdir())[:10]:
        print('  ', sub)

# >>> EDIT THIS LINE to point at your competition dataset folder <<<
DATA_ROOT = Path('/kaggle/input/competitions/multimodal-cancer-classification-challenge-2026')

# Quick sanity check
assert DATA_ROOT.exists(), f'{DATA_ROOT} not found - fix the slug above'
assert (DATA_ROOT / 'train.csv').exists(), 'train.csv missing'
assert (DATA_ROOT / 'BF' / 'train').exists(), 'BF/train missing'
assert (DATA_ROOT / 'FL' / 'train').exists(), 'FL/train missing'
print('Data layout OK.')
print('Sample BF file:', next((DATA_ROOT / 'BF' / 'train').iterdir()))

from PIL import Image
import numpy as np

def probe_modality(folder, n=8):
    modes, bands, sizes = set(), set(), set()
    for p in list(sorted(folder.iterdir()))[:n]:
        im = Image.open(p)
        modes.add(im.mode)
        bands.add(im.getbands())
        sizes.add(im.size)
    return modes, bands, sizes

bf_modes, bf_bands, bf_sizes = probe_modality(DATA_ROOT / 'BF' / 'train')
fl_modes, fl_bands, fl_sizes = probe_modality(DATA_ROOT / 'FL' / 'train')
print('BF  modes:', bf_modes, '| bands:', bf_bands, '| sizes:', bf_sizes)
print('FL  modes:', fl_modes, '| bands:', fl_bands, '| sizes:', fl_sizes)

# Decide FL channel count from what the files actually contain.
_fl_band_lens = {len(b) for b in fl_bands}
assert len(_fl_band_lens) == 1, f'Inconsistent FL channel counts across files: {fl_bands}'
FL_CHANNELS = _fl_band_lens.pop()
assert FL_CHANNELS in (3, 4), f'Unexpected FL channel count: {FL_CHANNELS}'
print(f'\n>>> FL_CHANNELS = {FL_CHANNELS}'
      + ('  (4-channel: autofluorescence channel WILL be kept)' if FL_CHANNELS == 4
         else '  (3-channel RGB)'))


import re
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from types import SimpleNamespace

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)
if DEVICE.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

# All configuration in one place. Edit and re-run.
args = SimpleNamespace(
    data_root   = DATA_ROOT,
    out_dir     = Path('/kaggle/working'),
    mode        = 'MM',                         # 'BF', 'FL', or 'MM'
    fusion      = 'E',                          # 'E' (early) or 'L' (late) -- only for MM
    arch        = 'convnextv2_tiny.fcmae_ft_in22k_in1k',
    fl_channels = FL_CHANNELS,                  # auto-detected in Cell 5 -- do not hardcode
    size        = 128,                          # native resolution; 224 = ImageNet default
    epochs      = 15,                           # v3: was 10; v2 fold 1 was still climbing at ep9
    batch_size  = 128,                          # drop to 64 if you OOM
    lr          = 8e-5,
    wd          = 0.15,                         # v3: was 0.25; mild L2, not a wall
    dropout     = 0.15,                         # v3: was 0.3; light dropout before the head
    n_folds     = 2,                            # only 12 patients -- 2 folds is the right call
    use_mixup   = True,
    mixup_alpha = 0.4,                          # v3: was 0.6; back to gentler mixup
    use_amp     = True,
    use_tta     = True,                         # test-time augmentation at inference
    num_workers = 4,                            # v2: was 2; heavier augmentation needs more workers
    seed        = 0,
)
args.out_dir.mkdir(exist_ok=True, parents=True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

PAT_RE = re.compile(r'pat_(\d+)')

def patient_id(name: str) -> str:
    m = PAT_RE.search(name)
    if m is None:
        raise ValueError(f'No patient ID in filename: {name}')
    return m.group(1)

def make_folds(train_csv, n_folds=3, seed=0):
    df = pd.read_csv(train_csv)
    df['patient'] = df['Name'].map(patient_id)
    df['fold'] = -1
    # StratifiedGroupKFold: groups (patients) never split, classes kept balanced per fold.
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (_, val_idx) in enumerate(
            sgkf.split(df, df['Diagnosis'], df['patient'])):
        df.loc[val_idx, 'fold'] = fold_idx
    assert (df['fold'] >= 0).all(), 'Some rows were not assigned a fold'
    return df

full_df = make_folds(args.data_root / 'train.csv', n_folds=args.n_folds, seed=args.seed)
print(f'Total rows: {len(full_df)}')
print(f'Total patients: {full_df["patient"].nunique()}')
print(f'Patients per fold: {full_df.groupby("fold")["patient"].nunique().to_dict()}')
print(f'Cancer rate (cells) per fold: '
      f'{full_df.groupby("fold")["Diagnosis"].mean().round(3).to_dict()}')

# Patient-level cancer balance per fold -- watch for a degenerate fold.
_pat = full_df.groupby('patient').agg(fold=('fold','first'), label=('Diagnosis','first'))
print(f'Cancer patients per fold: '
      f'{_pat.groupby("fold")["label"].sum().astype(int).to_dict()}')
print(f'Total patients per fold:  '
      f'{_pat.groupby("fold")["label"].count().to_dict()}')
for f in range(args.n_folds):
    sub = _pat[_pat.fold == f]
    if sub['label'].nunique() < 2:
        print(f'  WARNING: fold {f} has only one class at patient level -- '
              f'its AUC will be meaningless.')
full_df.head()

# Class weights from the actual training distribution (replaces the paper's hardcoded
# 0.78/0.22, which were specific to their dataset).
pos_rate = full_df['Diagnosis'].mean()
neg_rate = 1 - pos_rate
# Inverse-frequency: rare class weighted up.
W_POS = float(neg_rate)
W_NEG = float(pos_rate)
print(f'Cancer rate (cells): {pos_rate:.4f} | Healthy rate: {neg_rate:.4f}')
print(f'Class weights -> W_POS={W_POS:.4f}, W_NEG={W_NEG:.4f}')

def compute_norm_stats(data_root, names, subdir='train', modality='BF',
                       fl_channels=3, sample_size=2000):
    rng = np.random.default_rng(0)
    sample = rng.choice(names, size=min(sample_size, len(names)), replace=False)
    folder = 'BF' if modality == 'BF' else 'FL'
    n_channels = 3 if modality == 'BF' else fl_channels
    target_mode = 'RGBA' if (modality == 'FL' and fl_channels == 4) else 'RGB'
    sums = np.zeros(n_channels, dtype=np.float64)
    sqs = np.zeros(n_channels, dtype=np.float64)
    count = 0
    for nm in tqdm(sample, desc=f'Norm stats {modality}'):
        path = Path(data_root) / folder / subdir / nm
        arr = np.asarray(Image.open(path).convert(target_mode), dtype=np.float32) / 255.0
        sums += arr.reshape(-1, n_channels).sum(axis=0)
        sqs += (arr.reshape(-1, n_channels) ** 2).sum(axis=0)
        count += arr.shape[0] * arr.shape[1]
    mean = sums / count
    std = np.sqrt(np.maximum(sqs / count - mean ** 2, 1e-12))
    return tuple(mean.tolist()), tuple(std.tolist())

BF_MEAN, BF_STD = compute_norm_stats(args.data_root, full_df['Name'].tolist(),
                                     modality='BF')
FL_MEAN, FL_STD = compute_norm_stats(args.data_root, full_df['Name'].tolist(),
                                     modality='FL', fl_channels=args.fl_channels)
print(f'BF mean: {BF_MEAN}\nBF std:  {BF_STD}')
print(f'FL mean: {FL_MEAN}\nFL std:  {FL_STD}')
assert len(FL_MEAN) == args.fl_channels, 'FL stat channel count mismatch'

class KaggleOCDataset(Dataset):
    def __init__(self, data_root, df, mode='MM', split='train', size=128,
                 bf_mean=None, bf_std=None, fl_mean=None, fl_std=None,
                 fl_channels=3, bf_subdir='train', fl_subdir='train'):
        assert split in ('train', 'val', 'test'), f'bad split: {split}'
        self.root = Path(data_root)
        self.df = df.reset_index(drop=True)
        self.mode, self.split, self.size = mode, split, size
        self.fl_channels = fl_channels
        self.bf_subdir, self.fl_subdir = bf_subdir, fl_subdir
        self.fl_mode = 'RGBA' if fl_channels == 4 else 'RGB'

        if split == 'train':
            # BF photometric: stronger stain/illumination jitter (v2) to break the
            # patient-identity shortcut -- staining/lighting is what separates the 12 patients.
            self.tf_bf = transforms.Compose([
                transforms.RandomChoice(
                    [transforms.RandomPosterize(3, p=1.0),
                     transforms.GaussianBlur(5, 1.5),
                     transforms.RandomSolarize(100, p=1.0)],
                    p=[0.4, 0.2, 0.4]),
                transforms.ColorJitter(brightness=0.6, contrast=0.4,
                                       saturation=0.4, hue=0.25),
                transforms.Resize((size, size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(bf_mean, bf_std),
            ])
            # FL photometric: blur deliberately left UNCHANGED -- sigma up to 3.2 is already
            # near the range that destroys fine nuclear texture (the MAC signal).
            # ColorJitter on a 4-channel image is invalid; only jitter the RGB part.
            fl_aug = ([transforms.ColorJitter(brightness=0.8, contrast=0.8,
                                              saturation=0.8, hue=0.5)]
                      if fl_channels == 3 else [])
            self.tf_fl = transforms.Compose(fl_aug + [
                transforms.Resize((size, size), antialias=True),
                transforms.ToTensor(),
                transforms.GaussianBlur(5, sigma=(0.3, 3.2)),
                transforms.Normalize(fl_mean, fl_std),
            ])
            # Geometric stage runs on the CONCATENATED tensor, so BF and FL receive the
            # SAME spatial transform and stay aligned. v2 adds rotation, affine, erasing.
            # Cells have no canonical orientation, so these are label-preserving.
            self.tf_geom = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08),
                                        scale=(0.9, 1.1), shear=8),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
            ])
        else:  # 'val' or 'test'
            self.tf_bf = transforms.Compose([
                transforms.Resize((size, size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(bf_mean, bf_std),
            ])
            self.tf_fl = transforms.Compose([
                transforms.Resize((size, size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(fl_mean, fl_std),
            ])
            self.tf_geom = None

    def __len__(self):
        return len(self.df)

    def _load_bf(self, name):
        return Image.open(self.root / 'BF' / self.bf_subdir / name).convert('RGB')

    def _load_fl(self, name):
        return Image.open(self.root / 'FL' / self.fl_subdir / name).convert(self.fl_mode)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row['Name']
        label = float(row['Diagnosis']) if 'Diagnosis' in row else -1.0
        bf = fl = None
        if self.mode in ('BF', 'MM'):
            bf = self.tf_bf(self._load_bf(name))
        if self.mode in ('FL', 'MM'):
            fl = self.tf_fl(self._load_fl(name))
        if self.mode == 'MM':
            x = torch.cat([bf, fl], dim=0)
        else:
            x = bf if self.mode == 'BF' else fl
        if self.tf_geom is not None:
            x = self.tf_geom(x)
        return {'x': x, 'label': label, 'name': name}


import timm

def make_backbone(in_channels, out_dim=1,
                  model_name='convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True):
    """timm model with the stem patched for `in_channels`. Returns (model, feature_dim)."""
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim)

    # ConvNeXt stem is Conv2d(3, dim, k=4, s=4) then LayerNorm. Assert before patching
    # so a future timm restructure fails loudly instead of silently.
    assert hasattr(model, 'stem') and isinstance(model.stem[0], nn.Conv2d), \
        f'Unexpected stem layout for {model_name}: {getattr(model, "stem", None)}'
    old_stem = model.stem[0]
    new_stem = nn.Conv2d(in_channels, old_stem.out_channels,
                         kernel_size=old_stem.kernel_size,
                         stride=old_stem.stride,
                         padding=old_stem.padding,
                         bias=old_stem.bias is not None)
    with torch.no_grad():
        if in_channels >= 3:
            new_stem.weight[:, :3] = old_stem.weight
            if in_channels > 3:
                new_stem.weight[:, 3:] = old_stem.weight.mean(
                    dim=1, keepdim=True).repeat(1, in_channels - 3, 1, 1)
        else:
            new_stem.weight.copy_(old_stem.weight[:, :in_channels])
        if old_stem.bias is not None:
            new_stem.bias.copy_(old_stem.bias)
    model.stem[0] = new_stem
    return model, model.num_features


class SingleModalNet(nn.Module):
    def __init__(self, in_channels, model_name, dropout=0.3):
        super().__init__()
        # v2: headless backbone + explicit head with dropout before the classifier.
        self.backbone, fdim = make_backbone(in_channels, out_dim=0, model_name=model_name)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fdim, 1))
    def forward(self, x):
        return self.head(self.backbone(x))


class EarlyFusionNet(nn.Module):
    def __init__(self, total_channels, model_name, dropout=0.3):
        super().__init__()
        # v2: headless backbone + explicit head with dropout before the classifier.
        self.backbone, fdim = make_backbone(total_channels, out_dim=0, model_name=model_name)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fdim, 1))
    def forward(self, x):
        return self.head(self.backbone(x))


class LateFusionNet(nn.Module):
    def __init__(self, c_bf, c_fl, model_name, dropout=0.3):
        super().__init__()
        self.bf, fdim = make_backbone(c_bf, out_dim=0, model_name=model_name)
        self.fl, _    = make_backbone(c_fl, out_dim=0, model_name=model_name)
        self.c_bf = c_bf
        self.head = nn.Sequential(
            nn.Linear(fdim * 2, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, 1))
    def forward(self, x):
        bf, fl = x[:, :self.c_bf], x[:, self.c_bf:]
        return self.head(torch.cat([self.bf(bf), self.fl(fl)], dim=1))


def build_model(args):
    dp = getattr(args, 'dropout', 0.3)
    if args.mode == 'BF':
        return SingleModalNet(3, args.arch, dropout=dp)
    if args.mode == 'FL':
        return SingleModalNet(args.fl_channels, args.arch, dropout=dp)
    total = 3 + args.fl_channels
    if args.fusion == 'E':
        return EarlyFusionNet(total, args.arch, dropout=dp)
    if args.fusion == 'L':
        return LateFusionNet(3, args.fl_channels, args.arch, dropout=dp)
    raise ValueError('Intermediate fusion needs CAFNet/MMTM/HcCNN from the original repo.')


# Smoke test
_m = build_model(args).to(DEVICE)
_in_c = (3 + args.fl_channels if args.mode == 'MM'
         else 3 if args.mode == 'BF' else args.fl_channels)
_x = torch.randn(2, _in_c, args.size, args.size).to(DEVICE)
print('Forward pass shape:', _m(_x).shape)
print(f'Total params: {sum(p.numel() for p in _m.parameters()):,}')
del _m, _x
torch.cuda.empty_cache()

def mixup(x, y, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def weighted_bce(logits, y, w_pos=None, w_neg=None):
    w_pos = W_POS if w_pos is None else w_pos
    w_neg = W_NEG if w_neg is None else w_neg
    w = torch.where(y > 0.5, torch.full_like(y, w_pos), torch.full_like(y, w_neg))
    return F.binary_cross_entropy_with_logits(logits, y, weight=w)

def run_epoch(model, loader, optimizer, scaler, train_mode,
              use_mixup, use_amp, mixup_alpha=0.4):
    model.train(train_mode)
    all_y, all_s, all_names = [], [], []
    total_loss = 0.0
    pbar = tqdm(loader, desc='train' if train_mode else 'eval', leave=False)
    for batch in pbar:
        x = batch['x'].to(DEVICE, non_blocking=True)
        y = batch['label'].to(DEVICE, non_blocking=True).float()
        if train_mode and use_mixup:
            x, y_a, y_b, lam = mixup(x, y, alpha=mixup_alpha)
        with torch.set_grad_enabled(train_mode):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x).squeeze(1)
                if train_mode and use_mixup:
                    loss = (lam * weighted_bce(logits, y_a)
                            + (1 - lam) * weighted_bce(logits, y_b))
                else:
                    loss = weighted_bce(logits, y)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * x.size(0)
        all_s.extend(torch.sigmoid(logits.float()).detach().cpu().tolist())
        all_y.extend(y.detach().cpu().tolist())
        all_names.extend(batch['name'])
    return (total_loss / len(loader.dataset),
            np.array(all_y), np.array(all_s), all_names)

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
    """Aggregate cell scores to patient level (mean) and compute AUC.
    This matches the weak-label structure of the task."""
    d = pd.DataFrame({'patient': [patient_id(n) for n in names], 'y': y, 's': s})
    g = d.groupby('patient').agg(y=('y', 'first'), s=('s', 'mean'))
    if g['y'].nunique() < 2:
        return float('nan')
    return metrics.roc_auc_score(g['y'], g['s'])


fold_oof   = np.zeros(len(full_df))
fold_summary = []

for fold in range(args.n_folds):
    print(f'\n========= FOLD {fold} =========')
    tr_df = full_df[full_df.fold != fold]
    va_df = full_df[full_df.fold == fold]

    common = dict(data_root=args.data_root, mode=args.mode, size=args.size,
                  bf_mean=BF_MEAN, bf_std=BF_STD, fl_mean=FL_MEAN, fl_std=FL_STD,
                  fl_channels=args.fl_channels)
    train_ds = KaggleOCDataset(df=tr_df, split='train', **common)
    val_ds   = KaggleOCDataset(df=va_df, split='val',   **common)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=False, prefetch_factor=4)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=False, prefetch_factor=4)

    model  = build_model(args).to(DEVICE)
    opt    = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sch    = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    best_smoothed = -1.0
    auc_history = []          # raw per-epoch validation cell-AUC
    ckpt_path = args.out_dir / f'fold{fold}.pt'
    for epoch in range(args.epochs):
        tr_loss, _, _, _ = run_epoch(model, train_dl, opt, scaler, train_mode=True,
                                     use_mixup=args.use_mixup, use_amp=args.use_amp,
                                     mixup_alpha=args.mixup_alpha)
        sch.step()
        va_loss, y, s, names = run_epoch(model, val_dl, opt, scaler, train_mode=False,
                                         use_mixup=False, use_amp=args.use_amp)
        m = cell_metrics(y, s)
        p_auc = patient_auc(names, y, s)
        auc_history.append(m['AUC'])
        # v3: select on a 3-epoch moving average of validation AUC, not the single
        # highest epoch -- the raw curve is noisy and picking its peak is optimistic.
        smoothed = float(np.mean(auc_history[-3:]))
        print(f'  epoch {epoch:02d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} '
              f'| cell_AUC={m["AUC"]:.4f} (sm {smoothed:.4f}) | pat_AUC={p_auc:.4f} '
              f'| F1={m["F1"]:.4f} Acc={m["Acc"]:.4f}')
        if smoothed > best_smoothed:
            best_smoothed = smoothed
            torch.save(model.state_dict(), ckpt_path)

    # OOF predictions from the saved (smoothed-best) checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    _, y, s, names = run_epoch(model, val_dl, opt, scaler, train_mode=False,
                               use_mixup=False, use_amp=args.use_amp)
    fold_oof[va_df.index] = s
    f_cauc = cell_metrics(y, s)['AUC']      # actual cell-AUC of the saved checkpoint
    f_pauc = patient_auc(names, y, s)
    fold_summary.append({'fold': fold, 'ckpt_cell_AUC': f_cauc,
                         'smoothed_AUC': best_smoothed, 'pat_AUC': f_pauc})
    print(f'  Fold {fold} saved-ckpt cell AUC: {f_cauc:.4f} '
          f'(smoothed {best_smoothed:.4f}) | patient AUC: {f_pauc:.4f}')
    del model, opt, sch, scaler, train_dl, val_dl
    torch.cuda.empty_cache()

print('\n=== Per-fold summary ===')
print(pd.DataFrame(fold_summary).to_string(index=False))

oof_cell = cell_metrics(full_df['Diagnosis'].values, fold_oof)
oof_pat  = patient_auc(full_df['Name'].tolist(),
                       full_df['Diagnosis'].values, fold_oof)
print('\n=== Out-of-fold metrics ===')
print(f'  Cell-level AUC:    {oof_cell["AUC"]:.4f}')
print(f'  Patient-level AUC: {oof_pat:.4f}   <-- closest to the competition target')
for k in ('F1', 'Acc', 'Prec', 'Rec'):
    print(f'  {k}: {oof_cell[k]:.4f}')


best_f1, best_thr = 0.0, 0.5
for thr in np.arange(0.1, 0.9, 0.02):
    f1 = metrics.f1_score(full_df['Diagnosis'].values,
                          (fold_oof > thr).astype(int), zero_division=0)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr
print(f'Best OOF F1 {best_f1:.4f} at threshold {best_thr:.2f}')


best_f1, best_thr = 0.0, 0.5
for thr in np.arange(0.1, 0.9, 0.02):
    f1 = metrics.f1_score(full_df['Diagnosis'].values,
                          (fold_oof > thr).astype(int), zero_division=0)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr
print(f'Best OOF F1 {best_f1:.4f} at threshold {best_thr:.2f}')

sub = pd.read_csv(args.data_root / 'sampleSubmission.csv')
print('Submission shape:', sub.shape, '| Columns:', list(sub.columns))
sub.head()

# TTA transforms operate on the already-normalised tensor batch.
def tta_views(x):
    """Return a list of augmented copies of batch x. Flips are label-preserving
    for cells (no canonical orientation)."""
    return [x,
            torch.flip(x, dims=[3]),          # horizontal
            torch.flip(x, dims=[2]),          # vertical
            torch.flip(x, dims=[2, 3])]       # both

test_df = sub.copy()
if 'Diagnosis' not in test_df.columns:
    test_df['Diagnosis'] = 0

test_ds = KaggleOCDataset(
    data_root=args.data_root, df=test_df, mode=args.mode, split='test', size=args.size,
    bf_mean=BF_MEAN, bf_std=BF_STD, fl_mean=FL_MEAN, fl_std=FL_STD,
    fl_channels=args.fl_channels, bf_subdir='test', fl_subdir='test')
test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                     num_workers=args.num_workers, pin_memory=True)

test_scores = np.zeros(len(test_df))
for fold in range(args.n_folds):
    model = build_model(args).to(DEVICE)
    model.load_state_dict(torch.load(args.out_dir / f'fold{fold}.pt',
                                     map_location=DEVICE))
    model.eval()
    scores = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.use_amp):
        for batch in tqdm(test_dl, desc=f'Test fold {fold}'):
            x = batch['x'].to(DEVICE)
            views = tta_views(x) if args.use_tta else [x]
            probs = torch.zeros(x.size(0), device=DEVICE)
            for v in views:
                probs += torch.sigmoid(model(v).squeeze(1).float())
            probs /= len(views)
            scores.extend(probs.cpu().tolist())
    test_scores += np.array(scores) / args.n_folds
    del model
    torch.cuda.empty_cache()

sub['Diagnosis'] = test_scores
sub_path = args.out_dir / 'submission.csv'
sub.to_csv(sub_path, index=False)
print(f'Wrote {sub_path}')
print(f'Score distribution: min={test_scores.min():.4f} max={test_scores.max():.4f} '
      f'mean={test_scores.mean():.4f} frac>0.5={(test_scores > 0.5).mean():.4f}')
sub.head()

# Final submission sanity check
sub_check = pd.read_csv('/kaggle/working/submission.csv')
print('Shape:', sub_check.shape)
print('Columns:', list(sub_check.columns))
print('Null values:', sub_check.isna().sum().to_dict())
assert sub_check.isna().sum().sum() == 0, 'NaNs in submission!'
assert sub_check['Diagnosis'].between(0, 1).all(), 'Scores outside [0,1]!'
assert len(sub_check) == len(pd.read_csv(args.data_root / 'sampleSubmission.csv')), \
    'Row count does not match sampleSubmission!'
print('\nScore stats:')
print(sub_check['Diagnosis'].describe())
print('\nFirst 5 rows:'); print(sub_check.head())
print('\nAll checks passed - safe to submit.')