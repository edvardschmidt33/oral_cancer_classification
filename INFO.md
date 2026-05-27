# Oral Cancer Classification — Project Guide

### NOTE TO READER: Give this to your AI-Agent together with the codebase to give you a basic understanding/inuition to implement it yourself


Binary classification of oral cancer from paired bright-field (BF) and
fluorescence (FL) microscopy images of single cells. Predictions are aggregated
to a per-cell probability; patients are used only as grouping keys to prevent
patient-level leakage between train and validation.

The framework supports three fusion architectures and an optional SimCLR
self-supervised pretraining stage for the early-fusion model.

---

## 1. Repository layout

```
oral_cancer_classification/
├── configs/                      # YAML configs, one per model variant
│   ├── config.yaml               # gated dual-encoder ConvNeXtV2-Nano
│   ├── config_ca.yaml            # cross-attention fusion ConvNeXtV2-Tiny
│   └── config_ef.yaml            # early-fusion 6-channel ConvNeXtV2-Tiny (+ SimCLR)
├── dataset/                      # data root (not versioned)
│   ├── BF/{train,test}/          # bright-field JPGs, filenames like pat_<id>_<cell>.jpg
│   ├── FL/{train,test}/          # fluorescence JPGs (same filenames as BF)
│   ├── train.csv                 # columns: Name, Diagnosis (0/1)
│   └── sampleSubmission.csv      # columns: Name, Diagnosis  (defines test ordering)
├── outputs/
│   ├── checkpoints/              # fold{N}_best.pt, fold{N}_last.pt, pretrained_ef_backbone.pt
│   └── submissions/              # submission_best[_tta].csv, submission_last[_tta].csv
├── src/
│   ├── augmentations.py          # BF/FL color transforms, shared-geometry wrapper, SimCLR aug
│   ├── dataset.py                # OralCancerDataset (supervised), SimCLRDataset (two views)
│   ├── models.py                 # GatedFusionModel, CrossAttentionFusionModel,
│   │                             #   EarlyFusionConcatModel, SimCLRModel
│   ├── pretrain.py               # SimCLR pretraining entrypoint (early-fusion backbone)
│   ├── train.py                  # supervised training entrypoint
│   ├── inference.py              # test-set prediction + submission CSV writer
│   └── utils.py                  # patient-grouped StratifiedGroupKFold, set_seed
├── requirements.txt
├── run.ipynb                     # Colab/Kaggle-style notebook orchestrating the above
├── plan.md / early_fusion_ca_plan.md  # design notes (not required to run)
└── INFO.md                       # this file
```

All training/inference commands assume the project root as the working
directory and are run as Python modules (`python -m src.<entry>`), so imports
resolve correctly.

---

## 2. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requirements (`requirements.txt`):
torch>=2.0, torchvision>=0.15, timm>=0.9, pandas, numpy, scikit-learn, pyyaml,
pillow, wandb.

A CUDA-capable GPU is strongly recommended. AMP (mixed-precision) is enabled
automatically when CUDA is available.

### Data layout

The dataset directory expected by the configs is (paths in
[configs/config.yaml](configs/config.yaml)):

- `dataset/BF/train/*.jpg`, `dataset/FL/train/*.jpg` — paired, same filenames
- `dataset/BF/test/*.jpg`,  `dataset/FL/test/*.jpg`  — paired, same filenames
- `dataset/train.csv` — `Name, Diagnosis` (0/1) for every train image
- `dataset/sampleSubmission.csv` — defines the test ordering used by inference

Filenames must contain a `pat_<id>` substring (see
[src/utils.py:12](src/utils.py#L12)) — this is how cells get grouped by patient
for the fold split.

[configs/config_ef.yaml](configs/config_ef.yaml) is set to a Kaggle-Colab path
(`/root/.cache/kagglehub/...`). Edit the `data:` block to point at your local
dataset before running locally.

---

## 3. Model variants

Each variant is enabled by selecting a config; the model class is dispatched
from `model.type` in [src/train.py:33](src/train.py#L33).

| Config             | `model.type`           | Class                          | Backbone                | Notes |
|--------------------|------------------------|--------------------------------|-------------------------|-------|
| `config.yaml`      | `gated` (default)      | `GatedFusionModel`             | ConvNeXtV2-Nano         | Two encoders, GMU gate, concat head |
| `config_ca.yaml`   | `cross_attention`      | `CrossAttentionFusionModel`    | ConvNeXtV2-Tiny         | Per-pixel local cross-attention, single backbone |
| `config_ef.yaml`   | `early_fusion_concat`  | `EarlyFusionConcatModel`       | ConvNeXtV2-Tiny         | 6-channel stem (BF\|FL), optional SimCLR pretrain |

All three use a 2-stage freeze schedule: warmup (backbone frozen) for the first
`training.freeze_backbone_epochs` epochs, then partial unfreeze of stages 2-3
and norm layers. Optimizer is rebuilt on the transition so the new trainable
params get fresh state.

### Training mechanics shared across variants

- Stratified group K-fold by `pat_<id>` ([src/utils.py:19](src/utils.py#L19)) —
  no patient appears in both train and val of a fold.
- Class-weighted BCE-with-logits; weights set per-config (`class_weights`).
- MixUp on both modalities with the **same** lambda and permutation
  ([src/train.py:103](src/train.py#L103)) to preserve BF–FL pairing.
- Gradient accumulation (`training.accum_steps`) so effective batch
  = `batch_size * accum_steps`.
- AMP via `torch.amp.autocast` + `GradScaler` when CUDA is available.
- Per-fold checkpoints saved as `fold{N}_best.pt` (best val AUC) and
  `fold{N}_last.pt` (final epoch).
- Optional W&B logging via `--wandb`.

---

## 4. Running

All commands are run from the repo root.

### Smoke test (one batch, then exit)

```bash
python -m src.train --config configs/config.yaml --fold 0 --smoke
python -m src.pretrain --config configs/config_ef.yaml --smoke
```

Use this first to verify data paths, model construction, and shapes.

### A. Gated dual-encoder (default)

```bash
# Train each fold (n_folds=2 in config.yaml)
python -m src.train --config configs/config.yaml --fold 0
python -m src.train --config configs/config.yaml --fold 1

# Inference: averages sigmoid probs across the listed folds' best checkpoints
python -m src.inference --config configs/config.yaml --folds 0 1
```

### B. Cross-attention fusion

```bash
python -m src.train --config configs/config_ca.yaml --fold 0
python -m src.train --config configs/config_ca.yaml --fold 1
python -m src.train --config configs/config_ca.yaml --fold 2

python -m src.inference --config configs/config_ca.yaml --folds 0 1 2
```

TTA (8-view D4: rotations × h-flip) is `true` by default in this config.

### C. Early-fusion concat (with optional SimCLR pretrain)

1. (Optional but recommended) Pretrain the 6-channel backbone with SimCLR:

   ```bash
   python -m src.pretrain --config configs/config_ef.yaml
   ```

   Writes `outputs/checkpoints/pretrained_ef_backbone.pt`. The pool is BF+FL
   pairs from `train/` and `test/` combined (labels not needed). Two augmented
   views per cell drive an NT-Xent loss
   ([src/pretrain.py:29](src/pretrain.py#L29)). Toggle via
   `pretraining.enabled` in the config.

2. Supervised fine-tune. If `pretraining.enabled: true` and the checkpoint
   exists, `train.py` auto-loads it into `model.backbone` before training
   ([src/train.py:260](src/train.py#L260)); otherwise it falls back to the
   ImageNet-init backbone with a warning.

   ```bash
   python -m src.train --config configs/config_ef.yaml --fold 0
   python -m src.train --config configs/config_ef.yaml --fold 1
   python -m src.train --config configs/config_ef.yaml --fold 2
   ```

3. Inference:

   ```bash
   python -m src.inference --config configs/config_ef.yaml --folds 0 1 2
   ```

### Inference flags

```
--folds 0 1 2          # which fold checkpoints to ensemble (mean of sigmoid probs)
--ckpt best|last       # which checkpoint per fold to load (default: best)
--tta / --no-tta       # override config's inference.tta
--output PATH          # custom submission path
```

Submissions are written to `outputs/submissions/submission_<ckpt>[_tta].csv`
with columns matching `sampleSubmission.csv` (`Name`, `Diagnosis`).

### W&B logging

Add `--wandb` to `src.train`. Defaults: project `oral-cancer-classification`,
run name `fold{N}`. Override with `--wandb-project` / `--wandb-run-name`.

---

## 5. File-by-file reference

### `src/dataset.py`

- **`OralCancerDataset`** — supervised dataset. Loads paired BF/FL JPGs from
  two parallel directories using the same filename. `bf_transform` and
  `fl_transform` are PIL-level color augs run independently per modality;
  `geo_transform` is replayed with the same random seed on both modalities so
  flips/rotations stay aligned. Returns `(bf_tensor, fl_tensor, label,
  filename)`. `labels=None` is used at test time.
- **`SimCLRDataset`** — produces **two** independently augmented 6-channel
  views (BF concat FL along channel) of the same cell. Used only by
  `pretrain.py`.

### `src/augmentations.py`

- `build_bf_color_transform` — bright-field-specific colorish jitter
  (posterize / blur / solarize one-of, plus ColorJitter).
- `build_fl_color_transform` — fluorescence-specific: strong brightness +
  contrast jitter and Gaussian blur.
- `build_shared_geo_transform` — horizontal/vertical flip; applied with a
  shared seed in the dataset.
- `SimCLRAugmentation` — callable that returns one augmented `(bf, fl)` pair
  with shared geometry (h/v flip + rotation) and independent per-modality
  color. Called twice per cell in `SimCLRDataset` to form the contrastive
  pair.

### `src/models.py`

- **`GatedFusionModel`** — two ConvNeXtV2 encoders (BF + FL). Features
  projected through `tanh`, gated with a sigmoid over the concat
  ([src/models.py:30](src/models.py#L30)) (Arevalo et al. 2017 GMU), then the
  gated branches are concatenated for the head. `set_encoders_freeze_strategy`
  supports `'full' | 'partial' | 'none'`.
- **`ModalityProjection`** — small conv stack lifting a 3-channel modality to
  `proj_dim` features at full spatial resolution.
- **`LocalCrossAttention`** — symmetric per-pixel cross-attention between BF
  and FL within a local `window_size × window_size` neighborhood. Both
  directions computed; outputs are residual-added back into each modality.
  Includes a learned relative position bias per head.
- **`CrossAttentionFusionModel`** — pipeline: per-modality projection → local
  cross-attention → concat + 1×1 conv → kernel-4/stride-4 stem-bypass
  ([src/models.py:186](src/models.py#L186)) → ConvNeXtV2 backbone (stem
  replaced by `Identity`) → head. `set_freeze_strategy` controls the
  **backbone**; fusion + head are always trainable.
- **`EarlyFusionConcatModel`** — 6-channel input (BF concat FL on the channel
  dim). The backbone's 3-channel stem conv is replaced by a 6-channel
  equivalent, initialized by duplicating the pretrained weight along the new
  input channels and halving so activation magnitude is preserved
  ([src/models.py:267](src/models.py#L267)). `set_freeze_strategy` keeps the
  new stem trainable in `'warmup'` (since it's untrained), unlike the other
  models which freeze the whole backbone during warmup.
- **`SimCLRModel`** — wraps the same patched 6-channel backbone with an MLP
  projection head, returns L2-normalized projections. After pretraining,
  `model.backbone.state_dict()` is saved — its keys match
  `EarlyFusionConcatModel.backbone` exactly, so loading is a one-liner in
  `train.py`.

### `src/pretrain.py`

- Builds a SimCLR loader from BF+FL `train/` ∪ `test/` (no labels).
  `pretraining.subsample_frac < 1.0` deterministically subsamples (seeded by
  `split.seed`) for quick iterations.
- **`nt_xent_loss`** — symmetric InfoNCE. Sets the self-similarity diagonal to
  the dtype's minimum (not `-1e9`, which overflows under fp16 AMP).
- **`apply_freeze`** — `'partial'` keeps stem + stages 2-3 + late norms
  trainable; `'none'` unfreezes everything.
- Saves `model.backbone.state_dict()` to `pretraining.checkpoint_path` after
  every epoch.

### `src/train.py`

- `build_model` dispatches on `cfg['model']['type']` and returns
  `(model, set_freeze_fn, (warmup_key, partial_key))`. The keys differ between
  the gated model (`'full'`/`'partial'`) and the others (`'warmup'`/
  `'partial'`).
- If `cfg['pretraining'].enabled` and `model.type == 'early_fusion_concat'`,
  loads the SimCLR-pretrained backbone before training; warns if the
  checkpoint is missing and continues with ImageNet init.
- Training loop: warmup with backbone frozen → switch to partial unfreeze at
  epoch `freeze_backbone_epochs` (optimizer + scheduler rebuilt) → loops with
  MixUp, AMP, gradient accumulation.
- `test` returns cell-level AUC plus a per-patient summary (mean predicted
  probability and cell count) — labels are constant within a patient, so a
  within-patient AUC is undefined.
- Saves `fold{N}_best.pt` (on val-AUC improvement) and `fold{N}_last.pt`
  (every epoch).

### `src/inference.py`

- Loads each requested fold's checkpoint, builds the matching model from the
  config, runs the test loader, and averages sigmoid probabilities across
  folds.
- Test-set filenames come from `sampleSubmission.csv` so the output ordering
  always matches the expected submission ordering.
- `--tta` toggles 8-view D4 averaging (4 rotations × {identity, h-flip}); BF
  and FL are transformed in lockstep to preserve their pairing
  ([src/inference.py:77](src/inference.py#L77)).

### `src/utils.py`

- `extract_patient_id` — pulls the `pat_<id>` substring out of a filename.
- `get_patient_splits` — `StratifiedGroupKFold` over cells, grouped by
  patient. Asserts no patient leaks between train/val of any fold; prints
  per-fold patient counts, cell counts, and cancer rates; warns if a fold has
  only one class at the patient level (its AUC would be meaningless).
- `set_seed` — seeds Python `random`, NumPy, and torch (CPU + CUDA).

---

## 6. Typical end-to-end workflow

```bash
# 0. Verify data + model build
python -m src.train --config configs/config_ef.yaml --fold 0 --smoke

# 1. (Early-fusion only) Self-supervised pretraining
python -m src.pretrain --config configs/config_ef.yaml

# 2. Supervised K-fold training
for f in 0 1 2; do
  python -m src.train --config configs/config_ef.yaml --fold $f --wandb
done

# 3. Ensemble inference + submission
python -m src.inference --config configs/config_ef.yaml --folds 0 1 2
# -> outputs/submissions/submission_best_tta.csv
```

---

## 7. Notes / gotchas

- The early-fusion config currently points to a Kaggle/Colab dataset path —
  edit `data:` in [configs/config_ef.yaml](configs/config_ef.yaml) for local
  runs.
- The gated config defaults to `n_folds: 2`; the other two use 3. Use
  `--folds` in inference to match what you actually trained.
- W&B and TTA are opt-in but cheap. TTA roughly 8x's inference cost.
- Patient-grouped folding is required — without it, val AUC will be wildly
  optimistic because of patient-level signal leaking through cell-level
  duplicates.
