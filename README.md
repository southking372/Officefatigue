# OfficeFatigue (NIPS2026DATASET Release)

This repository contains the released label files and benchmark evaluation utilities for the 25-participant OfficeFatigue benchmark.

The participant-level processed signal dataset is hosted on Hugging Face:

- [lemonademelon/OfficeFatigue](https://huggingface.co/datasets/lemonademelon/OfficeFatigue)

This release does not rebuild labels. It assumes you download the participant-level signal files from Hugging Face and evaluate them directly with the label CSV files stored here.

## Included files

- `labels/OFL/p01_OFL_labels.csv` ... `p25_OFL_labels.csv`
- `labels/OFS/p01_OFS_labels.csv` ... `p25_OFS_labels.csv`
- `components/ml_model_utils.py`
- `scripts/run_ml_benchmarks.py`
- `scripts/run_minirocket.py`
- `scripts/run_window_tsai_baselines.py`
- `scripts/run_tsl_baselines.py`
- `scripts/run_ts2vec_official.py`

## Label file format

Each per-participant label CSV contains exactly two columns:

- `cognitive_fatigue_label`
- `physical_fatigue_label`

There are no extra metadata columns inside the per-participant label files.

## How the labels correspond to the 25 participants

For participant `pXX`:

- `labels/OFL/pXX_OFL_labels.csv` corresponds to `pXX/OFL_signal.npy`
- `labels/OFS/pXX_OFS_labels.csv` corresponds to `pXX/OFS_signal.npy`

The correspondence rule is by row order:

- row `0` in a participant label CSV corresponds to labeled unit `0`
- row `1` corresponds to labeled unit `1`
- and so on

So if you segment the participant signal in benchmark order, labels are aligned row-by-row.

## Expected local signal layout

After downloading the Hugging Face dataset, your local signal directory should look like:

```text
OfficeFatigue/
  p01/
    OFL_signal.npy
    OFS_signal.npy
  p02/
    OFL_signal.npy
    OFS_signal.npy
  ...
  p25/
    OFL_signal.npy
    OFS_signal.npy
```

## Unit sizes

- OFS: 84 label rows per participant
- OFL: typically 678 to 858 label rows per participant

The benchmark scripts assume 30-second windows at 125 Hz:

- `3750` samples per labeled unit

## Installation

Run all commands below from the repository root. Create an environment and install the required packages:

```bash
pip install -r requirements.txt
```

## Machine-learning evaluation

Example: run OFL machine-learning baselines

```bash
python scripts/run_ml_benchmarks.py \
  --signal-root /path/to/OfficeFatigue \
  --labels-dir labels/OFL \
  --subset OFL \
  --outdir results/ml_ofl
```

Example: run OFS machine-learning baselines

```bash
python scripts/run_ml_benchmarks.py \
  --signal-root /path/to/OfficeFatigue \
  --labels-dir labels/OFS \
  --subset OFS \
  --outdir results/ml_ofs
```

Default ML models:

- LR
- SVM
- LDA
- KNN
- DT
- RF
- GB
- MLP

## Deep-learning evaluation

The repository now includes heavier benchmark wrappers that are closer to the paper-style model families than the earlier lightweight reference script.

Included deep scripts:

- `scripts/run_minirocket.py`
  - MiniRocket via `sktime`
- `scripts/run_window_tsai_baselines.py`
  - `1D CNN`
  - `TCN`
  - `InceptionTime`
  - `PatchTST`
  - `Multimodal Ref.` (Attention-Fusion-style two-branch model)
- `scripts/run_tsl_baselines.py`
  - `Informer`
  - `TimesNet`
  - wraps the official THUML Time-Series-Library codebase
- `scripts/run_ts2vec_official.py`
  - TS2Vec representation learning + linear probe
  - wraps an external TS2Vec repository

These deep scripts are intentionally heavier. Some of them require external upstream repositories and prebuilt benchmark sequence files. They are included so that the model families are represented more faithfully, even if they are not the minimal one-command path.

### Deep dependencies

Install the base Python dependencies first:

```bash
pip install -r requirements.txt
```

For the heavier scripts, you may also need:

```bash
pip install tsai sktime
```

For `Informer`, `TimesNet`, and `TS2Vec`, clone the corresponding upstream repositories locally and pass them through `--repo`.

### Minimal direct deep evaluation

If you want a minimal local deep evaluation path based only on this repository plus the Hugging Face participant signals, use the machine-learning script above first. The heavier deep wrappers below are closer to the benchmark model families, but they expect prepared benchmark inputs such as:

- `analysis_units.csv`
- `lopo_splits.json`
- `windows_125hz.npz` or equivalent prepared sequence files

### Example: MiniRocket

```bash
python scripts/run_minirocket.py \
  --units /path/to/analysis_units.csv \
  --splits /path/to/lopo_splits.json \
  --sequences /path/to/windows_125hz.npz \
  --split-key OFL \
  --out results/minirocket_ofl.csv
```

### Example: tsai baselines

```bash
python scripts/run_window_tsai_baselines.py \
  --units /path/to/analysis_units.csv \
  --splits /path/to/lopo_splits.json \
  --sequences /path/to/windows_125hz.npz \
  --split-key OFL \
  --out results/window_tsai_ofl.csv \
  --epochs 30
```

### Example: Informer / TimesNet

```bash
python scripts/run_tsl_baselines.py \
  --repo /path/to/Time-Series-Library \
  --units /path/to/analysis_units.csv \
  --splits /path/to/lopo_splits.json \
  --sequences /path/to/windows_125hz.npz \
  --split-key OFL \
  --out results/tsl_ofl.csv
```

### Example: TS2Vec

```bash
python scripts/run_ts2vec_official.py \
  --repo /path/to/ts2vec \
  --units /path/to/analysis_units.csv \
  --splits /path/to/lopo_splits.json \
  --sequences /path/to/windows_125hz.npz \
  --split-key OFL \
  --out results/ts2vec_ofl.csv
```

## Outputs

The direct ML script writes:

- `metrics_by_fold.csv`
- `predictions_by_unit.csv`
- `summary.csv`

The heavier deep wrappers write:

- a per-fold metrics CSV
- a summary CSV beside the requested output file

## Notes

- Each label row is matched to one contiguous 30-second signal segment in benchmark order.
- `scripts/run_ml_benchmarks.py` performs participant-level leave-one-participant-out evaluation directly from the downloaded `.npy` files and the released label CSV files.
- The heavier deep wrappers are included because they better match the intended model families, but they assume prepared benchmark tensors and, for some methods, external upstream repositories.
