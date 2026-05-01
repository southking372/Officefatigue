# OfficeFatigue

**OfficeFatigue** is a wrist-worn multimodal dataset and benchmark for cognitive and physical fatigue sensing in office settings.

## Overview

OfficeFatigue provides synchronized PPG (photoplethysmography) and IMU (inertial measurement unit) signals from smartwatch-like wearable devices, along with validated fatigue labels covering both cognitive and physical dimensions. The dataset supports research in:

- Multimodal physiological sensing
- Time-series classification
- Fatigue detection and monitoring
- Office well-being applications

## Dataset Contents

| Component | Description |
|-----------|-------------|
| **PPG signals** | Preprocessed photoplethysmography at 1Hz |
| **IMU signals** | 6-axis accelerometer + gyroscope at 1Hz |
| **Cognitive fatigue labels** | Three-level: low / medium / high |
| **Physical fatigue labels** | Three-level: low / medium / high |
| **Self-report questionnaires** | MFI (4-20) and NASA-TLX (0-20) scores |
| **Behavior context** | Activity labels: static / micro_move / shift |
| **Benchmark splits** | Subject-independent LOPO evaluation |

## Benchmark Subsets

| Subset | Description | Participants | Samples | Duration |
|--------|-------------|:------------:|:-------:|:--------:|
| **OFL** (OfficeFatigue-Long) | 30-minute analysis units | 6 | 91 | ~45 hours |
| **OFS** (OfficeFatigue-Short) | 5-minute analysis units | - | - | Coming soon |

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Baseline Models

```bash
# Deep learning baselines (CNN, TCN, InceptionTime, TimesNet, etc.)
python baselines/run_deep_baselines.py \
    --data-dir data/OFL \
    --out results/deep_baselines.csv

# Feature-based baselines (Random Forest, LightGBM)
python baselines/run_feature_baselines.py \
    --data-dir data/OFL \
    --out results/feature_baselines.csv
```

## Benchmark Results

### OfficeFatigue-Long (OFL) - Macro-F1

| Model | Cognitive Fatigue | Physical Fatigue |
|-------|:-----------------:|:----------------:|
| Random Forest | 0.503 ± 0.141 | 0.585 ± 0.176 |
| LightGBM | 0.479 ± 0.179 | 0.506 ± 0.189 |
| TCN | 0.570 ± 0.118 | 0.711 ± 0.163 |
| TimesNet | 0.623 ± 0.121 | 0.614 ± 0.155 |
| InceptionTime | 0.538 ± 0.167 | 0.563 ± 0.103 |

*Evaluation: Leave-One-Participant-Out (LOPO) cross-validation*

## Data Format

### Sequence Data (`sequences_1hz.npz`)

```python
import numpy as np
data = np.load('data/OFL/sequences_1hz.npz', allow_pickle=True)

data['unit_id']  # Sample identifiers, shape: (N,)
data['ppg']      # PPG signals, shape: (N, 1800, 1)
data['imu']      # IMU signals, shape: (N, 1800, 6)
```

### Labels (`analysis_units.csv`)

| Column | Description |
|--------|-------------|
| `unit_id` | Unique sample identifier |
| `participant_id` | Participant identifier |
| `cognitive_fatigue_label` | low / medium / high |
| `physical_fatigue_label` | low / medium / high |
| `mfi_mental_raw` | MFI mental fatigue score (4-20) |
| `mfi_physical_raw` | MFI physical fatigue score (4-20) |
| `nasa_mental_demand` | NASA-TLX mental demand (0-20) |
| `nasa_physical_demand` | NASA-TLX physical demand (0-20) |
| `behavior_context` | static / micro_move / shift |

### Benchmark Splits (`lopo_splits.json`)

```json
{
  "OFL": {
    "test_P001": {
      "train": ["OFL_P002_0000", ...],
      "val": ["OFL_P006_0000", ...],
      "test": ["OFL_P001_0000", ...]
    },
    ...
  }
}
```

## Responsible Use

This dataset is intended for **non-commercial research** on wearable sensing and office fatigue modeling. 

**Prohibited uses:**
- Employee surveillance or monitoring
- Clinical diagnosis or medical decision-making
- Safety-critical applications
- Re-identification of participants

## Citation

```bibtex
@inproceedings{officefatigue2026,
  title={OfficeFatigue: A Multimodal Dataset for Cognitive and Physical Fatigue Sensing},
  author={Anonymous},
  booktitle={NeurIPS 2026 Datasets and Benchmarks Track},
  year={2026}
}
```

## License

This dataset is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Files

| File | Description | Status |
|------|-------------|--------|
| `data/OFL/` | OfficeFatigue-Long subset | Available |
| `data/OFS/` | OfficeFatigue-Short subset | Coming soon |
| `baselines/` | Baseline model implementations | Available |
| `DATASHEET.md` | Dataset documentation | Available |
| `croissant.json` | Croissant metadata | Available |
