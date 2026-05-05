"""Run MiniRocket baseline with sktime.

This is the heavier paper-style baseline entry used for participant-level
LOPO evaluation on prepared OfficeFatigue sequence files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sktime.transformations.panel.rocket import MiniRocketMultivariate


LABEL_TO_ID = {"low": 0, "medium": 1, "high": 2}


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    vals = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        vals.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(vals))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-key", type=str, default=None)
    parser.add_argument("--targets", nargs="+", default=["cognitive_fatigue_label", "physical_fatigue_label"])
    parser.add_argument("--num-kernels", type=int, default=10000)
    args = parser.parse_args()

    units = pd.read_csv(args.units).set_index("unit_id")
    split_blob = json.loads(args.splits.read_text(encoding="utf-8"))
    split_key = args.split_key or next(iter(split_blob.keys()))
    splits = split_blob[split_key]
    seq = np.load(args.sequences, allow_pickle=True)
    unit_ids = seq["unit_id"].astype(str)
    x = np.concatenate([seq["ppg"], seq["imu"]], axis=-1).astype(np.float32)
    x = np.transpose(x, (0, 2, 1))
    id_to_idx = {u: i for i, u in enumerate(unit_ids)}
    rows = []

    for target in args.targets:
        y = units.loc[unit_ids, target].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
        scores = []
        for fold_name, split in splits.items():
            train_ids = split["train"] + split.get("val", [])
            test_ids = split["test"]
            train_idx = np.array([id_to_idx[u] for u in train_ids if u in id_to_idx], dtype=np.int64)
            test_idx = np.array([id_to_idx[u] for u in test_ids if u in id_to_idx], dtype=np.int64)
            transformer = MiniRocketMultivariate(num_kernels=args.num_kernels, random_state=7)
            x_train = transformer.fit_transform(x[train_idx])
            x_test = transformer.transform(x[test_idx])
            clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            clf.fit(x_train, y[train_idx])
            pred = clf.predict(x_test)
            result = {
                "subset": split_key,
                "target": target.replace("_fatigue_label", ""),
                "model": "MiniRocket",
                "fold": fold_name,
                "macro_f1": macro_f1(y[test_idx], pred),
                "accuracy": accuracy(y[test_idx], pred),
                "n_test": int(len(test_idx)),
            }
            rows.append(result)
            scores.append(result["macro_f1"])
            print(f"{target} {fold_name}: {result['macro_f1']:.3f}")
        print(f"SUMMARY {target}: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    summary = (
        out.groupby(["subset", "target", "model"])
        .agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"), accuracy=("accuracy", "mean"))
        .reset_index()
    )
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False))
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
