"""Run official TS2Vec representation + linear probe baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV


LABEL_TO_ID = {"low": 0, "medium": 1, "high": 2}


def macro_f1(y_true, y_pred, n_classes=3):
    vals = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        vals.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-key", type=str, default=None)
    parser.add_argument("--targets", nargs="+", default=["cognitive_fatigue_label", "physical_fatigue_label"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repr-dims", type=int, default=128)
    parser.add_argument("--hidden-dims", type=int, default=64)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--max-train-length", type=int, default=300)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    from ts2vec import TS2Vec

    units = pd.read_csv(args.units).set_index("unit_id")
    split_blob = json.loads(args.splits.read_text(encoding="utf-8"))
    split_key = args.split_key or next(iter(split_blob.keys()))
    splits = split_blob[split_key]
    seq = np.load(args.sequences, allow_pickle=True)
    unit_ids = seq["unit_id"].astype(str)
    x = np.concatenate([seq["ppg"], seq["imu"]], axis=-1).astype(np.float32)
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
            model = TS2Vec(
                input_dims=x.shape[-1],
                output_dims=args.repr_dims,
                hidden_dims=args.hidden_dims,
                depth=args.depth,
                device="cuda",
                batch_size=args.batch_size,
                lr=1e-3,
                max_train_length=args.max_train_length,
            )
            model.fit(x[train_idx], n_epochs=args.epochs, verbose=False)
            train_repr = model.encode(x[train_idx], encoding_window="full_series")
            test_repr = model.encode(x[test_idx], encoding_window="full_series")
            if train_repr.ndim == 3 and train_repr.shape[1] == 1:
                train_repr = train_repr.squeeze(1)
            if test_repr.ndim == 3 and test_repr.shape[1] == 1:
                test_repr = test_repr.squeeze(1)
            train_repr = train_repr.reshape(train_repr.shape[0], -1)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
            clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            clf.fit(train_repr, y[train_idx])
            pred = clf.predict(test_repr)
            result = {
                "subset": split_key,
                "target": target.replace("_fatigue_label", ""),
                "model": "TS2Vec",
                "fold": fold_name,
                "macro_f1": macro_f1(y[test_idx], pred),
                "accuracy": float(np.mean(y[test_idx] == pred)),
                "n_test": int(len(test_idx)),
            }
            rows.append(result)
            scores.append(result["macro_f1"])
            print(f"{target} {fold_name}: {result['macro_f1']:.3f}")
        print(f"SUMMARY {target}: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    summary = out.groupby(["subset", "target", "model"]).agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        accuracy=("accuracy", "mean"),
    ).reset_index()
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False))
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
