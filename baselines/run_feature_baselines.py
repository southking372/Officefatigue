"""Feature-based baselines for OfficeFatigue benchmark.

Models:
- Random Forest
- LightGBM
- Logistic Regression (Softmax)
- Nearest Centroid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ppg_mean", "ppg_std", "ppg_zero_pct",
    "hr_mean", "hr_std", "hrv_mean", "hrv_std",
    "acc_mag_mean", "acc_mag_std", "gyro_mag_mean", "gyro_mag_std",
]

LABEL_TO_ID = {"low": 0, "medium": 1, "high": 2}


class Standardizer:
    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_[self.std_ < 1e-9] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.where(np.isnan(x), self.mean_, x)
        return (x - self.mean_) / self.std_


class NearestCentroid:
    """Simple nearest centroid classifier."""
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> "NearestCentroid":
        self.classes_ = np.array(sorted(np.unique(y)))
        self.centroids_ = np.vstack([x[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        d = ((x[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return self.classes_[np.argmin(d, axis=1)]


class SoftmaxRegression:
    """Multinomial logistic regression."""
    
    def __init__(self, lr: float = 0.05, epochs: int = 800, l2: float = 1e-3, seed: int = 7):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SoftmaxRegression":
        rng = np.random.default_rng(self.seed)
        n, d = x.shape
        classes = np.array(sorted(np.unique(y)))
        self.classes_ = classes
        k = len(classes)
        y_map = np.array([np.where(classes == yi)[0][0] for yi in y])
        xb = np.c_[np.ones(n), x]
        self.w_ = rng.normal(0, 0.01, size=(d + 1, k))
        target = np.eye(k)[y_map]
        
        for _ in range(self.epochs):
            logits = xb @ self.w_
            logits -= logits.max(axis=1, keepdims=True)
            prob = np.exp(logits)
            prob /= prob.sum(axis=1, keepdims=True)
            grad = xb.T @ (prob - target) / n
            grad[1:] += self.l2 * self.w_[1:]
            self.w_ -= self.lr * grad
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xb = np.c_[np.ones(len(x)), x]
        pred = np.argmax(xb @ self.w_, axis=1)
        return self.classes_[pred]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def make_model(name: str):
    if name == "nearest_centroid":
        return NearestCentroid()
    if name == "softmax":
        return SoftmaxRegression()
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                objective="multiclass", num_class=3,
                n_estimators=250, learning_rate=0.04, num_leaves=15,
                min_child_samples=5, random_state=7, verbose=-1,
            )
        except ImportError:
            print("[warn] LightGBM not available, using softmax instead")
            return SoftmaxRegression()
    if name == "random_forest":
        try:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=2,
                class_weight="balanced", random_state=7, n_jobs=-1,
            )
        except ImportError:
            print("[warn] scikit-learn not available, using nearest_centroid instead")
            return NearestCentroid()
    raise ValueError(f"Unknown model: {name}")


def prepare_xy(df: pd.DataFrame, ids: list[str], target: str):
    part = df.set_index("unit_id").loc[ids]
    x = part[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = part[target].map(LABEL_TO_ID).to_numpy(dtype=int)
    return x, y


def run_fold(df: pd.DataFrame, split: dict[str, list[str]], target: str, model_name: str):
    train_ids = split["train"] + split.get("val", [])
    test_ids = split["test"]
    x_train, y_train = prepare_xy(df, train_ids, target)
    x_test, y_test = prepare_xy(df, test_ids, target)
    
    scaler = Standardizer().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    model = make_model(model_name)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    
    return {
        "macro_f1": macro_f1(y_test, pred),
        "accuracy": accuracy(y_test, pred),
        "n_test": len(y_test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature baselines on OfficeFatigue")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to data directory")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--models", nargs="+", 
                        default=["nearest_centroid", "softmax", "random_forest", "lightgbm"],
                        help="Models to evaluate")
    args = parser.parse_args()

    df = pd.read_csv(args.data_dir / "analysis_units.csv")
    with (args.data_dir / "lopo_splits.json").open("r", encoding="utf-8") as f:
        splits = json.load(f)

    rows = []
    for subset, subset_splits in splits.items():
        for target in ["cognitive_fatigue_label", "physical_fatigue_label"]:
            for model_name in args.models:
                fold_scores = []
                
                for fold_name, split in subset_splits.items():
                    if not split["train"] or not split["test"]:
                        continue
                    
                    result = run_fold(df, split, target, model_name)
                    result.update({
                        "subset": subset,
                        "target": target.replace("_fatigue_label", ""),
                        "model": model_name,
                        "fold": fold_name,
                    })
                    rows.append(result)
                    fold_scores.append(result["macro_f1"])
                
                if fold_scores:
                    print(f"{subset} {target} {model_name}: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    summary = (
        out.groupby(["subset", "target", "model"])
        .agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"), accuracy=("accuracy", "mean"))
        .reset_index()
    )
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    
    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
