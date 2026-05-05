import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.ml_model_utils import ID_TO_LABEL, LABEL_TO_ID, Standardizer, extract_unit_features, make_model

DISPLAY = {"lr": "LR", "svm": "SVM", "lda": "LDA", "knn": "KNN", "dt": "DT", "rf": "RF", "gb": "GB", "mlp": "MLP"}


def load_participant_windows(signal_root: Path, labels_dir: Path, subset: str):
    rows = []
    windows = []
    participants = []
    window = 3750
    for label_file in sorted(labels_dir.glob(f"*_{subset}_labels.csv")):
        pid = label_file.name.split("_")[0]
        labels = pd.read_csv(label_file)
        signal = np.load(signal_root / pid / f"{subset}_signal.npy")
        signal = signal[:, :7].astype(np.float32)
        n_units = len(labels)
        need = n_units * window
        if signal.shape[0] < need:
            raise ValueError(f"{pid} {subset} signal too short: need {need}, got {signal.shape[0]}")
        for idx in range(n_units):
            seg = signal[idx * window : (idx + 1) * window]
            windows.append(seg)
            participants.append(pid)
            rows.append(
                {
                    "unit_id": f"{subset}_{pid}_w{idx:05d}",
                    "participant_id": pid,
                    "unit_index": idx,
                    "cognitive_fatigue_label": labels.iloc[idx]["cognitive_fatigue_label"],
                    "physical_fatigue_label": labels.iloc[idx]["physical_fatigue_label"],
                }
            )
    meta = pd.DataFrame(rows)
    x = np.stack(windows).astype(np.float32)
    return meta, x, np.array(participants)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-root", type=Path, required=True)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--subset", choices=["OFL", "OFS"], required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["lr", "svm", "lda", "knn", "dt", "rf", "gb", "mlp"])
    args = ap.parse_args()

    meta, x_win, participants = load_participant_windows(args.signal_root, args.labels_dir, args.subset)
    x = extract_unit_features(x_win)
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    preds = []
    unique_participants = sorted(pd.unique(participants))
    for target in ["cognitive_fatigue_label", "physical_fatigue_label"]:
        y = meta[target].map(LABEL_TO_ID).to_numpy(int)
        tname = target.replace("_fatigue_label", "")
        for model_name in args.models:
            for test_pid in unique_participants:
                test_mask = participants == test_pid
                train_mask = ~test_mask
                xtr = x[train_mask]
                ytr = y[train_mask]
                xte = x[test_mask]
                yte = y[test_mask]
                sc = Standardizer().fit(xtr)
                xtr = sc.transform(xtr)
                xte = sc.transform(xte)
                clf = make_model(model_name)
                clf.fit(xtr, ytr)
                yp = clf.predict(xte)
                rep = classification_report(
                    yte, yp, labels=[0, 1, 2], target_names=["low", "medium", "high"], output_dict=True, zero_division=0
                )
                rows.append(
                    {
                        "subset": args.subset,
                        "target": tname,
                        "model": DISPLAY[model_name],
                        "fold": f"test_{test_pid}",
                        "macro_f1": f1_score(yte, yp, average="macro"),
                        "accuracy": accuracy_score(yte, yp),
                        "balanced_accuracy": balanced_accuracy_score(yte, yp),
                        "low_f1": rep["low"]["f1-score"],
                        "medium_f1": rep["medium"]["f1-score"],
                        "high_f1": rep["high"]["f1-score"],
                        "n_test": int(test_mask.sum()),
                    }
                )
                pm = meta.loc[test_mask, ["unit_id", "participant_id", "unit_index"]].copy()
                pm["subset"] = args.subset
                pm["target"] = tname
                pm["model"] = DISPLAY[model_name]
                pm["fold"] = f"test_{test_pid}"
                pm["y_true_id"] = yte
                pm["y_pred_id"] = yp
                pm["y_true"] = [ID_TO_LABEL[int(v)] for v in yte]
                pm["y_pred"] = [ID_TO_LABEL[int(v)] for v in yp]
                pm["correct"] = (yp == yte).astype(int)
                preds.append(pm)
                print(f"{args.subset} {tname} {DISPLAY[model_name]} test_{test_pid}: {rows[-1]['macro_f1']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    pd.concat(preds, ignore_index=True).to_csv(args.outdir / "predictions_by_unit.csv", index=False)
    out.to_csv(args.outdir / "metrics_by_fold.csv", index=False)
    summary = (
        out.groupby(["subset", "target", "model"])
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            low_f1=("low_f1", "mean"),
            medium_f1=("medium_f1", "mean"),
            high_f1=("high_f1", "mean"),
            n_folds=("fold", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(args.outdir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
