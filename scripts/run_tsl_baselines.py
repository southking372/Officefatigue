"""Run official THUML Time-Series-Library Informer and TimesNet baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


LABEL_TO_ID = {"low": 0, "medium": 1, "high": 2}


def make_windows(ppg: np.ndarray, imu: np.ndarray, window: int, stride: int):
    x = np.concatenate([ppg, imu], axis=-1).astype(np.float32)
    xs, interval_idx = [], []
    for i in range(len(x)):
        for start in range(0, x.shape[1] - window + 1, stride):
            xs.append(x[i, start : start + window])
            interval_idx.append(i)
    return np.stack(xs).astype(np.float32), np.array(interval_idx, dtype=np.int64)


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, interval_idx: np.ndarray, y_interval: np.ndarray, selected: np.ndarray):
        mask = np.isin(interval_idx, selected)
        self.x = torch.from_numpy(x[mask]).float()
        self.interval_idx = interval_idx[mask]
        self.y = torch.from_numpy(y_interval[self.interval_idx]).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.interval_idx[idx], self.y[idx]


class TSLClassifier(nn.Module):
    def __init__(self, model_name: str, repo: Path, seq_len: int, enc_in: int):
        super().__init__()
        sys.path.insert(0, str(repo))
        if model_name == "TimesNet":
            from models.TimesNet import Model

            cfg = SimpleNamespace(
                task_name="classification",
                seq_len=seq_len,
                label_len=0,
                pred_len=0,
                enc_in=enc_in,
                d_model=64,
                d_ff=128,
                e_layers=2,
                dropout=0.1,
                embed="timeF",
                freq="s",
                top_k=3,
                num_kernels=3,
                num_class=3,
            )
        elif model_name == "Informer":
            from models.Informer import Model

            cfg = SimpleNamespace(
                task_name="classification",
                seq_len=seq_len,
                label_len=0,
                pred_len=0,
                enc_in=enc_in,
                dec_in=enc_in,
                c_out=enc_in,
                d_model=64,
                d_ff=128,
                e_layers=2,
                d_layers=1,
                n_heads=4,
                factor=3,
                dropout=0.1,
                embed="timeF",
                freq="s",
                activation="gelu",
                distil=False,
                num_class=3,
            )
        else:
            raise ValueError(model_name)
        self.model = Model(cfg)

    def forward(self, x):
        mask = torch.ones(x.shape[0], x.shape[1], device=x.device)
        return self.model(x, mask, None, None)


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


def interval_predict(model, loader, n_intervals, device):
    model.eval()
    probs = np.zeros((n_intervals, 3), dtype=np.float64)
    counts = np.zeros(n_intervals, dtype=np.float64)
    with torch.no_grad():
        for x, interval_idx, _ in loader:
            p = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
            for j, interval in enumerate(interval_idx.numpy()):
                probs[interval] += p[j]
                counts[interval] += 1
    counts[counts == 0] = 1
    return probs / counts[:, None]


def train_fold(model_name, repo, x_win, win_interval, y, train_idx, val_idx, test_idx, args, device):
    torch.manual_seed(args.seed)
    model = TSLClassifier(model_name, repo, seq_len=args.window, enc_in=x_win.shape[-1]).to(device)
    counts = np.bincount(y[train_idx], minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(WindowDataset(x_win, win_interval, y, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(x_win, win_interval, y, val_idx), batch_size=args.batch_size)
    test_loader = DataLoader(WindowDataset(x_win, win_interval, y, test_idx), batch_size=args.batch_size)
    best_state, best_val = None, -1.0
    patience = args.patience
    for _ in range(args.epochs):
        model.train()
        for xb, _, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
        val_probs = interval_predict(model, val_loader, len(y), device)
        val_pred = val_probs[val_idx].argmax(axis=1)
        val_f1 = macro_f1(y[val_idx], val_pred)
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
        if patience <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    probs = interval_predict(model, test_loader, len(y), device)
    pred = probs[test_idx].argmax(axis=1)
    return {
        "macro_f1": macro_f1(y[test_idx], pred),
        "accuracy": float(np.mean(y[test_idx] == pred)),
        "n_test": int(len(test_idx)),
        "best_val_macro_f1": float(best_val),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-key", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=["Informer", "TimesNet"])
    parser.add_argument("--targets", nargs="+", default=["cognitive_fatigue_label", "physical_fatigue_label"])
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    units = pd.read_csv(args.units).set_index("unit_id")
    split_blob = json.loads(args.splits.read_text(encoding="utf-8"))
    split_key = args.split_key or next(iter(split_blob.keys()))
    splits = split_blob[split_key]
    seq = np.load(args.sequences, allow_pickle=True)
    unit_ids = seq["unit_id"].astype(str)
    x_win, win_interval = make_windows(seq["ppg"].astype(np.float32), seq["imu"].astype(np.float32), args.window, args.stride)
    id_to_idx = {u: i for i, u in enumerate(unit_ids)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} windows={len(x_win)} channels={x_win.shape[-1]}")
    rows = []
    for target in args.targets:
        y = units.loc[unit_ids, target].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
        for model_name in args.models:
            scores = []
            for fold_name, split in splits.items():
                train_idx = np.array([id_to_idx[u] for u in split["train"] if u in id_to_idx], dtype=np.int64)
                val_idx = np.array([id_to_idx[u] for u in split.get("val", []) if u in id_to_idx], dtype=np.int64)
                test_idx = np.array([id_to_idx[u] for u in split["test"] if u in id_to_idx], dtype=np.int64)
                result = train_fold(model_name, args.repo, x_win, win_interval, y, train_idx, val_idx, test_idx, args, device)
                result.update({"subset": split_key, "target": target.replace("_fatigue_label", ""), "model": model_name, "fold": fold_name})
                rows.append(result)
                scores.append(result["macro_f1"])
                print(f"{target} {model_name} {fold_name}: {result['macro_f1']:.3f}")
            print(f"SUMMARY {target} {model_name}: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
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
