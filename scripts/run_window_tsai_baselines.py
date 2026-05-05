"""Window-level deep baselines with interval-level probability aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tsai.models.InceptionTime import InceptionTime
from tsai.models.PatchTST import PatchTST
from tsai.models.TCN import TCN


LABEL_TO_ID = {"low": 0, "medium": 1, "high": 2}


def make_windows(ppg: np.ndarray, imu: np.ndarray, window: int, stride: int):
    x = np.concatenate([ppg, imu], axis=-1).astype(np.float32)
    xs, interval_idx = [], []
    for i in range(len(x)):
        for start in range(0, x.shape[1] - window + 1, stride):
            xs.append(x[i, start : start + window])
            interval_idx.append(i)
    xs = np.stack(xs).astype(np.float32)
    interval_idx = np.array(interval_idx, dtype=np.int64)
    return xs, interval_idx


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, interval_idx: np.ndarray, y_interval: np.ndarray, selected_intervals: np.ndarray):
        mask = np.isin(interval_idx, selected_intervals)
        self.x = torch.from_numpy(x[mask]).permute(0, 2, 1).float()
        self.interval_idx = interval_idx[mask]
        self.y = torch.from_numpy(y_interval[self.interval_idx]).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.interval_idx[idx], self.y[idx]


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int = 7, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class CNN1DRef(nn.Module):
    def __init__(self, c_in=7, c_out=3):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(c_in, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(128, c_out),
        )

    def forward(self, x):
        return self.net(x)


class MultiCNNRef(nn.Module):
    def __init__(self, ppg_channels=1, imu_channels=6, c_out=3):
        super().__init__()
        self.ppg_channels = ppg_channels
        self.ppg = nn.Sequential(ConvBlock(ppg_channels, 32), ConvBlock(32, 64), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.imu = nn.Sequential(ConvBlock(imu_channels, 32), ConvBlock(32, 64), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(0.15), nn.Linear(128, c_out))

    def forward(self, x):
        return self.head(torch.cat([self.ppg(x[:, : self.ppg_channels]), self.imu(x[:, self.ppg_channels :])], dim=1))


class TsaiWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if out.ndim == 3:
            if out.shape[-1] == 3:
                out = out.mean(dim=1)
            elif out.shape[1] == 3:
                out = out.mean(dim=-1)
        return out


def make_model(name: str, seq_len: int, c_in: int, ppg_channels: int):
    if name == "1D CNN":
        return CNN1DRef(c_in=c_in)
    if name == "Multimodal Ref.":
        return MultiCNNRef(ppg_channels=ppg_channels, imu_channels=c_in - ppg_channels)
    if name == "InceptionTime":
        return TsaiWrapper(InceptionTime(c_in=c_in, c_out=3, seq_len=seq_len, nf=32))
    if name == "TCN":
        return TsaiWrapper(TCN(c_in=c_in, c_out=3, layers=[32, 32, 64], ks=5, conv_dropout=0.1, fc_dropout=0.2))
    if name == "PatchTST":
        return TsaiWrapper(
            PatchTST(
                c_in=c_in,
                c_out=3,
                seq_len=seq_len,
                pred_dim=3,
                n_layers=2,
                n_heads=4,
                d_model=64,
                d_ff=128,
                patch_len=10,
                stride=5,
                dropout=0.1,
                attn_dropout=0.05,
            )
        )
    raise ValueError(name)


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
            idx = interval_idx.numpy()
            for j, interval in enumerate(idx):
                probs[interval] += p[j]
                counts[interval] += 1
    counts[counts == 0] = 1
    probs /= counts[:, None]
    return probs


def train_fold(model_name, x_win, win_interval, y, train_idx, val_idx, test_idx, args, device, ppg_channels):
    torch.manual_seed(args.seed)
    model = make_model(model_name, args.window, x_win.shape[-1], ppg_channels).to(device)
    fit_idx = np.unique(np.concatenate([train_idx, val_idx])) if args.train_val else train_idx
    counts = np.bincount(y[fit_idx], minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(WindowDataset(x_win, win_interval, y, fit_idx), batch_size=args.batch_size, shuffle=True)
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
    test_probs = interval_predict(model, test_loader, len(y), device)
    pred = test_probs[test_idx].argmax(axis=1)
    return {
        "macro_f1": macro_f1(y[test_idx], pred),
        "accuracy": float(np.mean(y[test_idx] == pred)),
        "n_test": int(len(test_idx)),
        "best_val_macro_f1": float(best_val),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-key", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=["1D CNN", "TCN", "InceptionTime", "PatchTST", "Multimodal Ref."])
    parser.add_argument("--targets", nargs="+", default=["cognitive_fatigue_label", "physical_fatigue_label"])
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ppg-channels", type=int, default=1)
    parser.add_argument("--train-val", action="store_true")
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
    print(f"device={device} windows={len(x_win)}")
    rows = []

    for target in args.targets:
        y = units.loc[unit_ids, target].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
        for model_name in args.models:
            scores = []
            for fold_name, split in splits.items():
                train_idx = np.array([id_to_idx[u] for u in split["train"] if u in id_to_idx], dtype=np.int64)
                val_idx = np.array([id_to_idx[u] for u in split.get("val", []) if u in id_to_idx], dtype=np.int64)
                test_idx = np.array([id_to_idx[u] for u in split["test"] if u in id_to_idx], dtype=np.int64)
                result = train_fold(model_name, x_win, win_interval, y, train_idx, val_idx, test_idx, args, device, args.ppg_channels)
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
