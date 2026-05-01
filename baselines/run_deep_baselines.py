"""Deep learning baselines for OfficeFatigue benchmark.

Models:
- CNN1D: Simple 1D convolutional network
- TCN: Temporal Convolutional Network with dilated convolutions
- InceptionTime: Multi-scale convolutional network
- TimesNet: Multi-scale temporal CNN
- PatchTST: Transformer-based patch model
- MultimodalCNN: Separate encoders for PPG and IMU
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


LABEL_TO_ID = {"low": 0, "medium": 1, "high": 2}


class SequenceDataset(Dataset):
    def __init__(self, ppg: np.ndarray, imu: np.ndarray, labels: np.ndarray):
        x = np.concatenate([ppg, imu], axis=-1)
        self.x = torch.from_numpy(x).permute(0, 2, 1).float()
        self.ppg = torch.from_numpy(ppg).permute(0, 2, 1).float()
        self.imu = torch.from_numpy(imu).permute(0, 2, 1).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.x[idx], self.ppg[idx], self.imu[idx], self.labels[idx]


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = ((kernel - 1) // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    """Simple 1D CNN baseline."""
    
    def __init__(self, in_channels: int = 7, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, n_classes))

    def forward(self, x, ppg=None, imu=None):
        return self.head(self.net(x))


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(channels, channels, kernel=5, dilation=dilation),
            nn.Dropout(0.15),
            ConvBlock(channels, channels, kernel=5, dilation=dilation),
        )

    def forward(self, x):
        return x + self.net(x)


class TCN(nn.Module):
    """Temporal Convolutional Network."""
    
    def __init__(self, in_channels: int = 7, n_classes: int = 3):
        super().__init__()
        self.stem = ConvBlock(in_channels, 64, kernel=7, stride=2)
        self.blocks = nn.Sequential(*[TCNBlock(64, d) for d in [1, 2, 4, 8, 16]])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 128), nn.ReLU(inplace=True), 
            nn.Dropout(0.2), nn.Linear(128, n_classes)
        )

    def forward(self, x, ppg=None, imu=None):
        return self.head(self.pool(self.blocks(self.stem(x))))


class InceptionBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        branch_ch = out_ch // 4
        self.b1 = ConvBlock(in_ch, branch_ch, kernel=9)
        self.b2 = ConvBlock(in_ch, branch_ch, kernel=19)
        self.b3 = ConvBlock(in_ch, branch_ch, kernel=39)
        self.b4 = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), ConvBlock(in_ch, branch_ch, kernel=1))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class InceptionTime(nn.Module):
    """InceptionTime-style multi-scale CNN."""
    
    def __init__(self, in_channels: int = 7, n_classes: int = 3):
        super().__init__()
        self.reduce = ConvBlock(in_channels, 64, kernel=1)
        self.net = nn.Sequential(
            InceptionBlock(64, 128),
            nn.MaxPool1d(2),
            InceptionBlock(128, 128),
            nn.MaxPool1d(2),
            InceptionBlock(128, 128),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, n_classes))

    def forward(self, x, ppg=None, imu=None):
        return self.head(self.net(self.reduce(x)))


class TimesBlockLite(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.branches = nn.ModuleList([
            ConvBlock(channels, channels, kernel=3, dilation=1),
            ConvBlock(channels, channels, kernel=5, dilation=2),
            ConvBlock(channels, channels, kernel=7, dilation=4),
        ])
        self.mix = ConvBlock(channels * 3, channels, kernel=1)

    def forward(self, x):
        return x + self.mix(torch.cat([b(x) for b in self.branches], dim=1))


class TimesNet(nn.Module):
    """TimesNet-style multi-scale temporal CNN."""
    
    def __init__(self, in_channels: int = 7, n_classes: int = 3):
        super().__init__()
        self.stem = ConvBlock(in_channels, 64, kernel=7, stride=2)
        self.blocks = nn.Sequential(
            TimesBlockLite(64), nn.MaxPool1d(2), 
            TimesBlockLite(64), TimesBlockLite(64)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), 
            nn.Linear(64, 128), nn.ReLU(inplace=True), 
            nn.Dropout(0.2), nn.Linear(128, n_classes)
        )

    def forward(self, x, ppg=None, imu=None):
        return self.head(self.blocks(self.stem(x)))


class PatchTST(nn.Module):
    """Transformer-based patch model."""
    
    def __init__(self, in_channels: int = 7, n_classes: int = 3, 
                 patch_len: int = 60, stride: int = 30, d_model: int = 96):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(in_channels * patch_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=192, 
            dropout=0.15, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x, ppg=None, imu=None):
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 2, 1, 3).flatten(2)
        z = self.encoder(self.proj(patches))
        return self.head(z.mean(dim=1))


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128, stride=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class MultimodalCNN(nn.Module):
    """Multimodal CNN with separate PPG and IMU encoders."""
    
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.ppg_encoder = ConvEncoder(1)
        self.imu_encoder = ConvEncoder(6)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True), 
            nn.Dropout(0.2), nn.Linear(128, n_classes)
        )

    def forward(self, x, ppg=None, imu=None):
        return self.head(torch.cat([self.ppg_encoder(ppg), self.imu_encoder(imu)], dim=1))


def make_model(name: str) -> nn.Module:
    models = {
        "cnn1d": CNN1D,
        "tcn": TCN,
        "inceptiontime": InceptionTime,
        "timesnet": TimesNet,
        "patchtst": PatchTST,
        "multimodal_cnn": MultimodalCNN,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    return models[name]()


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


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ys, ps = [], []
    for x, ppg, imu, y in loader:
        logits = model(x.to(device), ppg.to(device), imu.to(device))
        ys.append(y.numpy())
        ps.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_eval(model_name: str, ppg, imu, y, train_idx, val_idx, test_idx, args, device):
    torch.manual_seed(args.seed)
    model = make_model(model_name).to(device)
    
    counts = np.bincount(y[train_idx], minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        SequenceDataset(ppg[train_idx], imu[train_idx], y[train_idx]), 
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        SequenceDataset(ppg[val_idx], imu[val_idx], y[val_idx]), 
        batch_size=args.batch_size
    )
    test_loader = DataLoader(
        SequenceDataset(ppg[test_idx], imu[test_idx], y[test_idx]), 
        batch_size=args.batch_size
    )

    best_state = None
    best_val = -1.0
    patience_left = args.patience
    
    for epoch in range(args.epochs):
        model.train()
        for x, ppg_b, imu_b, y_b in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device), ppg_b.to(device), imu_b.to(device))
            loss = criterion(logits, y_b.to(device))
            loss.backward()
            optimizer.step()
        
        val_true, val_pred = predict(model, val_loader, device)
        val_f1 = macro_f1(val_true, val_pred)
        
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
        
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    
    test_true, test_pred = predict(model, test_loader, device)
    return {
        "macro_f1": macro_f1(test_true, test_pred),
        "accuracy": accuracy(test_true, test_pred),
        "n_test": int(len(test_true)),
        "best_val_macro_f1": float(best_val),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deep learning baselines on OfficeFatigue")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to data directory (e.g., data/OFL)")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--models", nargs="+", 
                        default=["cnn1d", "tcn", "inceptiontime", "timesnet", "patchtst", "multimodal_cnn"],
                        help="Models to evaluate")
    parser.add_argument("--targets", nargs="+", 
                        default=["cognitive_fatigue_label", "physical_fatigue_label"],
                        help="Target labels")
    parser.add_argument("--epochs", type=int, default=60, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    # Load data
    units = pd.read_csv(args.data_dir / "analysis_units.csv").set_index("unit_id")
    splits = json.loads((args.data_dir / "lopo_splits.json").read_text(encoding="utf-8"))["OFL"]
    seq = np.load(args.data_dir / "sequences_1hz.npz", allow_pickle=True)
    
    unit_ids = seq["unit_id"].astype(str)
    ppg = seq["ppg"].astype(np.float32)
    imu = seq["imu"].astype(np.float32)
    id_to_idx = {u: i for i, u in enumerate(unit_ids)}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data shape: PPG {ppg.shape}, IMU {imu.shape}")

    rows = []
    for target in args.targets:
        y = units.loc[unit_ids, target].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
        
        for model_name in args.models:
            fold_scores = []
            
            for fold_name, split in splits.items():
                train_idx = np.array([id_to_idx[u] for u in split["train"] if u in id_to_idx], dtype=np.int64)
                val_idx = np.array([id_to_idx[u] for u in split["val"] if u in id_to_idx], dtype=np.int64)
                test_idx = np.array([id_to_idx[u] for u in split["test"] if u in id_to_idx], dtype=np.int64)
                
                if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
                    continue
                
                result = train_eval(model_name, ppg, imu, y, train_idx, val_idx, test_idx, args, device)
                result.update({
                    "subset": "OFL", 
                    "target": target.replace("_fatigue_label", ""), 
                    "model": model_name, 
                    "fold": fold_name
                })
                rows.append(result)
                fold_scores.append(result["macro_f1"])
                print(f"{target} {model_name} {fold_name}: F1={result['macro_f1']:.3f}")
            
            if fold_scores:
                print(f"  => {model_name} mean: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")

    # Save results
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    
    summary = (
        out.groupby(["subset", "target", "model"])
        .agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"), accuracy=("accuracy", "mean"))
        .reset_index()
    )
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
