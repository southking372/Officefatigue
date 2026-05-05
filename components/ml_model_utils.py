import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

LABEL_TO_ID = {"low": 0, "medium": 1, "high": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

FEATURE_COLUMNS = [
    "ppg_mean",
    "ppg_std",
    "ppg_zero_pct",
    "acc_mag_mean",
    "acc_mag_std",
    "gyro_mag_mean",
    "gyro_mag_std",
]


class Standardizer:
    def fit(self, x):
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_[self.std_ < 1e-9] = 1.0
        return self

    def transform(self, x):
        x = np.where(np.isnan(x), self.mean_, x)
        return (x - self.mean_) / self.std_


def extract_unit_features(x):
    ppg = x[:, :, 0]
    acc = x[:, :, 1:4]
    gyro = x[:, :, 4:7]
    acc_mag = np.linalg.norm(acc, axis=2)
    gyro_mag = np.linalg.norm(gyro, axis=2)
    feats = np.column_stack(
        [
            ppg.mean(axis=1),
            ppg.std(axis=1),
            (np.abs(ppg) < 1e-8).mean(axis=1),
            acc_mag.mean(axis=1),
            acc_mag.std(axis=1),
            gyro_mag.mean(axis=1),
            gyro_mag.std(axis=1),
        ]
    ).astype(np.float32)
    return feats


def make_model(name):
    if name == "lr":
        return LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="auto", random_state=7)
    if name == "svm":
        return SVC(C=2.0, kernel="rbf", gamma="scale", class_weight="balanced")
    if name == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=9, weights="distance")
    if name == "dt":
        return DecisionTreeClassifier(max_depth=8, min_samples_leaf=3, class_weight="balanced", random_state=7)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=7,
            n_jobs=-1,
        )
    if name == "gb":
        return GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=7)
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=7,
        )
    raise ValueError(name)
