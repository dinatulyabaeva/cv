from pathlib import Path
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(r"C:\Users\dinat\mrhisum_project")

FEATURES_PATH = PROJECT_DIR / "yt8m_features_1000.h5"
LABELS_PATH = PROJECT_DIR / "mr_hisum.h5"

MODEL_PATH = PROJECT_DIR / "mlp_baseline.pt"

SEED = 42
BATCH_SIZE = 512
EPOCHS = 20
LR = 1e-3


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# LOAD VIDEO IDS
# ============================================================

with h5py.File(FEATURES_PATH, "r") as f:
    video_ids = list(f.keys())

print("Всего видео:", len(video_ids))


# ------------------------------------------------------------
# Очень важно:
# split делаем ПО ВИДЕО, а не по отдельным кадрам.
# Иначе кадры одного видео попадут и в train, и в test.
# ------------------------------------------------------------

rng = np.random.default_rng(SEED)
rng.shuffle(video_ids)

n = len(video_ids)

n_train = int(n * 0.70)
n_val = int(n * 0.15)

train_ids = video_ids[:n_train]

val_ids = video_ids[
    n_train:n_train + n_val
]

test_ids = video_ids[
    n_train + n_val:
]

print("\nSplit:")
print("Train:", len(train_ids))
print("Val:  ", len(val_ids))
print("Test: ", len(test_ids))


# ============================================================
# DATASET
# ============================================================

class FrameDataset(Dataset):

    def __init__(
        self,
        video_ids,
        features_path,
        labels_path
    ):

        self.X = []
        self.y = []

        mismatches = 0

        with h5py.File(
            features_path,
            "r"
        ) as features_file, h5py.File(
            labels_path,
            "r"
        ) as labels_file:

            for video_id in video_ids:

                if video_id not in labels_file:
                    continue

                rgb = features_file[
                    video_id
                ]["rgb"][:].astype(
                    np.float32
                )

                target = labels_file[
                    video_id
                ]["gtscore"][:].astype(
                    np.float32
                )

                # Проверяем соответствие времени
                if len(rgb) != len(target):

                    print(
                        f"Length mismatch: "
                        f"{video_id}: "
                        f"rgb={len(rgb)}, "
                        f"target={len(target)}"
                    )

                    mismatches += 1

                    # На всякий случай выравниваем
                    length = min(
                        len(rgb),
                        len(target)
                    )

                    rgb = rgb[:length]
                    target = target[:length]

                self.X.append(rgb)
                self.y.append(target)

        self.X = np.concatenate(
            self.X,
            axis=0
        )

        self.y = np.concatenate(
            self.y,
            axis=0
        )

        print(
            f"Frames: {len(self.y)}, "
            f"mismatches: {mismatches}"
        )


    def __len__(self):
        return len(self.y)


    def __getitem__(self, idx):

        x = torch.from_numpy(
            self.X[idx]
        )

        y = torch.tensor(
            self.y[idx],
            dtype=torch.float32
        )

        return x, y


# ============================================================
# CREATE DATASETS
# ============================================================

print("\nСоздаём train dataset...")

train_dataset = FrameDataset(
    train_ids,
    FEATURES_PATH,
    LABELS_PATH
)

print("\nСоздаём validation dataset...")

val_dataset = FrameDataset(
    val_ids,
    FEATURES_PATH,
    LABELS_PATH
)

print("\nСоздаём test dataset...")

test_dataset = FrameDataset(
    test_ids,
    FEATURES_PATH,
    LABELS_PATH
)


# ============================================================
# LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

class HighlightMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(128, 1),

            # gtscore находится в [0, 1]
            nn.Sigmoid()
        )


    def forward(self, x):

        return self.model(x).squeeze(-1)


model = HighlightMLP().to(device)

print("\nModel:")
print(model)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader):

    model.eval()

    losses = []

    predictions = []
    targets = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)
            y = y.to(device)

            pred = model(X)

            loss = criterion(
                pred,
                y
            )

            losses.append(
                loss.item()
            )

            predictions.extend(
                pred.cpu().numpy()
            )

            targets.extend(
                y.cpu().numpy()
            )

    predictions = np.array(predictions)
    targets = np.array(targets)

    mse = np.mean(
        (predictions - targets) ** 2
    )

    rho = spearmanr(
        predictions,
        targets
    ).statistic

    return (
        np.mean(losses),
        mse,
        rho
    )


# ============================================================
# TRAIN
# ============================================================

best_val_loss = float("inf")


for epoch in range(1, EPOCHS + 1):

    model.train()

    train_losses = []

    for X, y in train_loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(X)

        loss = criterion(
            pred,
            y
        )

        loss.backward()

        optimizer.step()

        train_losses.append(
            loss.item()
        )

    train_loss = np.mean(
        train_losses
    )

    val_loss, val_mse, val_rho = evaluate(
        model,
        val_loader
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train={train_loss:.5f} | "
        f"val={val_loss:.5f} | "
        f"MSE={val_mse:.5f} | "
        f"Spearman={val_rho:.4f}"
    )

    # ----------------------------------------
    # сохраняем лучшую модель
    # ----------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("  -> best model saved")


# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 60)
print("TEST")
print("=" * 60)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

test_loss, test_mse, test_rho = evaluate(
    model,
    test_loader
)

print(f"Test loss:     {test_loss:.6f}")
print(f"Test MSE:      {test_mse:.6f}")
print(f"Test Spearman: {test_rho:.4f}")

print("\nModel saved:")
print(MODEL_PATH)