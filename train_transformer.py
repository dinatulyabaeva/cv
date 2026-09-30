from pathlib import Path
import random
import math

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

MODEL_PATH = PROJECT_DIR / "temporal_transformer.pt"

SEED = 42

BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4

INPUT_DIM = 1024
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.2


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# SAME VIDEO SPLIT AS MLP
# ============================================================

with h5py.File(FEATURES_PATH, "r") as f:
    video_ids = list(f.keys())

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

class VideoDataset(Dataset):

    def __init__(
        self,
        video_ids,
        features_path,
        labels_path
    ):

        self.data = []

        mismatches = 0

        with h5py.File(
            features_path,
            "r"
        ) as ff, h5py.File(
            labels_path,
            "r"
        ) as lf:

            for video_id in video_ids:

                if video_id not in lf:
                    continue

                rgb = ff[
                    video_id
                ]["rgb"][:].astype(
                    np.float32
                )

                target = lf[
                    video_id
                ]["gtscore"][:].astype(
                    np.float32
                )

                if len(rgb) != len(target):

                    mismatches += 1

                    length = min(
                        len(rgb),
                        len(target)
                    )

                    rgb = rgb[:length]
                    target = target[:length]

                self.data.append(
                    (
                        video_id,
                        rgb,
                        target
                    )
                )

        print(
            f"Videos: {len(self.data)}, "
            f"mismatches: {mismatches}"
        )


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        video_id, rgb, target = self.data[idx]

        return (
            video_id,
            torch.from_numpy(rgb),
            torch.from_numpy(target)
        )


# ============================================================
# COLLATE
#
# Видео имеют разную длину.
# Дополняем последовательности нулями.
# ============================================================

def collate_fn(batch):

    video_ids = [
        item[0]
        for item in batch
    ]

    features = [
        item[1]
        for item in batch
    ]

    targets = [
        item[2]
        for item in batch
    ]

    lengths = torch.tensor(
        [len(x) for x in features],
        dtype=torch.long
    )

    max_len = int(
        lengths.max()
    )

    batch_size = len(batch)

    X = torch.zeros(
        batch_size,
        max_len,
        INPUT_DIM,
        dtype=torch.float32
    )

    y = torch.zeros(
        batch_size,
        max_len,
        dtype=torch.float32
    )

    mask = torch.ones(
        batch_size,
        max_len,
        dtype=torch.bool
    )

    for i, (
        feature,
        target
    ) in enumerate(
        zip(features, targets)
    ):

        length = len(feature)

        X[i, :length] = feature
        y[i, :length] = target

        # False = настоящий временной шаг
        # True = padding
        mask[i, :length] = False

    return (
        video_ids,
        X,
        y,
        mask,
        lengths
    )


# ============================================================
# DATA
# ============================================================

print("\nTrain dataset:")
train_dataset = VideoDataset(
    train_ids,
    FEATURES_PATH,
    LABELS_PATH
)

print("\nValidation dataset:")
val_dataset = VideoDataset(
    val_ids,
    FEATURES_PATH,
    LABELS_PATH
)

print("\nTest dataset:")
test_dataset = VideoDataset(
    test_ids,
    FEATURES_PATH,
    LABELS_PATH
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=5000
    ):

        super().__init__()

        position = torch.arange(
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            )
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        pe = torch.zeros(
            max_len,
            d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )


    def forward(self, x):

        return (
            x
            + self.pe[:, :x.size(1)]
        )


# ============================================================
# TEMPORAL TRANSFORMER
# ============================================================

class TemporalTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        # 1024 -> 128
        self.input_projection = nn.Linear(
            INPUT_DIM,
            D_MODEL
        )

        self.positional_encoding = (
            PositionalEncoding(
                D_MODEL
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=NHEAD,
                dim_feedforward=DIM_FEEDFORWARD,
                dropout=DROPOUT,
                batch_first=True,
                activation="gelu"
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=NUM_LAYERS
            )
        )

        self.output = nn.Sequential(
            nn.Linear(
                D_MODEL,
                64
            ),
            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                64,
                1
            ),

            nn.Sigmoid()
        )


    def forward(
        self,
        x,
        padding_mask
    ):

        x = self.input_projection(x)

        x = self.positional_encoding(x)

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        x = self.output(x)

        return x.squeeze(-1)


model = TemporalTransformer().to(device)

print("\nModel:")
print(model)


# ============================================================
# MASKED MSE
# ============================================================

def masked_mse(
    prediction,
    target,
    padding_mask
):

    valid = ~padding_mask

    return torch.mean(
        (
            prediction[valid]
            - target[valid]
        ) ** 2
    )


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_predictions = []
    all_targets = []

    losses = []

    video_spearman = []

    with torch.no_grad():

        for (
            video_ids,
            X,
            y,
            mask,
            lengths
        ) in loader:

            X = X.to(device)
            y = y.to(device)
            mask = mask.to(device)

            prediction = model(
                X,
                mask
            )

            loss = masked_mse(
                prediction,
                y,
                mask
            )

            losses.append(
                loss.item()
            )

            for i, length in enumerate(lengths):

                length = int(length)

                pred_i = (
                    prediction[
                        i,
                        :length
                    ]
                    .cpu()
                    .numpy()
                )

                target_i = (
                    y[
                        i,
                        :length
                    ]
                    .cpu()
                    .numpy()
                )

                all_predictions.extend(
                    pred_i
                )

                all_targets.extend(
                    target_i
                )

                rho = spearmanr(
                    pred_i,
                    target_i
                ).statistic

                if not np.isnan(rho):
                    video_spearman.append(
                        rho
                    )

    all_predictions = np.asarray(
        all_predictions
    )

    all_targets = np.asarray(
        all_targets
    )

    mse = np.mean(
        (
            all_predictions
            - all_targets
        ) ** 2
    )

    global_rho = spearmanr(
        all_predictions,
        all_targets
    ).statistic

    mean_video_rho = np.mean(
        video_spearman
    )

    return (
        np.mean(losses),
        mse,
        global_rho,
        mean_video_rho
    )


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    losses = []

    for (
        video_ids,
        X,
        y,
        mask,
        lengths
    ) in train_loader:

        X = X.to(device)
        y = y.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        prediction = model(
            X,
            mask
        )

        loss = masked_mse(
            prediction,
            y,
            mask
        )

        loss.backward()

        # защита от больших градиентов
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        losses.append(
            loss.item()
        )

    train_loss = np.mean(
        losses
    )

    (
        val_loss,
        val_mse,
        val_global_rho,
        val_video_rho
    ) = evaluate(
        model,
        val_loader
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train={train_loss:.5f} | "
        f"val={val_loss:.5f} | "
        f"MSE={val_mse:.5f} | "
        f"global rho={val_global_rho:.4f} | "
        f"video rho={val_video_rho:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "  -> best model saved"
        )


# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 70)
print("TEST")
print("=" * 70)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

(
    test_loss,
    test_mse,
    test_global_rho,
    test_video_rho
) = evaluate(
    model,
    test_loader
)


print(
    f"Test loss:           "
    f"{test_loss:.6f}"
)

print(
    f"Test MSE:            "
    f"{test_mse:.6f}"
)

print(
    f"Test global Spearman:"
    f" {test_global_rho:.4f}"
)

print(
    f"Test video Spearman: "
    f"{test_video_rho:.4f}"
)

print("\nModel saved:")
print(MODEL_PATH)