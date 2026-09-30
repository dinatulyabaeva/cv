from pathlib import Path
import math
import random

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(r"C:\Users\dinat\mrhisum_project")

FEATURES_PATH = PROJECT_DIR / "yt8m_features_1000.h5"
LABELS_PATH = PROJECT_DIR / "mr_hisum.h5"

MLP_PATH = PROJECT_DIR / "mlp_baseline.pt"
TRANSFORMER_PATH = PROJECT_DIR / "temporal_transformer.pt"

RESULTS_DIR = PROJECT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

SEED = 42

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
# SAME TEST SPLIT
# ============================================================

with h5py.File(FEATURES_PATH, "r") as f:
    video_ids = list(f.keys())

rng = np.random.default_rng(SEED)
rng.shuffle(video_ids)

n = len(video_ids)

n_train = int(n * 0.70)
n_val = int(n * 0.15)

test_ids = video_ids[n_train + n_val:]

print("Test videos:", len(test_ids))


# ============================================================
# MLP
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

            nn.Sigmoid()
        )

    def forward(self, x):

        return self.model(x).squeeze(-1)


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
# TRANSFORMER
# ============================================================

class TemporalTransformer(nn.Module):

    def __init__(self):

        super().__init__()

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
        padding_mask=None
    ):

        x = self.input_projection(x)

        x = self.positional_encoding(x)

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        return (
            self.output(x)
            .squeeze(-1)
        )


# ============================================================
# LOAD MODELS
# ============================================================

mlp = HighlightMLP().to(device)

mlp.load_state_dict(
    torch.load(
        MLP_PATH,
        map_location=device
    )
)

mlp.eval()


transformer = TemporalTransformer().to(device)

transformer.load_state_dict(
    torch.load(
        TRANSFORMER_PATH,
        map_location=device
    )
)

transformer.eval()

print("Models loaded.")


# ============================================================
# HELPERS
# ============================================================

def safe_spearman(a, b):

    rho = spearmanr(
        a,
        b
    ).statistic

    if np.isnan(rho):
        return 0.0

    return float(rho)


def mse(a, b):

    return float(
        np.mean(
            (a - b) ** 2
        )
    )


# ============================================================
# INFERENCE
# ============================================================

results = []

predictions = {}


with h5py.File(
    FEATURES_PATH,
    "r"
) as ff, h5py.File(
    LABELS_PATH,
    "r"
) as lf:

    with torch.no_grad():

        for idx, video_id in enumerate(
            test_ids,
            start=1
        ):

            rgb = (
                ff[video_id]["rgb"][:]
                .astype(np.float32)
            )

            target = (
                lf[video_id]["gtscore"][:]
                .astype(np.float32)
            )

            length = min(
                len(rgb),
                len(target)
            )

            rgb = rgb[:length]
            target = target[:length]

            # ----------------------------------------
            # MLP
            # ----------------------------------------

            X_mlp = torch.from_numpy(
                rgb
            ).to(device)

            mlp_pred = (
                mlp(X_mlp)
                .cpu()
                .numpy()
            )

            # ----------------------------------------
            # TRANSFORMER
            # shape:
            # [1, time, 1024]
            # ----------------------------------------

            X_transformer = (
                torch.from_numpy(rgb)
                .unsqueeze(0)
                .to(device)
            )

            transformer_pred = (
                transformer(
                    X_transformer
                )[0]
                .cpu()
                .numpy()
            )

            # ----------------------------------------
            # Metrics
            # ----------------------------------------

            mlp_mse = mse(
                mlp_pred,
                target
            )

            transformer_mse = mse(
                transformer_pred,
                target
            )

            mlp_rho = safe_spearman(
                mlp_pred,
                target
            )

            transformer_rho = safe_spearman(
                transformer_pred,
                target
            )

            results.append({

                "video_id": video_id,

                "length": length,

                "target_mean": float(
                    np.mean(target)
                ),

                "target_max": float(
                    np.max(target)
                ),

                "mlp_mse": mlp_mse,

                "transformer_mse":
                    transformer_mse,

                "mse_improvement":
                    mlp_mse
                    - transformer_mse,

                "mlp_spearman":
                    mlp_rho,

                "transformer_spearman":
                    transformer_rho,

                "spearman_improvement":
                    transformer_rho
                    - mlp_rho
            })

            predictions[video_id] = {

                "target": target,

                "mlp": mlp_pred,

                "transformer":
                    transformer_pred
            }

            if (
                idx % 25 == 0
                or idx == len(test_ids)
            ):

                print(
                    f"Processed "
                    f"{idx}/{len(test_ids)}"
                )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(results)

csv_path = (
    RESULTS_DIR
    / "per_video_metrics.csv"
)

df.to_csv(
    csv_path,
    index=False
)

print("\nSaved:")
print(csv_path)


# ============================================================
# OVERALL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PER-VIDEO SUMMARY")
print("=" * 70)

print(
    "MLP mean video Spearman:",
    round(
        df["mlp_spearman"].mean(),
        4
    )
)

print(
    "Transformer mean video Spearman:",
    round(
        df[
            "transformer_spearman"
        ].mean(),
        4
    )
)

print(
    "Mean Spearman improvement:",
    round(
        df[
            "spearman_improvement"
        ].mean(),
        4
    )
)

print(
    "Transformer better by Spearman:",
    int(
        (
            df[
                "spearman_improvement"
            ] > 0
        ).sum()
    ),
    "/",
    len(df)
)

print(
    "Transformer better by MSE:",
    int(
        (
            df[
                "mse_improvement"
            ] > 0
        ).sum()
    ),
    "/",
    len(df)
)


# ============================================================
# SELECT CASES
# ============================================================

best_transformer = (
    df.sort_values(
        "transformer_spearman",
        ascending=False
    )
    .iloc[0]
)

worst_transformer = (
    df.sort_values(
        "transformer_spearman",
        ascending=True
    )
    .iloc[0]
)

biggest_improvement = (
    df.sort_values(
        "spearman_improvement",
        ascending=False
    )
    .iloc[0]
)

biggest_degradation = (
    df.sort_values(
        "spearman_improvement",
        ascending=True
    )
    .iloc[0]
)


# Типичный пример:
# максимально близкий к median Transformer rho

median_rho = (
    df[
        "transformer_spearman"
    ].median()
)

median_index = (
    df[
        "transformer_spearman"
    ]
    .sub(median_rho)
    .abs()
    .idxmin()
)

median_case = df.loc[
    median_index
]


selected_cases = {

    "best_transformer":
        best_transformer["video_id"],

    "typical_case":
        median_case["video_id"],

    "worst_transformer":
        worst_transformer["video_id"],

    "biggest_improvement":
        biggest_improvement["video_id"],

    "biggest_degradation":
        biggest_degradation["video_id"]
}


print("\n" + "=" * 70)
print("SELECTED CASES")
print("=" * 70)

for case_name, video_id in (
    selected_cases.items()
):

    row = df[
        df["video_id"] == video_id
    ].iloc[0]

    print(
        f"\n{case_name}: "
        f"{video_id}"
    )

    print(
        f"  MLP rho: "
        f"{row['mlp_spearman']:.4f}"
    )

    print(
        f"  Transformer rho: "
        f"{row['transformer_spearman']:.4f}"
    )

    print(
        f"  improvement: "
        f"{row['spearman_improvement']:.4f}"
    )


# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_video(
    video_id,
    case_name
):

    data = predictions[
        video_id
    ]

    target = data["target"]
    mlp_pred = data["mlp"]
    transformer_pred = (
        data["transformer"]
    )

    row = df[
        df["video_id"]
        == video_id
    ].iloc[0]

    time = np.arange(
        len(target)
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        time,
        target,
        linewidth=2,
        label="Ground truth"
    )

    plt.plot(
        time,
        mlp_pred,
        linewidth=1.5,
        alpha=0.8,
        label=(
            "MLP "
            f"(rho={row['mlp_spearman']:.3f})"
        )
    )

    plt.plot(
        time,
        transformer_pred,
        linewidth=1.5,
        alpha=0.8,
        label=(
            "Temporal Transformer "
            f"(rho={row['transformer_spearman']:.3f})"
        )
    )

    plt.xlabel(
        "Temporal step"
    )

    plt.ylabel(
        "Highlight score"
    )

    plt.ylim(
        -0.05,
        1.05
    )

    plt.title(
        f"{case_name}: {video_id}"
    )

    plt.legend()

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / f"{case_name}_{video_id}.png"
    )

    plt.savefig(
        output_path,
        dpi=180
    )

    plt.close()

    print(
        "Plot saved:",
        output_path
    )


# ============================================================
# CREATE CASE PLOTS
# ============================================================

print("\nCreating plots...")

for case_name, video_id in (
    selected_cases.items()
):

    plot_video(
        video_id,
        case_name
    )


# ============================================================
# DISTRIBUTION OF SPEARMAN
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    df["mlp_spearman"],
    bins=20,
    alpha=0.6,
    label="MLP"
)

plt.hist(
    df["transformer_spearman"],
    bins=20,
    alpha=0.6,
    label="Temporal Transformer"
)

plt.xlabel(
    "Per-video Spearman correlation"
)

plt.ylabel(
    "Number of videos"
)

plt.title(
    "Distribution of per-video Spearman correlation"
)

plt.legend()

plt.tight_layout()

distribution_path = (
    PLOTS_DIR
    / "spearman_distribution.png"
)

plt.savefig(
    distribution_path,
    dpi=180
)

plt.close()


# ============================================================
# MLP vs TRANSFORMER SCATTER
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    df["mlp_spearman"],
    df["transformer_spearman"],
    alpha=0.65
)

minimum = min(
    df["mlp_spearman"].min(),
    df["transformer_spearman"].min()
)

maximum = max(
    df["mlp_spearman"].max(),
    df["transformer_spearman"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel(
    "MLP Spearman"
)

plt.ylabel(
    "Temporal Transformer Spearman"
)

plt.title(
    "Per-video model comparison"
)

plt.tight_layout()

scatter_path = (
    PLOTS_DIR
    / "mlp_vs_transformer_spearman.png"
)

plt.savefig(
    scatter_path,
    dpi=180
)

plt.close()


# ============================================================
# TOP / WORST TABLES
# ============================================================

columns = [

    "video_id",

    "mlp_mse",
    "transformer_mse",

    "mlp_spearman",
    "transformer_spearman",

    "spearman_improvement"
]


best_table = (
    df.sort_values(
        "transformer_spearman",
        ascending=False
    )
    [columns]
    .head(10)
)

worst_table = (
    df.sort_values(
        "transformer_spearman",
        ascending=True
    )
    [columns]
    .head(10)
)

improvement_table = (
    df.sort_values(
        "spearman_improvement",
        ascending=False
    )
    [columns]
    .head(10)
)


best_table.to_csv(
    RESULTS_DIR
    / "best_transformer_videos.csv",
    index=False
)

worst_table.to_csv(
    RESULTS_DIR
    / "worst_transformer_videos.csv",
    index=False
)

improvement_table.to_csv(
    RESULTS_DIR
    / "largest_improvements.csv",
    index=False
)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    "Results directory:",
    RESULTS_DIR
)

print(
    "Plots directory:",
    PLOTS_DIR
)