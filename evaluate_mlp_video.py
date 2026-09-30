from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr


PROJECT_DIR = Path(r"C:\Users\dinat\mrhisum_project")

FEATURES_PATH = PROJECT_DIR / "yt8m_features_1000.h5"
LABELS_PATH = PROJECT_DIR / "mr_hisum.h5"
MODEL_PATH = PROJECT_DIR / "mlp_baseline.pt"

SEED = 42


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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

test_ids = video_ids[
    n_train + n_val:
]

print("Test videos:", len(test_ids))


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

            nn.Sigmoid()
        )


    def forward(self, x):

        return self.model(x).squeeze(-1)


model = HighlightMLP().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# EVALUATION
# ============================================================

video_rhos = []

all_predictions = []
all_targets = []

per_video_results = []


with h5py.File(
    FEATURES_PATH,
    "r"
) as ff, h5py.File(
    LABELS_PATH,
    "r"
) as lf:

    with torch.no_grad():

        for video_id in test_ids:

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

            X = torch.from_numpy(
                rgb
            ).to(device)

            prediction = (
                model(X)
                .cpu()
                .numpy()
            )

            rho = spearmanr(
                prediction,
                target
            ).statistic

            if not np.isnan(rho):

                video_rhos.append(rho)

                per_video_results.append(
                    (
                        video_id,
                        rho
                    )
                )

            all_predictions.extend(
                prediction
            )

            all_targets.extend(
                target
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
    video_rhos
)


print("\n" + "=" * 60)
print("MLP TEST RESULTS")
print("=" * 60)

print(f"MSE:                 {mse:.6f}")
print(f"Global Spearman:     {global_rho:.4f}")
print(f"Mean video Spearman: {mean_video_rho:.4f}")


# ============================================================
# BEST / WORST VIDEOS
# ============================================================

per_video_results.sort(
    key=lambda x: x[1]
)


print("\nWorst 5 videos:")

for video_id, rho in per_video_results[:5]:

    print(
        f"{video_id}: "
        f"{rho:.4f}"
    )


print("\nBest 5 videos:")

for video_id, rho in per_video_results[-5:]:

    print(
        f"{video_id}: "
        f"{rho:.4f}"
    )