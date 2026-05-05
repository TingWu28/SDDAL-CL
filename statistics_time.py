import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 0) Configuration
# =========================================================
beamshape = 'rec'
seeds     = [1, 2, 3, 4, 5, 6]

AXIS_LABEL_FONT_SIZE = 18
TITLE_FONT_SIZE      = 20
LEGEND_FONT_SIZE     = 15
TICK_FONT_SIZE       = 14

SEED_MARKERS = ["o", "s", "^", "D", "v"]

# =========================================================
# 1) Path settings
# =========================================================
base_dir     = os.path.dirname(os.path.abspath(__file__))
baseline_dir = os.path.join(base_dir, "SDDAL_InShaPe_cl")
cl_dir       = os.path.join(base_dir, "SDDAL_InShaPe_cl")   # update when CL variant exists

# Experiment folder name pattern inside each dir
def exp_folder(parent_dir, seed, suffix=""):
    return os.path.join(parent_dir, f"Design_{beamshape}_{seed}{suffix}")

# =========================================================
# 2) Load timing_log.csv for each strategy / seed
#    Returns dict: seed -> DataFrame (columns: dataset_size,
#    cumul_trainer_s, per_round_trainer_s)
#    Missing files are skipped with a warning.
# =========================================================
def load_timing(parent_dir, label, suffix=""):
    data = {}
    for seed in seeds:
        csv_path = os.path.join(exp_folder(parent_dir, seed, suffix), "timing_log.csv")
        if not os.path.isfile(csv_path):
            print(f"[SKIP] {label} seed={seed}: timing_log.csv not found at {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        required = {"dataset_size", "cumul_trainer_s", "per_round_trainer_s"}
        if not required.issubset(df.columns):
            print(f"[SKIP] {label} seed={seed}: missing columns in {csv_path}")
            continue
        data[seed] = df
        print(f"  Loaded {label} seed={seed}: {len(df)} rounds")
    return data

print("=" * 60)
print("Loading timing logs...")
print("=" * 60)
baseline_data = load_timing(baseline_dir, "baseline", suffix="")
cl_data       = load_timing(cl_dir,       "CL",       suffix="_cl")

# =========================================================
# 3) Helpers
# =========================================================
def interp_to_common_x(data_dict, metric_col):
    """
    Interpolate each seed's curve onto a common dataset_size grid
    so we can average across seeds cleanly.
    Returns (x_common, matrix) where matrix rows = seeds.
    """
    all_x = sorted(set(
        x for df in data_dict.values() for x in df["dataset_size"].tolist()
    ))
    if len(all_x) == 0:
        return np.array([]), np.full((0, 0), np.nan)

    x_common = np.array(all_x, dtype=float)
    rows = []
    for df in data_dict.values():
        x = df["dataset_size"].values.astype(float)
        y = df[metric_col].values.astype(float)
        y_interp = np.interp(x_common, x, y,
                             left=np.nan, right=np.nan)
        rows.append(y_interp)
    return x_common, np.array(rows)


def safe_mean_std(matrix):
    """Column-wise mean and std ignoring NaN."""
    with np.errstate(all='ignore'):
        mean = np.nanmean(matrix, axis=0)
        std  = np.nanstd(matrix,  axis=0)
    return mean, std

# =========================================================
# 4) Plot helpers
# =========================================================
def plot_all_seeds(ax, data_dict, metric_col, color, label_prefix):
    seed_list = list(data_dict.keys())
    for i, seed in enumerate(seed_list):
        df  = data_dict[seed]
        x   = df["dataset_size"].values
        y   = df[metric_col].values
        ax.plot(
            x, y,
            marker=SEED_MARKERS[i % len(SEED_MARKERS)],
            markersize=5,
            linewidth=1.8,
            color=color,
            alpha=0.55,
            label=f"{label_prefix} seed={seed}",
        )


def plot_mean(ax, data_dict, metric_col, color, label):
    if not data_dict:
        return
    x_common, matrix = interp_to_common_x(data_dict, metric_col)
    if x_common.size == 0:
        return
    mean, _ = safe_mean_std(matrix)
    valid = np.isfinite(mean)
    ax.plot(
        x_common[valid], mean[valid],
        marker="o",
        markersize=6,
        linewidth=2.5,
        color=color,
        alpha=0.95,
        label=label,
    )


def finalise(ax, title, ylabel, save_path):
    ax.set_xlabel("Dataset size (# training samples)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# =========================================================
# 5) Graph 1 — cumul_trainer_s, all individual seeds
# =========================================================
fig, ax = plt.subplots(figsize=(10, 6))
plot_all_seeds(ax, baseline_data, "cumul_trainer_s", color="green",  label_prefix="baseline")
plot_all_seeds(ax, cl_data,       "cumul_trainer_s", color="steelblue", label_prefix="CL")
finalise(ax,
         title="Cumulative trainer time — all seeds",
         ylabel="Cumulative trainer time (s)",
         save_path=os.path.join(base_dir, "time_all_seeds_cumul_trainer.png"))

# =========================================================
# 6) Graph 2 — cumul_trainer_s, mean only
# =========================================================
fig, ax = plt.subplots(figsize=(10, 6))
plot_mean(ax, baseline_data, "cumul_trainer_s", color="green",     label="baseline mean")
plot_mean(ax, cl_data,       "cumul_trainer_s", color="steelblue", label="CL mean")
finalise(ax,
         title="Cumulative trainer time — mean",
         ylabel="Cumulative trainer time (s)",
         save_path=os.path.join(base_dir, "time_mean_cumul_trainer.png"))

# =========================================================
# 7) Graph 3 — per_round_trainer_s, all individual seeds
# =========================================================
fig, ax = plt.subplots(figsize=(10, 6))
plot_all_seeds(ax, baseline_data, "per_round_trainer_s", color="green",     label_prefix="baseline")
plot_all_seeds(ax, cl_data,       "per_round_trainer_s", color="steelblue", label_prefix="CL")
finalise(ax,
         title="Per-round trainer time — all seeds",
         ylabel="Trainer time per round (s)",
         save_path=os.path.join(base_dir, "time_all_seeds_per_trainer.png"))

# =========================================================
# 8) Graph 4 — per_round_trainer_s, mean only
# =========================================================
fig, ax = plt.subplots(figsize=(10, 6))
plot_mean(ax, baseline_data, "per_round_trainer_s", color="green",     label="baseline mean")
plot_mean(ax, cl_data,       "per_round_trainer_s", color="steelblue", label="CL mean")
finalise(ax,
         title="Per-round trainer time — mean",
         ylabel="Trainer time per round (s)",
         save_path=os.path.join(base_dir, "time_mean_per_trainer.png"))

print("\nAll timing figures generated.")
