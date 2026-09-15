import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "ex1_time"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 1. Basic settings
# =========================
DATA_DIR = RESULTS_DIR
B = 50  # number of replications

files = {
    150: "out_simu_results_ex1_time_B50_s25_n150_N1500.xlsx",
    200: "out_simu_results_ex1_time_B50_s25_n200_N1500.xlsx",
    300: "out_simu_results_ex1_time_B50_s25_n300_N1500.xlsx",
}

model_name = [
    r'$\Theta$-IPOD',
    "Sparse-LTS",
    "Multi-task Lasso",
    "PTL",
    "Trans-Lasso",
    "Trans-PtLR",
    "Trans-CO",
]

colors = [
    '#4E5D6C',
    '#7A8F6A',
    '#B6A36A',
    '#8A6F8F',
    '#9A7B63',
    '#6F9AA6',
    '#BF1D2D',  # Trans-CO - red
]

color_map = dict(zip(model_name, colors))

markers = {
    r'$\Theta$-IPOD': "o",
    "Sparse-LTS": "s",
    "Multi-task Lasso": "^",
    "PTL": "D",
    "Trans-Lasso": "v",
    "Trans-PtLR": "P",
    "Trans-CO": "*",
}
method_cols = {
    r'$\Theta$-IPOD': "time_IPOD",
    "Sparse-LTS": "time_Sparse-LTS",
    "Multi-task Lasso": "time_Multi-task-Lasso",
    "PTL": "time_PTL",
    "Trans-Lasso": "time_Trans-Lasso",
    "Trans-PtLR": "time_Trans-PtLR",
    "Trans-CO": "time_Trans-CO",
}

# =========================
# 2. Read data and summarize runtime
# =========================
records = []

for n, file_name in files.items():
    file_path = os.path.join(DATA_DIR, file_name)

    # 读取原始 50 次重复实验结果
    df = pd.read_excel(file_path, sheet_name="Model Results")

    for method, col in method_cols.items():
        runtime = df[col].dropna().astype(float)

        records.append({
            "n": n,
            "method": method,
            "mean": runtime.mean(),
            "std": runtime.std(ddof=1),
            "se": runtime.std(ddof=1) / np.sqrt(len(runtime)),
            "median": runtime.median(),
            "q25": runtime.quantile(0.25),
            "q75": runtime.quantile(0.75),
        })

summary = pd.DataFrame(records)

# =========================
# 3. Plot: journal/conference style
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(6, 3.6))

x_values = np.array([150, 200, 300])

for method in model_name:
    sub = summary[summary["method"] == method].sort_values("n")

    x = sub["n"].values
    y = sub["median"].values

    yerr_lower = y - sub["q25"].values
    yerr_upper = sub["q75"].values - y
    yerr = np.vstack([yerr_lower, yerr_upper])

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        label=method,
        color=color_map[method],
        marker=markers[method],
        markersize=5.0 if method != "Trans-CO" else 6.5,
        markerfacecolor="white",
        markeredgecolor=color_map[method],
        markeredgewidth=1.0,
        linewidth=1.45 if method != "Trans-CO" else 1.8,
        elinewidth=0.8,
        capsize=2.2,
        capthick=0.8,
        alpha=0.95,
        zorder=3 if method == "Trans-CO" else 2,
    )
ax.set_yscale("log")

ax.set_xlabel(r"Target sample size $n$")
ax.set_ylabel("Runtime (seconds, log scale)")

ax.set_xticks(x_values)
ax.set_xlim(140, 310)

# Cleaner grid
ax.grid(axis="y", which="major", linestyle="--", linewidth=0.45, alpha=0.35)
ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.35, alpha=0.18)
ax.grid(axis="x", visible=False)

# Remove unnecessary borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Put legend outside to avoid crowding the panel
legend = ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True,
    fancybox=False,
    edgecolor="0.85",
    framealpha=1.0,
    borderpad=0.6,
    labelspacing=0.45,
    handlelength=1.7,
)

legend.get_frame().set_linewidth(0.6)

fig.tight_layout()

fig.savefig(RESULTS_DIR / "runtime_ex1_N1500_s25.pdf", bbox_inches="tight")
fig.savefig(RESULTS_DIR / "runtime_ex1_N1500_s25.png", dpi=600, bbox_inches="tight")

plt.show()
