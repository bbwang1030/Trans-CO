import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib.ticker import MaxNLocator
from pathlib import Path


# ============================================================
# 1. Path
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "ex9"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 单独保存 6 张子图
PLOTS_DIR = RESULTS_DIR / "subplots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Simulation settings
# ============================================================

B = 50

s_vals = [25, 75]
r_vals = [2,4,6]
N_vals = [1500]

# ============================================================
# 3. Model settings
#
# 第一项：Excel 文件中的列名
# 第二项：图中显示的名字
# ============================================================

plot_models = [
    ("IPOD", r'$\Theta$-IPOD'),
    ("Sparse-LTS", "Sparse-LTS"),
    ("Multi-task-Lasso", "Multi-task Lasso"),
    ("PTL", "PTL"),
    ("Trans-Lasso", "Trans-Lasso"),
    ("Trans-PtLR", "Trans-PtLR"),
    ("Trans-CO", "Trans-CO"),
]

model_cols = [item[0] for item in plot_models]
model_names = [item[1] for item in plot_models]

num_models = len(plot_models)


# ============================================================
# 4. Colors
#
# 顺序必须和 plot_models 完全一致
# ============================================================

colors = [
    '#4E5D6C',  # Theta-IPOD
    '#7A8F6A',  # Sparse-LTS
    '#B6A36A',  # Multi-task Lasso
    '#8A6F8F',  # PTL
    '#9A7B63',  # Trans-Lasso
    '#6F9AA6',  # Trans-PtLR
    '#BF1D2D',  # Trans-CO
]


# ============================================================
# 5. Publication-quality matplotlib settings
# ============================================================

plt.rcParams.update({

    # Font
    'font.family': 'serif',
    'font.size': 10.0,

    # Axis labels
    'axes.labelsize': 11.0,

    # Tick labels
    'xtick.labelsize': 10.0,
    'ytick.labelsize': 10.0,

    # Legend
    'legend.fontsize': 10.0,

    # Axis
    'axes.linewidth': 0.9,

    # Ticks
    'xtick.major.width': 0.9,
    'ytick.major.width': 0.9,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,

    # Save
    'savefig.dpi': 600,

    # PDF font embedding
    'pdf.fonttype': 42,
    'ps.fonttype': 42,

    # Math font
    'mathtext.fontset': 'stix',
})


# ============================================================
# 6. Read all simulation results
#
# log_mse:
#
# s × n × N × model × replication
#
# = 2 × 3 × 3 × 7 × 50
# ============================================================

log_mse = np.full(
    (
        len(s_vals),
        len(r_vals),
        len(N_vals),
        num_models,
        B
    ),
    np.nan
)


# ============================================================
# Optional F1-score results
# ============================================================

f1_results = []


for s_idx, s in enumerate(s_vals):

    print(f"\nReading results for s = {s}")

    for n_idx, n in enumerate(r_vals):

        for N_idx, N in enumerate(N_vals):

            file = (
                RESULTS_DIR
                / f"out_simu_results_ex9_B{B}_s{s}_n200_N{N}_{n}.xlsx"
            )

            print(f"  Reading: {file.name}")

            if not file.exists():

                raise FileNotFoundError(
                    f"\nCannot find file:\n{file}\n"
                )


            # ------------------------------------------------
            # Read Excel
            # ------------------------------------------------

            df = pd.read_excel(
                file,
                sheet_name='Model Results'
            )


            # ------------------------------------------------
            # Check required model columns
            # ------------------------------------------------

            missing_cols = [
                col
                for col in model_cols
                if col not in df.columns
            ]

            if missing_cols:

                raise ValueError(
                    f"\nMissing columns in {file.name}:\n"
                    f"{missing_cols}"
                )


            # ------------------------------------------------
            # Save log(MSE)
            # ------------------------------------------------

            for model_idx, col in enumerate(model_cols):

                values = (
                    pd.to_numeric(
                        df[col],
                        errors='coerce'
                    )
                    .dropna()
                    .to_numpy()
                )

                if len(values) < B:

                    raise ValueError(
                        f"{file.name}: "
                        f"column '{col}' contains only "
                        f"{len(values)} valid values; "
                        f"B = {B}."
                    )


                log_mse[
                    s_idx,
                    n_idx,
                    N_idx,
                    model_idx,
                    :
                ] = values[:B]


            # ------------------------------------------------
            # Optional F1 scores
            # ------------------------------------------------

            if (
                'f1_score_IPOD' in df.columns
                and
                'f1_score_Trans-CO' in df.columns
            ):

                ipod_f1 = pd.to_numeric(
                    df['f1_score_IPOD'],
                    errors='coerce'
                ).dropna()

                transco_f1 = pd.to_numeric(
                    df['f1_score_Trans-CO'],
                    errors='coerce'
                ).dropna()


                f1_results.append({

                    's': s,
                    'n': n,
                    'N': N,

                    'IPOD_mean':
                        ipod_f1.mean(),

                    'IPOD_std':
                        ipod_f1.std(ddof=0),

                    'TransCO_mean':
                        transco_f1.mean(),

                    'TransCO_std':
                        transco_f1.std(ddof=0),
                })


# ============================================================
# 7. Print F1 results
# ============================================================

if len(f1_results) > 0:

    f1_df = pd.DataFrame(f1_results)

    print("\n")
    print("=" * 80)
    print("F1 SCORE SUMMARY")
    print("=" * 80)

    print(
        f1_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ============================================================
# 8. Create positions
#
# Three groups:
#
# n = 150
# n = 200
# n = 300
#
# Each group contains 7 methods.
# ============================================================

def create_positions(
        num_groups,
        num_models,
        group_gap=1.3
):

    positions = []
    group_centers = []

    current_position = 1.0


    for group_idx in range(num_groups):

        group_positions = []


        for model_idx in range(num_models):

            group_positions.append(
                current_position
            )

            positions.append(
                current_position
            )

            current_position += 1.0


        group_centers.append(
            np.mean(group_positions)
        )

        current_position += group_gap


    return positions, group_centers


positions, group_centers = create_positions(
    num_groups=len(r_vals),
    num_models=num_models,
    group_gap=1.3
)


# ============================================================
# 9. Generate six separate figures
#
# No:
#   (a), (b), ...
#   s = ...
#   N = ...
#
# These should be added by LaTeX subcaption.
#
# Every subplot DOES contain:
#   log(MSE)
# ============================================================

for s_idx, s in enumerate(s_vals):

    for N_idx, N in enumerate(N_vals):


        # ====================================================
        # Create one individual figure
        # ====================================================

        fig, ax = plt.subplots(
            figsize=(4.75, 2.55)
        )


        # ----------------------------------------------------
        # Collect data
        # ----------------------------------------------------

        all_data = []


        for n_idx in range(len(r_vals)):

            for model_idx in range(num_models):

                values = log_mse[
                    s_idx,
                    n_idx,
                    N_idx,
                    model_idx,
                    :
                ]

                values = values[
                    np.isfinite(values)
                ]

                all_data.append(
                    values
                )


        # ----------------------------------------------------
        # Boxplot
        # ----------------------------------------------------

        box = ax.boxplot(

            all_data,

            positions=positions,

            widths=0.58,

            patch_artist=True,

            vert=True,


            # Median
            medianprops={
                'color': 'white',
                'linewidth': 1.3
            },


            # Box
            boxprops={
                'linewidth': 0.85,
                'edgecolor': '#555555'
            },


            # Whisker
            whiskerprops={
                'linewidth': 0.85,
                'color': '#555555'
            },


            # Cap
            capprops={
                'linewidth': 0.85,
                'color': '#555555'
            },


            # Outlier
            flierprops={
                'marker': 'o',
                'markersize': 3.0,
                'markerfacecolor': 'none',
                'markeredgecolor': '#555555',
                'markeredgewidth': 0.7,
                'linestyle': 'none'
            },


            showmeans=False
        )


        # ----------------------------------------------------
        # Colors
        # ----------------------------------------------------

        color_indices = np.tile(
            np.arange(num_models),
            len(r_vals)
        )


        for patch, color_idx in zip(
                box['boxes'],
                color_indices
        ):

            patch.set_facecolor(
                colors[color_idx]
            )


        # ----------------------------------------------------
        # X axis
        # ----------------------------------------------------

        ax.set_xticks(
            group_centers
        )

        ax.set_xticklabels(
            [str(n) for n in r_vals]
        )

        ax.tick_params(
            axis='x',
            which='major',
            labelsize=10.0,
            pad=2
        )


        # ----------------------------------------------------
        # Y axis
        # ----------------------------------------------------

        ax.yaxis.set_major_locator(
            MaxNLocator(
                nbins=6
            )
        )

        ax.tick_params(
            axis='y',
            which='major',
            labelsize=10.0,
            pad=2
        )


        # ====================================================
        # IMPORTANT:
        #
        # Every single subplot has log(MSE)
        # ====================================================

        ax.set_ylabel(
            'log(MSE)',
            fontsize=11.0,
            labelpad=3.5
        )


        # ----------------------------------------------------
        # Grid
        # ----------------------------------------------------

        ax.set_axisbelow(True)

        ax.grid(
            axis='y',
            linestyle='-',
            linewidth=0.5,
            alpha=0.25
        )


        # ----------------------------------------------------
        # Axis border
        # ----------------------------------------------------

        for spine in ax.spines.values():

            spine.set_linewidth(
                0.85
            )

            spine.set_color(
                '#777777'
            )


        # ----------------------------------------------------
        # Layout
        # ----------------------------------------------------

        fig.subplots_adjust(
            left=0.18,
            right=0.98,
            bottom=0.16,
            top=0.97
        )


        # ----------------------------------------------------
        # Output filenames
        #

        # ----------------------------------------------------

        pdf_file = (
            PLOTS_DIR
            / f"ex9_s{s}_N{N}.pdf"
        )

        png_file = (
            PLOTS_DIR
            / f"ex9_s{s}_N{N}.png"
        )


        # ----------------------------------------------------
        # Save PDF
        # ----------------------------------------------------

        fig.savefig(
            pdf_file,
            bbox_inches='tight',
            pad_inches=0.03
        )


        # ----------------------------------------------------
        # Save PNG
        # ----------------------------------------------------

        fig.savefig(
            png_file,
            dpi=600,
            bbox_inches='tight',
            pad_inches=0.03
        )


        print(
            f"Saved subplot: "
            f"s={s}, N={N}"
        )

        print(
            f"    PDF: {pdf_file}"
        )

        print(
            f"    PNG: {png_file}"
        )


        plt.close(fig)


# ============================================================
# 10. Generate a separate legend figure
#
# Seven methods in one horizontal row.
# ============================================================

legend_handles = [

    mpatches.Patch(
        facecolor=colors[i],
        edgecolor='#555555',
        linewidth=0.6,
        label=model_names[i]
    )

    for i in range(num_models)
]


# ============================================================
# Legend-only figure
# ============================================================

legend_fig = plt.figure(
    figsize=(7.2, 0.50)
)


legend_fig.legend(

    handles=legend_handles,

    loc='center',

    ncol=7,

    fontsize=10.0,

    frameon=False,

    handlelength=1.25,
    handleheight=0.9,

    handletextpad=0.35,

    columnspacing=1.0,

    borderaxespad=0.0
)


# ============================================================
# Save legend
# ============================================================

legend_pdf = (
    PLOTS_DIR
    / "ex9_legend.pdf"
)

legend_png = (
    PLOTS_DIR
    / "ex9_legend.png"
)


legend_fig.savefig(
    legend_pdf,
    bbox_inches='tight',
    pad_inches=0.03
)


legend_fig.savefig(
    legend_png,
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.03
)


plt.close(legend_fig)


# ============================================================
# 11. Final message
# ============================================================

print("\n")
print("=" * 80)
print("ALL FIGURES SAVED")
print("=" * 80)

print("\nSix subplots:")

for s in s_vals:
    for N in N_vals:

        print(
            PLOTS_DIR
            / f"ex9_s{s}_N{N}.pdf"
        )

