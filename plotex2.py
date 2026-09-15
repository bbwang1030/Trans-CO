import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib.ticker import MaxNLocator
from pathlib import Path


# ============================================================
# 1. Path
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "ex2_K_O"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 单独保存每张子图和 legend
PLOTS_DIR = RESULTS_DIR / "subplots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Simulation settings
#
# Example 2:
#   s   = 30, 70
#   rho = 0.05, 0.1
#   K   = 2, 4, 6, 8, 10
#
# 因此一共生成：
#   2 × 2 = 4 张子图
# ============================================================

B = 50

s_vals = [30, 70]
rho_vals = [0.05, 0.1]
K_vals = [2, 4, 6, 8, 10]


# ============================================================
# 3. Model settings
#
# 第一项：Excel 文件中的列名
# 第二项：图中显示的名称
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
# 顺序与 plot_models 完全一致
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

    # Axis lines
    'axes.linewidth': 0.9,

    # Ticks
    'xtick.major.width': 0.9,
    'ytick.major.width': 0.9,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,

    # High-resolution PNG
    'savefig.dpi': 600,

    # Better PDF font embedding
    'pdf.fonttype': 42,
    'ps.fonttype': 42,

    # Math font
    'mathtext.fontset': 'stix',
})


# ============================================================
# 6. Read all simulation results
#
# log_mse dimensions:
#
# s × K × rho × model × replication
#
# = 2 × 5 × 2 × 7 × 50
# ============================================================

log_mse = np.full(
    (
        len(s_vals),
        len(K_vals),
        len(rho_vals),
        num_models,
        B
    ),
    np.nan
)


# Optional F1-score results
f1_results = []


for s_idx, s in enumerate(s_vals):

    print(f"\nReading results for s = {s}")

    for K_idx, K in enumerate(K_vals):

        for rho_idx, rho in enumerate(rho_vals):

            # ------------------------------------------------
            # Excel filename
            #
            # Example:
            # out_simu_results_ex2_s30_o0.05_K2.xlsx
            # ------------------------------------------------

            file = (
                RESULTS_DIR
                / f"out_simu_results_ex2_s{s}_o{rho}_K{K}.xlsx"
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
                    K_idx,
                    rho_idx,
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
                    'K': K,
                    'rho': rho,

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
# 8. Helper function
#
# 每个 K 是一个 group
# 每个 group 里面有 7 个方法
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
    num_groups=len(K_vals),
    num_models=num_models,
    group_gap=1.3
)


# ============================================================
# 9. Generate separate subplot figures
#
# 每张图：
#
#   - 没有标题
#   - 没有 (a), (b), ...
#   - 没有 s=..., rho=...
#   - 有 log(MSE)
#   - 有 K = 2,4,6,8,10 的横坐标
#
# (a)-(d)、s 和 rho 全部交给 LaTeX subfigure
# ============================================================

for s_idx, s in enumerate(s_vals):

    for rho_idx, rho in enumerate(rho_vals):


        # ====================================================
        # Create individual figure
        # ====================================================

        fig, ax = plt.subplots(
            figsize=(4.75, 2.55)
        )


        # ----------------------------------------------------
        # Collect data
        # ----------------------------------------------------

        all_data = []


        for K_idx in range(len(K_vals)):

            for model_idx in range(num_models):

                values = log_mse[
                    s_idx,
                    K_idx,
                    rho_idx,
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


            # Box border
            boxprops={
                'linewidth': 0.85,
                'edgecolor': '#555555'
            },


            # Whiskers
            whiskerprops={
                'linewidth': 0.85,
                'color': '#555555'
            },


            # Caps
            capprops={
                'linewidth': 0.85,
                'color': '#555555'
            },


            # Outliers
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
        # Fill box colors
        # ----------------------------------------------------

        color_indices = np.tile(
            np.arange(num_models),
            len(K_vals)
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
        #
        # K = 2, 4, 6, 8, 10
        # ----------------------------------------------------

        ax.set_xticks(
            group_centers
        )

        ax.set_xticklabels(
            [str(K) for K in K_vals]
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


        # ----------------------------------------------------
        # Every subplot has y-axis label
        # ----------------------------------------------------

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
        # IMPORTANT:
        #
        # No title here.
        #
        # s, rho and (a)-(d) will be added in LaTeX.
        # ----------------------------------------------------


        # ----------------------------------------------------
        # Layout
        # ----------------------------------------------------

        fig.subplots_adjust(
            left=0.19,
            right=0.98,
            bottom=0.16,
            top=0.98
        )


        # ----------------------------------------------------
        # Output filenames
        #
        # Example:
        #
        # ex2_s30_rho0.05.png
        # ex2_s30_rho0.05.pdf
        # ----------------------------------------------------

        png_file = (
            PLOTS_DIR
            / f"ex2_s{s}_rho{rho}.png"
        )

        pdf_file = (
            PLOTS_DIR
            / f"ex2_s{s}_rho{rho}.pdf"
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


        # ----------------------------------------------------
        # Save PDF
        # ----------------------------------------------------

        fig.savefig(
            pdf_file,
            bbox_inches='tight',
            pad_inches=0.03
        )


        print(
            f"Saved subplot: "
            f"s={s}, rho={rho}"
        )

        print(
            f"    PNG: {png_file}"
        )

        print(
            f"    PDF: {pdf_file}"
        )


        plt.close(fig)


# ============================================================
# 10. Generate separate legend figure
#
# Seven methods
# One horizontal row
#
# No title
# No figure number
#
# LaTeX 可以把它直接放在所有 subfigures 下方
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
    figsize=(7.2, 0.55)
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
# Legend filenames
# ============================================================

legend_png = (
    PLOTS_DIR
    / "ex2_legend.png"
)

legend_pdf = (
    PLOTS_DIR
    / "ex2_legend.pdf"
)


# ============================================================
# Save legend PNG
# ============================================================

legend_fig.savefig(
    legend_png,
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.03
)


# ============================================================
# Save legend PDF
# ============================================================

legend_fig.savefig(
    legend_pdf,
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


print("\nSubplots:")

for s in s_vals:

    for rho in rho_vals:

        print(
            PLOTS_DIR
            / f"ex2_s{s}_rho{rho}.png"
        )

        print(
            PLOTS_DIR
            / f"ex2_s{s}_rho{rho}.pdf"
        )


print("\nLegend:")

print(
    legend_png
)

print(
    legend_pdf
)
