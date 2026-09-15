from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# 1. Paths
# ============================================================

RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "insurance_region_raw_charge_mape"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

summary_path = (
    RESULTS_DIR
    / "insurance_region_summary_results.csv"
)

detail_path = (
    RESULTS_DIR
    / "insurance_region_detail_results.csv"
)

out_dir = (
    RESULTS_DIR
    / "insurance_region_publication_plots"
)

out_dir.mkdir(
    parents=True,
    exist_ok=True,
)

qq_dir = (
    RESULTS_DIR
    / "qq_plots"
)

qq_dir.mkdir(
    parents=True,
    exist_ok=True,
)

select_dir = (
    qq_dir
    / "select"
)

select_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. Main figure size
# ============================================================

FIG_W = 12.5
FIG_H = 2.5

DPI = 600


# ============================================================
# 3. QQ plot settings
# ============================================================

# ------------------------------------------------------------
# Original QQ plot
# ------------------------------------------------------------

QQ_FIG_W = 6.2
QQ_FIG_H = 5.8

QQ_DPI = 600

# Theoretical quantiles / Sample quantiles
QQ_LABEL_FONTSIZE = 18

# Tick labels
QQ_TICK_FONTSIZE = 15

# p-value on original QQ plots
QQ_PVALUE_FONTSIZE = 20

QQ_PVALUE_X = 0.97
QQ_PVALUE_Y = 0.08

QQ_MARKER_SIZE = 6.0
QQ_LINEWIDTH = 1.6


# ------------------------------------------------------------
# Selected QQ plot
#
# These settings control the p-value on the FINAL four plots.
# ------------------------------------------------------------

SELECT_QQ_DPI = 600

# Slightly larger than the original QQ p-value
SELECT_PVALUE_FONTSIZE = 12

# Position in the selected PNG
# smaller Y = lower
SELECT_PVALUE_X = 0.965
SELECT_PVALUE_Y = 0.17


# ============================================================
# 4. Matplotlib settings
# ============================================================

plt.rcParams.update({

    # Font
    "font.family": "serif",
    "font.size": 11.0,

    # Axis labels and titles
    "axes.labelsize": 13.0,
    "axes.titlesize": 13.0,

    # Tick labels
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,

    # Legend
    "legend.fontsize": 10.8,

    # Axis lines
    "axes.linewidth": 0.9,

    # Ticks
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,

    # Output
    "savefig.dpi": DPI,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Math font
    "mathtext.fontset": "stix",
})


# ============================================================
# 5. Function for ORIGINAL QQ plots
# ============================================================

def save_original_qq_plot(
    residuals,
    target_domain,
    algorithm,
    rep,
    shapiro_pvalue=None,
):
    """
    Generate and save an original high-resolution QQ plot.

    Parameters
    ----------
    residuals : array-like
        Residuals for the current repetition.

    target_domain : str
        Target region.

    algorithm : str
        Algorithm name.

    rep : int
        Actual repetition number.

    shapiro_pvalue : float or None
        Shapiro-Wilk p-value. If None, calculate here.

    Returns
    -------
    Path
        Saved QQ plot path.
    """

    # --------------------------------------------------------
    # Clean residuals
    # --------------------------------------------------------

    residuals = np.asarray(
        residuals,
        dtype=float,
    ).reshape(-1)

    residuals = residuals[
        np.isfinite(residuals)
    ]

    if residuals.size < 3:

        raise ValueError(
            "At least 3 finite residuals are required "
            "to generate a QQ plot."
        )

    # --------------------------------------------------------
    # Shapiro p-value
    # --------------------------------------------------------

    if shapiro_pvalue is None:

        _, shapiro_pvalue = stats.shapiro(
            residuals
        )

    shapiro_pvalue = float(
        shapiro_pvalue
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            QQ_FIG_W,
            QQ_FIG_H,
        )
    )

    stats.probplot(
        residuals,
        dist="norm",
        plot=ax,
    )

    # --------------------------------------------------------
    # QQ points
    # --------------------------------------------------------

    if len(ax.lines) >= 1:

        qq_points = ax.lines[0]

        qq_points.set_markersize(
            QQ_MARKER_SIZE
        )

        qq_points.set_marker(
            "o"
        )

        qq_points.set_linestyle(
            "none"
        )

        qq_points.set_alpha(
            0.85
        )

    # --------------------------------------------------------
    # Reference line
    # --------------------------------------------------------

    if len(ax.lines) >= 2:

        qq_line = ax.lines[1]

        qq_line.set_linewidth(
            QQ_LINEWIDTH
        )

        qq_line.set_alpha(
            0.90
        )

    # --------------------------------------------------------
    # Remove scipy default title
    # --------------------------------------------------------

    ax.set_title("")

    # --------------------------------------------------------
    # Large axis labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "Theoretical quantiles",
        fontsize=QQ_LABEL_FONTSIZE,
        labelpad=9,
    )

    ax.set_ylabel(
        "Sample quantiles",
        fontsize=QQ_LABEL_FONTSIZE,
        labelpad=9,
    )

    # --------------------------------------------------------
    # Large tick labels
    # --------------------------------------------------------

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=QQ_TICK_FONTSIZE,
        width=1.0,
        length=4.0,
    )

    # --------------------------------------------------------
    # Spines
    # --------------------------------------------------------

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.spines[
        "left"
    ].set_linewidth(1.0)

    ax.spines[
        "bottom"
    ].set_linewidth(1.0)

    ax.grid(False)

    # --------------------------------------------------------
    # Shapiro p-value on original QQ plot
    # --------------------------------------------------------

    ax.text(
        QQ_PVALUE_X,
        QQ_PVALUE_Y,
        (
            "Shapiro p-value = "
            f"{shapiro_pvalue:.2f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=QQ_PVALUE_FONTSIZE,
        fontfamily="serif",
        zorder=10,
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor="white",
            edgecolor="none",
            alpha=0.88,
        ),
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.18,
        right=0.97,
        bottom=0.18,
        top=0.97,
    )

    # --------------------------------------------------------
    # Output filename
    #
    # IMPORTANT:
    # rep is the actual repetition number.
    # Nothing is fixed to rep=0.
    # --------------------------------------------------------

    qq_path = (
        qq_dir
        / (
            f"qq_{algorithm}_"
            f"target_{target_domain}_"
            f"rep_{int(rep)}.png"
        )
    )

    fig.savefig(
        qq_path,
        dpi=QQ_DPI,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close(fig)

    return qq_path


# ============================================================
# 6. Helper for FINAL selected QQ plots
# ============================================================

def save_selected_qq_with_pvalue(
    src_path,
    dst_path,
    shapiro_pvalue,
):
    """
    Read the selected original QQ plot and explicitly add
    a larger Shapiro p-value label to the bottom-right corner.

    This guarantees that the four figures in qq_plots/select
    contain a clearly visible p-value.
    """

    src_path = Path(
        src_path
    )

    dst_path = Path(
        dst_path
    )

    if not src_path.exists():

        raise FileNotFoundError(
            f"QQ plot does not exist: {src_path}"
        )

    # --------------------------------------------------------
    # Read original high-resolution image
    # --------------------------------------------------------

    img = plt.imread(
        src_path
    )

    h, w = img.shape[:2]

    # --------------------------------------------------------
    # Preserve approximately the original physical size
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(
            w / SELECT_QQ_DPI,
            h / SELECT_QQ_DPI,
        ),
        dpi=SELECT_QQ_DPI,
    )

    # Fill the complete figure with the QQ image
    ax = fig.add_axes([
        0,
        0,
        1,
        1,
    ])

    ax.imshow(
        img
    )

    ax.axis(
        "off"
    )

    # --------------------------------------------------------
    # Larger Shapiro p-value for SELECTED plot
    #
    # The opaque/semi-opaque white box also helps cover any
    # smaller p-value text already present underneath.
    # --------------------------------------------------------

    ax.text(
        SELECT_PVALUE_X,
        SELECT_PVALUE_Y,
        (
            "Shapiro p-value = "
            f"{float(shapiro_pvalue):.2f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=SELECT_PVALUE_FONTSIZE,
        fontfamily="serif",
        zorder=20,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor="none",
            alpha=0.96,
        ),
    )

    # --------------------------------------------------------
    # Save selected figure
    # --------------------------------------------------------

    fig.savefig(
        dst_path,
        dpi=SELECT_QQ_DPI,
        pad_inches=0,
    )

    plt.close(
        fig
    )


# ============================================================
# 7. Read summary results
# ============================================================

df = pd.read_csv(
    summary_path
)

df["algorithm"] = df[
    "algorithm"
].replace({
    "Multi-task-Lasso":
        "Multi-task Lasso",

    "IPOD":
        r"$\Theta$-IPOD",
})


if "target_domain" in df.columns:

    domain_col = (
        "target_domain"
    )

elif "target_cut" in df.columns:

    domain_col = (
        "target_cut"
    )

else:

    raise ValueError(
        "Cannot find target domain column: "
        "target_domain or target_cut"
    )


df_plot = df[
    df[domain_col]
    .astype(str)
    != "ALL"
].copy()


# ============================================================
# 8. Domain and method order
# ============================================================

region_order = [
    "northeast",
    "northwest",
    "southeast",
    "southwest",
]


domains_in_data = (
    df_plot[
        domain_col
    ]
    .dropna()
    .unique()
    .tolist()
)


domain_order = [

    d
    for d in region_order
    if d in domains_in_data

] + [

    d
    for d in domains_in_data
    if d not in region_order

]


preferred_method_order = [

    r"$\Theta$-IPOD",

    "Sparse-LTS",

    "Multi-task Lasso",

    "PTL",

    "Trans-Lasso",

    "Trans-PtLR",

    "Trans-CO",
]


available_methods = (
    df_plot[
        "algorithm"
    ]
    .unique()
)


method_order = [

    m
    for m in preferred_method_order
    if m in available_methods

]


method_order += [

    m
    for m in available_methods
    if m not in method_order

]


# ============================================================
# 9. Helper functions
# ============================================================

def make_pivot(
    data,
    value_col,
):

    table = (

        data
        .dropna(
            subset=[
                value_col
            ]
        )
        .pivot(
            index=domain_col,
            columns="algorithm",
            values=value_col,
        )
        .reindex(
            domain_order
        )
        .reindex(
            columns=method_order
        )

    )

    return table.dropna(
        axis=1,
        how="all",
    )


def make_error_pivot(
    data,
    sd_col,
):

    if (
        sd_col is None
        or sd_col not in data.columns
    ):

        return None

    return (

        data
        .pivot(
            index=domain_col,
            columns="algorithm",
            values=sd_col,
        )
        .reindex(
            domain_order
        )
        .reindex(
            columns=method_order
        )

    )


def save_png_pdf(
    fig,
    output_prefix,
):

    fig.savefig(
        out_dir
        / f"{output_prefix}.png",
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    fig.savefig(
        out_dir
        / f"{output_prefix}.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )


# ============================================================
# 10. Plot grouped bar chart
# ============================================================

def plot_grouped_bar(
    data,
    value_col,
    sd_col,
    ylabel,
    xlabel,
    output_prefix,
    lower_is_better=True,
):

    table = make_pivot(
        data,
        value_col,
    )

    err_table = make_error_pivot(
        data,
        sd_col,
    )

    fig, ax = plt.subplots(
        figsize=(
            FIG_W,
            FIG_H,
        )
    )

    x = np.arange(
        len(
            table.index
        )
    )

    n_methods = len(
        table.columns
    )

    width = (
        0.82
        / max(
            n_methods,
            1,
        )
    )

    for i, method in enumerate(
        table.columns
    ):

        offset = (
            i
            - (
                n_methods - 1
            ) / 2
        ) * width

        y = (
            table[
                method
            ]
            .to_numpy(
                dtype=float
            )
        )

        yerr = None

        if (
            err_table is not None
            and method
            in err_table.columns
        ):

            yerr = (
                err_table
                .loc[
                    table.index,
                    method,
                ]
                .to_numpy(
                    dtype=float
                )
            )

        bars = ax.bar(
            x + offset,
            y,
            width=width,
            label=method,
            yerr=yerr,
            capsize=2.8,
            alpha=0.88,
            linewidth=0.7,
            zorder=3,
        )

        if method == "Trans-CO":

            for bar in bars:

                bar.set_hatch(
                    "//"
                )

                bar.set_linewidth(
                    1.2
                )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        table.index,
        rotation=0,
        ha="center",
        fontsize=11.5,
    )

    ax.set_xlabel(
        xlabel,
        fontsize=13,
        labelpad=2,
    )

    # --------------------------------------------------------
    # Y axis
    # --------------------------------------------------------

    ax.set_ylabel(
        ylabel,
        fontsize=13,
        labelpad=5,
    )

    ax.tick_params(
        axis="y",
        labelsize=11.5,
    )

    values = table.to_numpy(
        dtype=float
    )

    y_max = np.nanmax(
        values
    )

    ax.set_ylim(
        0,
        y_max * 1.22,
    )

    # --------------------------------------------------------
    # Grid and spines
    # --------------------------------------------------------

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.25,
        zorder=0,
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    # --------------------------------------------------------
    # Save without legend
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.09,
        right=0.995,
        top=0.95,
        bottom=0.28,
    )

    save_png_pdf(
        fig,
        f"noleg_{output_prefix}",
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.45,
        ),
        ncol=min(
            len(
                table.columns
            ),
            7,
        ),
        frameon=False,
        fontsize=10.3,
        title_fontsize=11,
        handlelength=1.3,
        handletextpad=0.4,
        columnspacing=0.85,
    )

    # --------------------------------------------------------
    # Save with legend
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.09,
        right=0.995,
        top=0.95,
        bottom=0.48,
    )

    save_png_pdf(
        fig,
        output_prefix,
    )

    plt.close(
        fig
    )


# ============================================================
# 11. Plot grouped point chart
# ============================================================

def plot_grouped_point(
    data,
    value_col,
    sd_col,
    ylabel,
    xlabel,
    output_prefix,
    lower_is_better=True,
    bounded_01=False,
):

    table = make_pivot(
        data,
        value_col,
    )

    err_table = make_error_pivot(
        data,
        sd_col,
    )

    fig, ax = plt.subplots(
        figsize=(
            FIG_W,
            FIG_H,
        )
    )

    x = np.arange(
        len(
            table.index
        )
    )

    n_methods = len(
        table.columns
    )

    offset_width = (
        0.72
        / max(
            n_methods,
            1,
        )
    )

    for i, method in enumerate(
        table.columns
    ):

        offset = (
            i
            - (
                n_methods - 1
            ) / 2
        ) * offset_width

        y = (
            table[
                method
            ]
            .to_numpy(
                dtype=float
            )
        )

        yerr = None

        if (
            err_table is not None
            and method
            in err_table.columns
        ):

            yerr = (
                err_table
                .loc[
                    table.index,
                    method,
                ]
                .to_numpy(
                    dtype=float
                )
            )

        marker = (
            "D"
            if method == "Trans-CO"
            else "o"
        )

        markersize = (
            6.5
            if method == "Trans-CO"
            else 5.0
        )

        linewidth = (
            1.2
            if method == "Trans-CO"
            else 1.0
        )

        ax.errorbar(
            x + offset,
            y,
            yerr=yerr,
            fmt=marker,
            markersize=markersize,
            capsize=3.0,
            elinewidth=1.0,
            linewidth=linewidth,
            label=method,
            zorder=(
                4
                if method == "Trans-CO"
                else 3
            ),
        )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        table.index,
        rotation=0,
        ha="center",
        fontsize=11.5,
    )

    ax.set_xlabel(
        xlabel,
        fontsize=13,
        labelpad=2,
    )

    # --------------------------------------------------------
    # Y axis
    # --------------------------------------------------------

    ax.set_ylabel(
        ylabel,
        fontsize=13,
        labelpad=5,
    )

    ax.tick_params(
        axis="y",
        labelsize=11.5,
    )

    values = table.to_numpy(
        dtype=float
    )

    y_min = np.nanmin(
        values
    )

    y_max = np.nanmax(
        values
    )

    margin = (
        0.18
        * (
            y_max
            - y_min
            + 1e-8
        )
    )

    if bounded_01:

        ax.set_ylim(
            max(
                0,
                y_min - margin,
            ),
            min(
                1.0,
                y_max + margin,
            ),
        )

    else:

        ax.set_ylim(
            y_min - margin,
            y_max + margin,
        )

    # --------------------------------------------------------
    # Grid and spines
    # --------------------------------------------------------

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.25,
        zorder=0,
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    # --------------------------------------------------------
    # Save without legend
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.09,
        right=0.995,
        top=0.95,
        bottom=0.28,
    )

    save_png_pdf(
        fig,
        f"noleg_{output_prefix}",
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.45,
        ),
        ncol=min(
            len(
                table.columns
            ),
            7,
        ),
        frameon=False,
        fontsize=10.3,
        title_fontsize=11,
        handletextpad=0.4,
        columnspacing=0.85,
    )

    # --------------------------------------------------------
    # Save with legend
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.09,
        right=0.995,
        top=0.95,
        bottom=0.48,
    )

    save_png_pdf(
        fig,
        output_prefix,
    )

    plt.close(
        fig
    )


# ============================================================
# 12. Generate main plots
# ============================================================

plot_grouped_bar(
    data=df_plot,
    value_col="mean_mape_charge",
    sd_col="sd_mape_charge",
    ylabel=r"$\mathrm{MAPE}_{\mathrm{te}}$ (%)",
    xlabel="Target region",
    output_prefix="insurance_region_test_MAPE_bar",
    lower_is_better=True,
)


plot_grouped_bar(
    data=df_plot,
    value_col="mean_MAPE_fit_charge",
    sd_col="sd_MAPE_fit_charge",
    ylabel=r"$\mathrm{MAPE}_{\mathrm{tr}}$ (%)",
    xlabel="Target region",
    output_prefix="insurance_region_train_MAPE_bar",
    lower_is_better=True,
)


plot_grouped_point(
    data=df_plot,
    value_col="mean_R2_train",
    sd_col="sd_R2_train",
    ylabel=r"$\mathrm{R}^2$",
    xlabel="Target region",
    output_prefix="insurance_region_R2_point",
    lower_is_better=False,
    bounded_01=True,
)


# ============================================================
# 13. Select best Trans-CO QQ plot
#
# For EACH region:
#
#   1. Look at ALL available Trans-CO repetitions.
#   2. Find the largest Shapiro p-value.
#   3. Obtain the actual rep corresponding to that maximum.
#   4. Load that QQ plot.
#   5. Add a larger p-value label to the bottom-right.
#   6. Save into qq_plots/select.
#
# NO rep is hard-coded.
# ============================================================

detail_df = pd.read_csv(
    detail_path
)


# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------

required_cols = [
    "target_domain",
    "algorithm",
    "rep",
    "Shapiro_pvalue",
]


missing_cols = [
    c
    for c in required_cols
    if c not in detail_df.columns
]


if missing_cols:

    raise ValueError(
        "Missing required columns in detail results: "
        f"{missing_cols}"
    )


# ------------------------------------------------------------
# Convert numeric fields explicitly
# ------------------------------------------------------------

detail_df["rep"] = pd.to_numeric(
    detail_df["rep"],
    errors="coerce",
)

detail_df["Shapiro_pvalue"] = pd.to_numeric(
    detail_df["Shapiro_pvalue"],
    errors="coerce",
)


# ------------------------------------------------------------
# Valid Trans-CO rows
# ------------------------------------------------------------

valid_mask = (
    (
        detail_df[
            "algorithm"
        ].astype(str)
        == "Trans-CO"
    )
    &
    detail_df[
        "rep"
    ].notna()
    &
    detail_df[
        "Shapiro_pvalue"
    ].notna()
)


# ------------------------------------------------------------
# Optional error filtering
# ------------------------------------------------------------

if "error" in detail_df.columns:

    valid_mask &= (
        detail_df[
            "error"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        == ""
    )


transco_df = detail_df[
    valid_mask
].copy()


if transco_df.empty:

    raise ValueError(
        "No valid Trans-CO rows with "
        "Shapiro_pvalue were found."
    )


transco_df[
    "rep"
] = (
    transco_df[
        "rep"
    ]
    .astype(int)
)


# ============================================================
# Inspect and select best rep
# ============================================================

print("\n")
print("=" * 80)
print("SELECTING BEST TRANS-CO QQ PLOT FOR EACH REGION")
print("=" * 80)


for region in region_order:

    # --------------------------------------------------------
    # All valid repetitions for current region
    # --------------------------------------------------------

    sub = transco_df[
        transco_df[
            "target_domain"
        ].astype(str)
        == region
    ].copy()


    if sub.empty:

        print(
            "[WARNING] "
            "No valid Trans-CO result found "
            f"for region: {region}"
        )

        continue


    # --------------------------------------------------------
    # Number of available repetitions
    # --------------------------------------------------------

    n_reps = (
        sub[
            "rep"
        ]
        .nunique()
    )


    print(
        f"\n[{region}] "
        f"Valid repetitions = {n_reps}"
    )


    # --------------------------------------------------------
    # Sort ALL reps by Shapiro p-value
    # --------------------------------------------------------

    sub = (
        sub
        .sort_values(
            by="Shapiro_pvalue",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # Best rep = largest Shapiro p-value
    # --------------------------------------------------------

    best_row = (
        sub.iloc[0]
    )


    best_rep = int(
        best_row[
            "rep"
        ]
    )


    best_pvalue = float(
        best_row[
            "Shapiro_pvalue"
        ]
    )


    print(
        f"[BEST] "
        f"region={region}, "
        f"rep={best_rep}, "
        f"Shapiro p-value={best_pvalue:.8g}"
    )


    # --------------------------------------------------------
    # Print top 5 reps for checking
    # --------------------------------------------------------

    print(
        "Top 5 repetitions:"
    )


    top_n = min(
        5,
        len(sub),
    )


    for rank in range(
        top_n
    ):

        row = (
            sub.iloc[
                rank
            ]
        )

        print(
            f"    rank {rank + 1}: "
            f"rep={int(row['rep'])}, "
            f"p="
            f"{float(row['Shapiro_pvalue']):.8g}"
        )


    # --------------------------------------------------------
    # Exact QQ file corresponding to the best rep
    #
    # Example:
    #
    # if best_rep = 37:
    #
    # qq_Trans-CO_target_northeast_rep_37.png
    #
    # NOT rep_0.
    # --------------------------------------------------------

    src_path = (
        qq_dir
        / (
            "qq_Trans-CO_"
            f"target_{region}_"
            f"rep_{best_rep}.png"
        )
    )


    if not src_path.exists():

        print(
            "[WARNING] "
            f"Best QQ plot does not exist: "
            f"{src_path}"
        )

        continue


    # --------------------------------------------------------
    # Selected output filename
    # --------------------------------------------------------

    dst_path = (
        select_dir
        / (
            "selected_qq_Trans-CO_"
            f"target_{region}_"
            f"rep_{best_rep}.png"
        )
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT simply copy the PNG.
    #
    # Re-render the selected image and explicitly add
    # the larger Shapiro p-value in the bottom-right.
    # --------------------------------------------------------

    save_selected_qq_with_pvalue(
        src_path=src_path,
        dst_path=dst_path,
        shapiro_pvalue=best_pvalue,
    )


    print(
        f"[Saved] "
        f"{region}: "
        f"rep={best_rep}, "
        f"Shapiro_pvalue={best_pvalue:.8g}"
    )

    print(
        f"        -> {dst_path}"
    )


# ============================================================
# 14. Final message
# ============================================================

print("\n")
print("=" * 80)
print("ALL FIGURES SAVED")
print("=" * 80)

print("\nMain plots saved in:")
print(out_dir)

print("\nOriginal QQ plots saved in:")
print(qq_dir)

print("\nSelected best QQ plots saved in:")
print(select_dir)
