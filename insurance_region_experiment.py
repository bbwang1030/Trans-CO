#!/usr/bin/env python3
"""
Insurance region-domain experiment for Trans_CO.py.

This version uses `region` as the domain variable.

Domains:
    northeast, northwest, southeast, southwest

For each run:
    - choose one region as target domain
    - use the remaining regions as source domains
    - exclude `region` from X to avoid domain leakage
    - include `sex` and `smoker` as ordinary categorical predictors
    - use domain-wise centering without explicit intercept
    - use raw charges as the response; no log transform is applied

Compared algorithms:
    1. IPOD
    2. Sparse-LTS
    3. Multi-task-Lasso
    4. PTL
    5. Trans-Lasso
    6. Trans-PtLR
    7. Trans-CO

Training metrics with gamma, when applicable:
    y_hat = X beta + gamma

Training metrics with suffix "_nogamma":
    y_hat = X beta

QQ plots and Shapiro-Wilk tests use:
    residual = y - X beta - gamma
for methods with additive gamma, and
    residual = y - X beta
otherwise.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from scipy import stats


INSURANCE_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
)


# ============================================================
# Domain settings
# ============================================================

DOMAIN_COLUMN = "region"

DOMAIN_ORDER = [
    "northeast",
    "northwest",
    "southeast",
    "southwest",
]

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "insurance_region_raw_charge_mape"
)


# ============================================================
# Final result columns
# ============================================================

RESULT_COLUMNS = [
    "rep",
    "domain_column",
    "target_domain",
    "algorithm",

    # Test-set metrics
    "rmse_charge",
    "mae_charge",
    "r2_charge",
    "mape_charge",

    # Training metrics using X beta + gamma when applicable
    "RSS",
    "RSS_charge",
    "RSS_percent_y_charge",
    "RMSE_fit_charge",
    "MAE_fit_charge",
    "MAPE_fit_charge",
    "R2_train",

    # Training metrics using X beta only
    "RSS_nogamma",
    "RSS_charge_nogamma",
    "RSS_percent_y_charge_nogamma",
    "RMSE_fit_charge_nogamma",
    "MAE_fit_charge_nogamma",
    "MAPE_fit_charge_nogamma",
    "R2_train_nogamma",

    # Outlier and residual diagnostics
    "outlier_rate_train",
    "Shapiro_stat",
    "Shapiro_pvalue",
]


# ============================================================
# Model assessment
# ============================================================

def model_assessment(
    y_train,
    X_train,
    beta_hat,
    gamma_hat=None,
    eps=1e-12,
    y_mean=None,
    y_scale=None,
):
    """
    Training-set model assessment.

    Adjusted fitted value:

        y_hat_adjusted = X beta + gamma

    No-gamma fitted value:

        y_hat_nogamma = X beta

    For methods without additive incidental parameters,
    gamma is zero, so the two sets of training metrics coincide.

    QQ/Shapiro residual:

        y - X beta - gamma

    y_train is the centered and standardized raw-charge response:

        y_train =
            (charges - target_domain_mean_charges) / y_scale
    """

    beta_hat = np.asarray(
        beta_hat
    ).reshape(-1)

    y_train = np.asarray(
        y_train
    ).reshape(-1)

    # --------------------------------------------------------
    # Regression component
    # --------------------------------------------------------

    fitted = (
        X_train
        @ beta_hat
    )

    # --------------------------------------------------------
    # Gamma
    # --------------------------------------------------------

    if gamma_hat is None:

        gamma = np.zeros_like(
            y_train
        )

    else:

        gamma = np.asarray(
            gamma_hat
        ).reshape(-1)

    # ========================================================
    # 1. Metrics using X beta + gamma
    # ========================================================

    y_fit_adjusted = (
        fitted
        + gamma
    )

    adjusted_resid = (
        y_train
        - y_fit_adjusted
    )

    RSS = float(
        np.sum(
            adjusted_resid ** 2
        )
    )

    # ========================================================
    # 2. Metrics using X beta only
    # ========================================================

    nogamma_resid = (
        y_train
        - fitted
    )

    RSS_nogamma = float(
        np.sum(
            nogamma_resid ** 2
        )
    )

    # --------------------------------------------------------
    # Training R^2
    # --------------------------------------------------------

    TSS = float(
        np.sum(
            (
                y_train
                - np.mean(y_train)
            ) ** 2
        )
    )

    R2_train = float(
        1.0
        - RSS
        / (TSS + eps)
    )

    R2_train_nogamma = float(
        1.0
        - RSS_nogamma
        / (TSS + eps)
    )

    # ========================================================
    # Shapiro-Wilk test
    #
    # IMPORTANT:
    # Continue to use adjusted residual:
    #
    #     y - X beta - gamma
    # ========================================================

    n = len(
        y_train
    )

    if n <= 5000:

        shapiro_result = stats.shapiro(
            adjusted_resid
        )

        Shapiro_stat = float(
            shapiro_result.statistic
        )

        Shapiro_pvalue = float(
            shapiro_result.pvalue
        )

    else:

        Shapiro_stat = np.nan
        Shapiro_pvalue = np.nan

    # ========================================================
    # Original-charge-scale training metrics
    # ========================================================

    RSS_charge = np.nan
    RSS_percent_y_charge = np.nan
    RMSE_fit_charge = np.nan
    MAE_fit_charge = np.nan
    MAPE_fit_charge = np.nan

    RSS_charge_nogamma = np.nan
    RSS_percent_y_charge_nogamma = np.nan
    RMSE_fit_charge_nogamma = np.nan
    MAE_fit_charge_nogamma = np.nan
    MAPE_fit_charge_nogamma = np.nan

    if (
        y_mean is not None
        and y_scale is not None
    ):

        # ----------------------------------------------------
        # True response on original charges scale
        # ----------------------------------------------------

        y_true_charge = (
            y_train
            * y_scale
            + y_mean
        )

        # ====================================================
        # A. Original training metrics:
        #    X beta + gamma
        # ====================================================

        y_fit_charge = (
            y_fit_adjusted
            * y_scale
            + y_mean
        )

        # Avoid negative medical-charge fitted values.
        y_fit_charge = np.maximum(
            y_fit_charge,
            0.0,
        )

        resid_charge = (
            y_true_charge
            - y_fit_charge
        )

        RSS_charge = float(
            np.sum(
                resid_charge ** 2
            )
        )

        RSS_percent_y_charge = float(
            RSS_charge
            / (
                np.sum(
                    y_true_charge ** 2
                )
                + eps
            )
            * 100.0
        )

        RMSE_fit_charge = float(
            np.sqrt(
                RSS_charge
                / n
            )
        )

        MAE_fit_charge = float(
            np.mean(
                np.abs(
                    resid_charge
                )
            )
        )

        MAPE_fit_charge = float(
            np.mean(
                np.abs(
                    resid_charge
                )
                / np.maximum(
                    np.abs(
                        y_true_charge
                    ),
                    eps,
                )
            )
            * 100.0
        )

        # ====================================================
        # B. No-gamma training metrics:
        #    X beta only
        # ====================================================

        y_fit_charge_nogamma = (
            fitted
            * y_scale
            + y_mean
        )

        # Apply the same nonnegative-charge rule.
        y_fit_charge_nogamma = np.maximum(
            y_fit_charge_nogamma,
            0.0,
        )

        resid_charge_nogamma = (
            y_true_charge
            - y_fit_charge_nogamma
        )

        RSS_charge_nogamma = float(
            np.sum(
                resid_charge_nogamma ** 2
            )
        )

        RSS_percent_y_charge_nogamma = float(
            RSS_charge_nogamma
            / (
                np.sum(
                    y_true_charge ** 2
                )
                + eps
            )
            * 100.0
        )

        RMSE_fit_charge_nogamma = float(
            np.sqrt(
                RSS_charge_nogamma
                / n
            )
        )

        MAE_fit_charge_nogamma = float(
            np.mean(
                np.abs(
                    resid_charge_nogamma
                )
            )
        )

        MAPE_fit_charge_nogamma = float(
            np.mean(
                np.abs(
                    resid_charge_nogamma
                )
                / np.maximum(
                    np.abs(
                        y_true_charge
                    ),
                    eps,
                )
            )
            * 100.0
        )

    return {
        # With gamma when applicable
        "RSS": RSS,
        "RSS_charge": RSS_charge,
        "RSS_percent_y_charge":
            RSS_percent_y_charge,
        "RMSE_fit_charge":
            RMSE_fit_charge,
        "MAE_fit_charge":
            MAE_fit_charge,
        "MAPE_fit_charge":
            MAPE_fit_charge,
        "R2_train":
            R2_train,

        # X beta only
        "RSS_nogamma":
            RSS_nogamma,
        "RSS_charge_nogamma":
            RSS_charge_nogamma,
        "RSS_percent_y_charge_nogamma":
            RSS_percent_y_charge_nogamma,
        "RMSE_fit_charge_nogamma":
            RMSE_fit_charge_nogamma,
        "MAE_fit_charge_nogamma":
            MAE_fit_charge_nogamma,
        "MAPE_fit_charge_nogamma":
            MAPE_fit_charge_nogamma,
        "R2_train_nogamma":
            R2_train_nogamma,

        # Residual diagnostics
        "Shapiro_stat":
            Shapiro_stat,
        "Shapiro_pvalue":
            Shapiro_pvalue,
    }


# ============================================================
# QQ plot
# ============================================================

def plot_qq_plot(
    y_train,
    X_train,
    beta_hat,
    gamma_hat=None,
    method_name="Trans-CO",
    target_domain="northeast",
    rep=0,
    output_dir="diagnostics",
):
    """
    QQ plot for adjusted residuals.

    For methods with additive incidental parameters:

        residual = y - X beta - gamma

    For methods without additive incidental parameters:

        residual = y - X beta

    The residuals are standardized before plotting.
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    beta_hat = np.asarray(
        beta_hat
    ).reshape(-1)

    y_train = np.asarray(
        y_train
    ).reshape(-1)

    fitted = (
        X_train
        @ beta_hat
    )

    if gamma_hat is None:

        gamma = np.zeros_like(
            y_train
        )

    else:

        gamma = np.asarray(
            gamma_hat
        ).reshape(-1)

    # --------------------------------------------------------
    # QQ residual definition remains unchanged:
    #
    #     y - X beta - gamma
    # --------------------------------------------------------

    residuals = (
        y_train
        - fitted
        - gamma
    )

    residuals_std = (
        residuals
        - np.mean(
            residuals
        )
    ) / (
        np.std(
            residuals
        )
        + 1e-12
    )

    safe_method = (
        str(method_name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

    safe_target = (
        str(target_domain)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

    filename_base = (
        output_dir
        / (
            f"qq_{safe_method}"
            f"_target_{safe_target}"
            f"_rep_{rep}"
        )
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    stats.probplot(
        residuals_std,
        dist="norm",
        plot=ax,
    )

    ax.set_title("")

    ax.set_xlabel(
        "Theoretical quantiles",
        fontsize = 28,
    )

    ax.set_ylabel(
        "Sample quantiles",
        fontsize=28,
    )

    # Tick-label font size
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=25,
    )

    fig.tight_layout()

    fig.savefig(
        filename_base.with_suffix(
            ".png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Load Trans_CO.py
# ============================================================

def load_trans_ipod_namespace(
    trans_ipod_path: str,
) -> Dict[str, object]:
    """
    Load functions from Trans_CO.py without running
    its bottom simulation loop.
    """

    path = Path(
        trans_ipod_path
    )

    code = path.read_text(
        encoding="utf-8"
    )

    markers = [
        "\nB=50",
        "\r\nB=50",
    ]

    cut_pos = None

    for marker in markers:

        pos = code.find(
            marker
        )

        if pos != -1:

            cut_pos = pos
            break

    if cut_pos is None:

        raise ValueError(
            "Could not find the top-level simulation marker "
            "`B=50` in Trans_CO.py. "
            "Please remove/comment the bottom simulation loop "
            "or add the marker."
        )

    prefix = code[
        :cut_pos
    ]

    ns: Dict[str, object] = {
        "__name__":
            "trans_ipod_loaded"
    }

    exec(
        compile(
            prefix,
            str(path),
            "exec",
        ),
        ns,
    )

    required = [
        "IPOD",
        "IPODTL",
        "PTL_estimator",
        "multitask_lasso_estimator",
        "run_r_baselines_temp",
    ]

    missing = [
        name
        for name in required
        if name not in ns
    ]

    if missing:

        raise ValueError(
            "Missing required functions "
            f"from Trans_CO.py: "
            f"{missing}"
        )

    return ns


# ============================================================
# Rscript
# ============================================================

def find_rscript() -> Optional[str]:
    """
    Find Rscript on PATH or at a common
    Windows installation path.
    """

    rscript = (
        shutil.which(
            "Rscript"
        )
        or shutil.which(
            "Rscript.exe"
        )
    )

    if rscript:

        return rscript

    windows_rscript = Path(
        r"C:\Program Files\R\R-4.4.2\bin\Rscript.exe"
    )

    if windows_rscript.exists():

        return str(
            windows_rscript
        )

    return None


def patch_r_runner(
    ns: Dict[str, object],
    r_baselines_dir: Path,
) -> None:
    """
    Patch the Windows-specific R runner to
    a portable Rscript runner.
    """

    def run_r_script_portable(
        script_path,
        data_dir,
        out_dir,
        seed,
    ):

        script_path = Path(
            script_path
        )

        if not script_path.is_absolute():

            script_path = (
                r_baselines_dir.parent
                / script_path
            )

        rscript = find_rscript()

        if rscript is None:

            raise FileNotFoundError(
                "Rscript was not found. "
                "Install R, add Rscript to PATH, "
                "or run with --skip_r."
            )

        out_dir = Path(
            out_dir
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cmd = [
            rscript,
            str(script_path),
            str(data_dir),
            str(out_dir),
            str(seed),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:

            raise RuntimeError(
                "R baseline failed.\n"
                f"Command: "
                f"{' '.join(cmd)}\n"
                f"STDOUT:\n"
                f"{result.stdout[-4000:]}\n"
                f"STDERR:\n"
                f"{result.stderr[-4000:]}"
            )

    ns["run_r_script"] = (
        run_r_script_portable
    )


# ============================================================
# Load insurance dataset
# ============================================================

def load_insurance(
    data_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the Medical Cost Personal Dataset.
    """

    if data_csv:

        df = pd.read_csv(
            data_csv
        )

    else:

        df = pd.read_csv(
            INSURANCE_CSV_URL
        )

    expected = {
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region",
        "charges",
    }

    missing = (
        expected
        - set(
            df.columns
        )
    )

    if missing:

        raise ValueError(
            "Insurance CSV is missing columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    # --------------------------------------------------------
    # Basic cleaning
    # --------------------------------------------------------

    df = (
        df
        .dropna(
            subset=list(
                expected
            )
        )
        .reset_index(
            drop=True
        )
    )

    df = df[
        (df["age"] > 0)
        & (df["bmi"] > 0)
        & (df["charges"] >= 0)
    ].reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Region = domain variable
    # --------------------------------------------------------

    df[
        DOMAIN_COLUMN
    ] = pd.Categorical(
        df[
            DOMAIN_COLUMN
        ]
        .astype(str)
        .str.lower(),
        categories=DOMAIN_ORDER,
        ordered=True,
    )

    df = (
        df
        .dropna(
            subset=[
                DOMAIN_COLUMN
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# Feature construction
# ============================================================

def make_feature_frame(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the feature matrix using fixed coding.

    region is excluded from X.

    Numeric:
        age
        bmi
        children

    Categorical:
        sex_male
        smoker_yes
    """

    X = pd.DataFrame(
        index=df.index
    )

    X["age"] = (
        df["age"]
        .astype(float)
    )

    X["bmi"] = (
        df["bmi"]
        .astype(float)
    )

    X["children"] = (
        df["children"]
        .astype(float)
    )

    sex = (
        df["sex"]
        .astype(str)
        .str.lower()
    )

    smoker = (
        df["smoker"]
        .astype(str)
        .str.lower()
    )

    X["sex_male"] = (
        sex == "male"
    ).astype(float)

    X["smoker_yes"] = (
        smoker == "yes"
    ).astype(float)

    return X


# ============================================================
# Build one replication
# ============================================================

def build_one_replication(
    df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    target_domain: str,
    n_target_train: int,
    n_target_test: int,
    n_source_per_domain: int,
    random_state: int,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    """
    Build one source/target split.

    Domain-wise centering is used without
    an explicit intercept.

    Scaling parameters are estimated only
    from training data.
    """

    rng = np.random.default_rng(
        random_state
    )

    # --------------------------------------------------------
    # Target data
    # --------------------------------------------------------

    target_idx_all = df.index[
        df[
            DOMAIN_COLUMN
        ].astype(str)
        == target_domain
    ].to_numpy()

    required_target_n = (
        n_target_train
        + n_target_test
    )

    if (
        len(target_idx_all)
        < required_target_n
    ):

        raise ValueError(
            f"Target {DOMAIN_COLUMN}="
            f"{target_domain} "
            f"has only "
            f"{len(target_idx_all)} rows, "
            f"but "
            f"n_target_train+n_target_test="
            f"{required_target_n}."
        )

    target_sample = rng.choice(
        target_idx_all,
        size=required_target_n,
        replace=False,
    )

    target_train_idx = (
        target_sample[
            :n_target_train
        ]
    )

    target_test_idx = (
        target_sample[
            n_target_train:
        ]
    )

    # --------------------------------------------------------
    # Source data
    # --------------------------------------------------------

    source_indices_by_domain: Dict[
        str,
        np.ndarray,
    ] = {}

    for domain in DOMAIN_ORDER:

        if domain == target_domain:

            continue

        idx_all = df.index[
            df[
                DOMAIN_COLUMN
            ].astype(str)
            == domain
        ].to_numpy()

        n_take = min(
            n_source_per_domain,
            len(idx_all),
        )

        source_indices_by_domain[
            domain
        ] = rng.choice(
            idx_all,
            size=n_take,
            replace=False,
        )

    # ========================================================
    # Domain-wise centering
    # ========================================================

    domain_train_indices = {
        target_domain:
            target_train_idx
    }

    for (
        domain,
        idx,
    ) in source_indices_by_domain.items():

        domain_train_indices[
            domain
        ] = idx

    x_mean_by_domain: Dict[
        str,
        np.ndarray,
    ] = {}

    y_mean_by_domain: Dict[
        str,
        float,
    ] = {}

    x_centered_all = []
    y_centered_all = []

    for (
        domain,
        idx,
    ) in domain_train_indices.items():

        X_raw_d = (
            feature_frame
            .loc[
                idx
            ]
            .to_numpy(
                dtype=float
            )
        )

        y_value_d = (
            df
            .loc[
                idx,
                "charges",
            ]
            .to_numpy(
                dtype=float
            )
        )

        x_mean_d = (
            X_raw_d.mean(
                axis=0
            )
        )

        y_mean_d = float(
            y_value_d.mean()
        )

        x_mean_by_domain[
            domain
        ] = x_mean_d

        y_mean_by_domain[
            domain
        ] = y_mean_d

        x_centered_all.append(
            X_raw_d
            - x_mean_d
        )

        y_centered_all.append(
            y_value_d
            - y_mean_d
        )

    # --------------------------------------------------------
    # Common scaling
    # --------------------------------------------------------

    x_centered_all = np.vstack(
        x_centered_all
    )

    y_centered_all = np.concatenate(
        y_centered_all
    )

    x_scale = x_centered_all.std(
        axis=0,
        ddof=0,
    )

    x_scale[
        x_scale < 1e-12
    ] = 1.0

    y_scale = float(
        y_centered_all.std(
            ddof=0
        )
    )

    if y_scale < 1e-12:

        y_scale = 1.0

    # --------------------------------------------------------
    # Transform functions
    # --------------------------------------------------------

    def transform_x(
        indices: np.ndarray,
        domain: str,
    ) -> np.ndarray:

        X_raw = (
            feature_frame
            .loc[
                indices
            ]
            .to_numpy(
                dtype=float
            )
        )

        return (
            X_raw
            - x_mean_by_domain[
                domain
            ]
        ) / x_scale

    def transform_y(
        indices: np.ndarray,
        domain: str,
    ) -> np.ndarray:

        y_value = (
            df
            .loc[
                indices,
                "charges",
            ]
            .to_numpy(
                dtype=float
            )
        )

        return (
            y_value
            - y_mean_by_domain[
                domain
            ]
        ) / y_scale

    # --------------------------------------------------------
    # Target training/test
    # --------------------------------------------------------

    X_target_train = transform_x(
        target_train_idx,
        target_domain,
    )

    Y_target_train = transform_y(
        target_train_idx,
        target_domain,
    )

    X_target_test = transform_x(
        target_test_idx,
        target_domain,
    )

    Y_target_test = transform_y(
        target_test_idx,
        target_domain,
    )

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    X_source = []
    Y_source = []

    for domain in DOMAIN_ORDER:

        if domain == target_domain:

            continue

        idx = (
            source_indices_by_domain[
                domain
            ]
        )

        X_source.append(
            transform_x(
                idx,
                domain,
            )
        )

        Y_source.append(
            transform_y(
                idx,
                domain,
            )
        )

    target_y_mean = float(
        y_mean_by_domain[
            target_domain
        ]
    )

    return (
        X_source,
        Y_source,
        X_target_train,
        Y_target_train,
        X_target_test,
        Y_target_test,
        target_y_mean,
        y_scale,
    )


# ============================================================
# Test-set evaluation
# ============================================================

def evaluate_beta(
    beta: Optional[np.ndarray],
    X_test: np.ndarray,
    y_test_std: np.ndarray,
    y_mean: float,
    y_scale: float,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Test-set prediction metrics.

    Test predictions always use:

        y_hat = X beta

    No gamma is used for test observations.
    """

    if beta is None:

        return {
            "rmse_charge":
                np.nan,
            "mae_charge":
                np.nan,
            "r2_charge":
                np.nan,
            "mape_charge":
                np.nan,
        }

    beta = np.asarray(
        beta
    ).reshape(-1)

    if (
        beta.shape[0]
        != X_test.shape[1]
    ):

        raise ValueError(
            f"beta has length "
            f"{beta.shape[0]}, "
            f"but X_test has "
            f"p={X_test.shape[1]}."
        )

    # --------------------------------------------------------
    # Prediction on standardized scale
    # --------------------------------------------------------

    y_pred_std = (
        X_test
        @ beta
    )

    # --------------------------------------------------------
    # Back to original charges scale
    # --------------------------------------------------------

    y_pred_charge = (
        y_pred_std
        * y_scale
        + y_mean
    )

    y_true_charge = (
        y_test_std
        * y_scale
        + y_mean
    )

    # Avoid negative charge predictions.
    # y_pred_charge = np.maximum(
    #     y_pred_charge,
    #     0.0,
    # )

    # --------------------------------------------------------
    # Test metrics
    # --------------------------------------------------------

    rmse_charge = float(
        np.sqrt(
            mean_squared_error(
                y_true_charge,
                y_pred_charge,
            )
        )
    )

    mae_charge = float(
        mean_absolute_error(
            y_true_charge,
            y_pred_charge,
        )
    )

    r2_charge = float(
        r2_score(
            y_true_charge,
            y_pred_charge,
        )
    )

    denominator = np.maximum(
        np.abs(
            y_true_charge
        ),
        eps,
    )

    mape_charge = float(
        np.mean(
            np.abs(
                y_true_charge
                - y_pred_charge
            )
            / denominator
        )
        * 100.0
    )

    return {
        "rmse_charge":
            rmse_charge,
        "mae_charge":
            mae_charge,
        "r2_charge":
            r2_charge,
        "mape_charge":
            mape_charge,
    }


# ============================================================
# Empty metrics
# ============================================================

def empty_test_metrics():
    return {
        "rmse_charge": np.nan,
        "mae_charge": np.nan,
        "r2_charge": np.nan,
        "mape_charge": np.nan,
    }


def empty_training_metrics():
    return {
        "RSS": np.nan,
        "RSS_charge": np.nan,
        "RSS_percent_y_charge": np.nan,
        "RMSE_fit_charge": np.nan,
        "MAE_fit_charge": np.nan,
        "MAPE_fit_charge": np.nan,
        "R2_train": np.nan,

        "RSS_nogamma": np.nan,
        "RSS_charge_nogamma": np.nan,
        "RSS_percent_y_charge_nogamma": np.nan,
        "RMSE_fit_charge_nogamma": np.nan,
        "MAE_fit_charge_nogamma": np.nan,
        "MAPE_fit_charge_nogamma": np.nan,
        "R2_train_nogamma": np.nan,

        "Shapiro_stat": np.nan,
        "Shapiro_pvalue": np.nan,
    }


# ============================================================
# Run algorithm safely
# ============================================================

def safe_run_algorithm(
    algorithm_name,
    fn,
):
    """
    Run one algorithm safely.
    """

    try:

        return fn()

    except Exception as exc:

        print(
            f"{algorithm_name} failed: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return None


# ============================================================
# One target domain / one replication
# ============================================================

def run_one_setting(
    ns: Dict[str, object],
    df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    target_domain: str,
    rep: int,
    seed,
    args,
    r_available: bool,
) -> List[Dict[str, object]]:
    """
    Run all seven algorithms for one target
    domain and one replication.
    """

    (
        X_source,
        Y_source,
        X_target,
        Y_target,
        X_test,
        Y_test,
        y_mean,
        y_scale,
    ) = build_one_replication(
        df=df,
        feature_frame=feature_frame,
        target_domain=target_domain,
        n_target_train=args.n_target_train,
        n_target_test=args.n_target_test,
        n_source_per_domain=args.n_source_per_domain,
        random_state=seed,
    )

    results: List[
        Dict[str, object]
    ] = []

    # ========================================================
    # Store one algorithm
    # ========================================================

    def add_result(
        algorithm: str,
        beta,
        gamma=None,
    ):

        # ----------------------------------------------------
        # Gamma / outlier information
        # ----------------------------------------------------

        if gamma is None:

            gamma_arr = None

        else:

            gamma_arr = np.asarray(
                gamma
            ).reshape(-1)

        if gamma_arr is None:

            outlier_rate_train = np.nan

        else:

            outlier_mask = (
                np.abs(
                    gamma_arr
                )
                > 1e-8
            )

            outlier_rate_train = float(
                np.mean(
                    outlier_mask
                )
            )

        # ----------------------------------------------------
        # Sparse-LTS returns outlier indicators rather than
        # additive mean-shift gamma.
        # ----------------------------------------------------

        if algorithm == "Sparse-LTS":

            gamma_for_residual = None

        else:

            gamma_for_residual = (
                gamma_arr
            )

        # ====================================================
        # Test-set metrics
        # ====================================================

        if beta is None:

            test_metrics = (
                empty_test_metrics()
            )

        else:

            try:

                test_metrics = evaluate_beta(
                    beta=beta,
                    X_test=X_test,
                    y_test_std=Y_test,
                    y_mean=y_mean,
                    y_scale=y_scale,
                )

            except Exception as exc:

                print(
                    f"Test assessment failed for "
                    f"{algorithm}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                test_metrics = (
                    empty_test_metrics()
                )

        # ====================================================
        # Training metrics
        # ====================================================

        if beta is None:

            training_metrics = (
                empty_training_metrics()
            )

        else:

            try:

                training_metrics = model_assessment(
                    y_train=Y_target,
                    X_train=X_target,
                    beta_hat=beta,
                    gamma_hat=gamma_for_residual,
                    y_mean=y_mean,
                    y_scale=y_scale,
                )

            except Exception as exc:

                print(
                    f"Training assessment failed for "
                    f"{algorithm}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                training_metrics = (
                    empty_training_metrics()
                )

        # ====================================================
        # Final row
        # ====================================================

        row = {
            "rep":
                rep,
            "domain_column":
                DOMAIN_COLUMN,
            "target_domain":
                target_domain,
            "algorithm":
                algorithm,

            **test_metrics,
            **training_metrics,

            "outlier_rate_train":
                outlier_rate_train,
        }

        # Guarantee fixed column structure.
        row = {
            col:
                row.get(
                    col,
                    np.nan,
                )
            for col in RESULT_COLUMNS
        }

        results.append(
            row
        )

        # ====================================================
        # QQ plot
        #
        # Residual remains:
        #
        #     y - X beta - gamma
        # ====================================================

        qq_targets = [
            x.strip()
            for x
            in args.qq_target_domains.split(",")
            if x.strip()
        ]

        if (
            beta is not None
            and target_domain
            in qq_targets
        ):

            try:

                plot_qq_plot(
                    y_train=Y_target,
                    X_train=X_target,
                    beta_hat=beta,
                    gamma_hat=gamma_for_residual,
                    method_name=algorithm,
                    target_domain=target_domain,
                    rep=rep,
                    output_dir=(
                        Path(
                            args.out_dir
                        )
                        / "qq_plots"
                    ),
                )

            except Exception as exc:

                print(
                    f"QQ plot failed for "
                    f"{algorithm}, "
                    f"target={target_domain}, "
                    f"rep={rep}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

    # ========================================================
    # 1. PTL
    # ========================================================

    out = safe_run_algorithm(
        "PTL",
        lambda:
            ns[
                "PTL_estimator"
            ](
                X_source,
                Y_source,
                X_target,
                Y_target,
            ),
    )

    if out is None:

        add_result(
            "PTL",
            None,
        )

    else:

        beta_ptl, _ = out

        add_result(
            "PTL",
            beta_ptl,
        )

    # ========================================================
    # 2. IPOD
    # ========================================================

    out = safe_run_algorithm(
        "IPOD",
        lambda:
            ns[
                "IPOD"
            ](
                X_target,
                Y_target,
                (
                    X_target
                    @ np.linalg.pinv(
                        X_target.T
                        @ X_target
                    )
                    @ X_target.T
                ),
                TOL=args.tol,
                eta=args.ipod_eta,
            ),
    )

    if out is None:

        add_result(
            "IPOD",
            None,
        )

    else:

        add_result(
            "IPOD",
            out.get(
                "beta"
            ),
            gamma=out.get(
                "gamma"
            ),
        )

    # ========================================================
    # 3. Trans-CO
    # ========================================================

    ns[
        "Y_target"
    ] = Y_target

    ns[
        "X_target"
    ] = X_target

    out = safe_run_algorithm(
        "Trans-CO",
        lambda:
            ns[
                "IPODTL"
            ](
                X_source,
                Y_source,
                X_target,
                Y_target,
                method="hard",
                TOL=args.tol,
                eta=args.transco_eta,
            ),
    )

    if out is None:

        add_result(
            "Trans-CO",
            None,
        )

    else:

        add_result(
            "Trans-CO",
            out.get(
                "beta"
            ),
            gamma=out.get(
                "gamma"
            ),
        )

    # ========================================================
    # 4. Multi-task-Lasso
    # ========================================================

    out = safe_run_algorithm(
        "Multi-task-Lasso",
        lambda:
            ns[
                "multitask_lasso_estimator"
            ](
                X_source,
                Y_source,
                X_target,
                Y_target,
                random_state=seed,
                max_iter=args.mtl_max_iter,
                tol=args.tol,
            ),
    )

    if out is None:

        add_result(
            "Multi-task-Lasso",
            None,
        )

    else:

        beta_mtl, _, _ = out

        add_result(
            "Multi-task-Lasso",
            beta_mtl,
        )

    # ========================================================
    # 5-7. R-side baselines
    # ========================================================

    if (
        r_available
        and not args.skip_r
    ):

        out = safe_run_algorithm(
            "R baselines",
            lambda:
                ns[
                    "run_r_baselines_temp"
                ](
                    X_source,
                    Y_source,
                    X_target,
                    Y_target,
                    seed,
                ),
        )

        if out is None:

            add_result(
                "Trans-Lasso",
                None,
            )

            add_result(
                "Sparse-LTS",
                None,
            )

            add_result(
                "Trans-PtLR",
                None,
            )

        else:

            # ------------------------------------------------
            # Trans-Lasso
            # ------------------------------------------------

            add_result(
                "Trans-Lasso",
                out.get(
                    "beta_hat_translasso"
                ),
            )

            # ------------------------------------------------
            # Sparse-LTS
            # ------------------------------------------------

            add_result(
                "Sparse-LTS",
                out.get(
                    "beta_hat_sparse_lts"
                ),
                gamma=out.get(
                    "outlier_hat_sparse_lts"
                ),
            )

            # ------------------------------------------------
            # Trans-PtLR
            # ------------------------------------------------

            add_result(
                "Trans-PtLR",
                out.get(
                    "beta_hat_transptlr"
                ),
            )

    else:

        if args.skip_r:

            print(
                "R baselines skipped "
                "because --skip_r was set."
            )

        else:

            print(
                "R baselines unavailable: "
                "Rscript or required R scripts "
                "were not found."
            )

        add_result(
            "Trans-Lasso",
            None,
        )

        add_result(
            "Sparse-LTS",
            None,
        )

        add_result(
            "Trans-PtLR",
            None,
        )

    return results


# ============================================================
# Summary
# ============================================================

def summarize_results(
    detail_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate mean and SD across replications.

    Shapiro p-values are summarized by their median,
    while Shapiro statistics are summarized by mean and SD.
    """

    if detail_df.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Metrics summarized by mean and SD
    # --------------------------------------------------------

    mean_sd_metrics = [
        # Test
        "rmse_charge",
        "mae_charge",
        "r2_charge",
        "mape_charge",

        # Training with gamma
        "RSS",
        "RSS_charge",
        "RSS_percent_y_charge",
        "RMSE_fit_charge",
        "MAE_fit_charge",
        "MAPE_fit_charge",
        "R2_train",

        # Training without gamma
        "RSS_nogamma",
        "RSS_charge_nogamma",
        "RSS_percent_y_charge_nogamma",
        "RMSE_fit_charge_nogamma",
        "MAE_fit_charge_nogamma",
        "MAPE_fit_charge_nogamma",
        "R2_train_nogamma",

        # Outlier / residual
        "outlier_rate_train",
        "Shapiro_stat",
    ]

    agg_dict = {}

    for col in mean_sd_metrics:

        if col in detail_df.columns:

            agg_dict[
                f"mean_{col}"
            ] = (
                col,
                "mean",
            )

            agg_dict[
                f"sd_{col}"
            ] = (
                col,
                "std",
            )

    # --------------------------------------------------------
    # Shapiro p-value
    # --------------------------------------------------------

    if (
        "Shapiro_pvalue"
        in detail_df.columns
    ):

        agg_dict[
            "median_Shapiro_pvalue"
        ] = (
            "Shapiro_pvalue",
            "median",
        )

    # ========================================================
    # Per-target-domain summary
    # ========================================================

    summary = (
        detail_df
        .groupby(
            [
                "target_domain",
                "algorithm",
            ],
            as_index=False,
        )
        .agg(
            **agg_dict
        )
    )

    # ========================================================
    # Overall summary
    # ========================================================

    overall = (
        detail_df
        .groupby(
            [
                "algorithm"
            ],
            as_index=False,
        )
        .agg(
            **agg_dict
        )
    )

    overall.insert(
        0,
        "target_domain",
        "ALL",
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    if (
        "mean_rmse_charge"
        in summary.columns
    ):

        summary = (
            summary
            .sort_values(
                [
                    "target_domain",
                    "mean_rmse_charge",
                    "algorithm",
                ]
            )
        )

        overall = (
            overall
            .sort_values(
                [
                    "mean_rmse_charge",
                    "algorithm",
                ]
            )
        )

    return pd.concat(
        [
            overall,
            summary,
        ],
        ignore_index=True,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trans_ipod",
        type=str,
        default=str(
            BASE_DIR
            / "Trans_CO.py"
        ),
    )

    parser.add_argument(
        "--data_csv",
        type=str,
        default="insurance.csv",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(
            DEFAULT_RESULTS_DIR
        ),
    )

    parser.add_argument(
        "--B",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--n_target_train",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--n_target_test",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--n_source_per_domain",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--target_domains",
        type=str,
        default=",".join(
            DOMAIN_ORDER
        ),
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--ipod_eta",
        type=float,
        default=1 / 20,
    )

    parser.add_argument(
        "--transco_eta",
        type=float,
        default=1 / 20,
    )

    parser.add_argument(
        "--mtl_max_iter",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--skip_r",
        action="store_true",
    )

    parser.add_argument(
        "--qq_target_domains",
        type=str,
        default=(
            "northeast,"
            "northwest,"
            "southeast,"
            "southwest"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Output directory
    # ========================================================

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load Trans_CO.py
    # ========================================================

    ns = load_trans_ipod_namespace(
        args.trans_ipod
    )

    # ========================================================
    # R baselines
    # ========================================================

    r_baselines_dir = (
        Path(
            args.trans_ipod
        )
        .resolve()
        .parent
        / "R_baselines"
    )

    r_scripts = [
        r_baselines_dir
        / "run_translasso.R",

        r_baselines_dir
        / "run_sparse_lts.R",

        r_baselines_dir
        / "run_transptlr.R",
    ]

    r_available = (
        all(
            path.exists()
            for path in r_scripts
        )
        and find_rscript()
        is not None
    )

    patch_r_runner(
        ns,
        r_baselines_dir,
    )

    # ========================================================
    # Data
    # ========================================================

    df = load_insurance(
        args.data_csv
    )

    feature_frame = make_feature_frame(
        df
    )

    # ========================================================
    # Target domains
    # ========================================================

    target_domains = [
        x.strip()
        for x
        in args.target_domains.split(",")
        if x.strip()
    ]

    unknown = [
        x
        for x in target_domains
        if x not in DOMAIN_ORDER
    ]

    if unknown:

        raise ValueError(
            f"Unknown region domains: "
            f"{unknown}. "
            f"Valid domains: "
            f"{DOMAIN_ORDER}"
        )

    # ========================================================
    # Information
    # ========================================================

    print(
        "Dataset rows after cleaning:",
        len(df),
    )

    print(
        "Domain column:",
        DOMAIN_COLUMN,
    )

    print(
        "\nDomain counts:"
    )

    print(
        df[
            DOMAIN_COLUMN
        ]
        .value_counts()
        .reindex(
            DOMAIN_ORDER
        )
    )

    print(
        "\nFeature dimension p:",
        feature_frame.shape[1],
    )

    print(
        "R baselines available:",
        r_available,
    )

    print(
        "Target domains:",
        target_domains,
    )

    # ========================================================
    # Repeated experiment
    # ========================================================

    all_rows: List[
        Dict[str, object]
    ] = []

    BASE_SEED = args.seed

    algorithm_order = [
        "IPOD",
        "Sparse-LTS",
        "Multi-task-Lasso",
        "PTL",
        "Trans-Lasso",
        "Trans-PtLR",
        "Trans-CO",
    ]

    for rep in range(
        args.B
    ):

        for (
            num,
            target_domain,
        ) in enumerate(
            target_domains
        ):

            print(
                "\n"
                f"Running rep={rep}, "
                f"target_{DOMAIN_COLUMN}="
                f"{target_domain}"
            )

            rep_seed = (
                BASE_SEED
                + num
                * args.B
                + rep
            )

            try:

                rows = run_one_setting(
                    ns=ns,
                    df=df,
                    feature_frame=feature_frame,
                    target_domain=target_domain,
                    rep=rep,
                    seed=rep_seed,
                    args=args,
                    r_available=r_available,
                )

                all_rows.extend(
                    rows
                )

                print_cols = [
                    "algorithm",

                    "rmse_charge",
                    "mae_charge",
                    "r2_charge",
                    "mape_charge",

                    "MAPE_fit_charge",
                    "R2_train",

                    "MAPE_fit_charge_nogamma",
                    "R2_train_nogamma",

                    "outlier_rate_train",
                    "Shapiro_pvalue",
                ]

                print(
                    pd.DataFrame(
                        rows
                    )[
                        print_cols
                    ]
                )

            except Exception as exc:

                print(
                    "Setting failed: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                for alg in algorithm_order:

                    failed_row = {
                        col:
                            np.nan
                        for col
                        in RESULT_COLUMNS
                    }

                    failed_row.update(
                        {
                            "rep":
                                rep,
                            "domain_column":
                                DOMAIN_COLUMN,
                            "target_domain":
                                target_domain,
                            "algorithm":
                                alg,
                        }
                    )

                    all_rows.append(
                        failed_row
                    )

    # ========================================================
    # Detail results
    # ========================================================

    detail_df = pd.DataFrame(
        all_rows,
        columns=RESULT_COLUMNS,
    )

    detail_df = (
        detail_df
        .sort_values(
            [
                "rep",
                "target_domain",
                "algorithm",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Summary results
    # ========================================================

    summary_df = summarize_results(
        detail_df
    )

    # ========================================================
    # Training/model-assessment table
    # ========================================================

    assessment_cols = [
        "rep",
        "domain_column",
        "target_domain",
        "algorithm",

        # With gamma
        "RSS",
        "RSS_charge",
        "RSS_percent_y_charge",
        "RMSE_fit_charge",
        "MAE_fit_charge",
        "MAPE_fit_charge",
        "R2_train",

        # Without gamma
        "RSS_nogamma",
        "RSS_charge_nogamma",
        "RSS_percent_y_charge_nogamma",
        "RMSE_fit_charge_nogamma",
        "MAE_fit_charge_nogamma",
        "MAPE_fit_charge_nogamma",
        "R2_train_nogamma",

        # Diagnostics
        "outlier_rate_train",
        "Shapiro_stat",
        "Shapiro_pvalue",
    ]

    assessment_df = (
        detail_df[
            assessment_cols
        ]
        .copy()
    )

    # ========================================================
    # Save
    # ========================================================

    detail_csv = (
        out_dir
        / "insurance_region_detail_results.csv"
    )

    summary_csv = (
        out_dir
        / "insurance_region_summary_results.csv"
    )

    assessment_csv = (
        out_dir
        / "insurance_region_model_assessment.csv"
    )

    excel_path = (
        out_dir
        / "insurance_region_results.xlsx"
    )

    detail_df.to_csv(
        detail_csv,
        index=False,
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    assessment_df.to_csv(
        assessment_csv,
        index=False,
    )

    with pd.ExcelWriter(
        excel_path
    ) as writer:

        detail_df.to_excel(
            writer,
            sheet_name="detail",
            index=False,
        )

        summary_df.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        assessment_df.to_excel(
            writer,
            sheet_name="model_assessment",
            index=False,
        )

    # ========================================================
    # Finished
    # ========================================================

    print(
        "\nSaved:"
    )

    print(
        " ",
        detail_csv,
    )

    print(
        " ",
        summary_csv,
    )

    print(
        " ",
        assessment_csv,
    )

    print(
        " ",
        excel_path,
    )

    print(
        "\nRetained detail columns:"
    )

    for col in RESULT_COLUMNS:

        print(
            " ",
            col,
        )

    if (
        not r_available
        and not args.skip_r
    ):

        print(
            "\nNOTE: R-side baselines "
            "were not run. "
            "To run all seven algorithms, "
            "put the following scripts in "
            "R_baselines next to Trans_CO.py:"
        )

        print(
            "  run_translasso.R"
        )

        print(
            "  run_sparse_lts.R"
        )

        print(
            "  run_transptlr.R"
        )

        print(
            "and make sure Rscript "
            "is available."
        )


if __name__ == "__main__":
    main()
