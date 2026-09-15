import numpy as np
import random
from sklearn.linear_model import LassoCV,LassoLarsCV, LinearRegression
from numpy.linalg import pinv
from numpy.linalg import inv, norm
from scipy.stats import norm as stats_norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.robust.robust_linear_model import RLM
import math
import pandas as pd

import os
import shutil
import subprocess
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
import tempfile
import gc
from Trans_CO import *

#     # 固定参数
#     # 计算中间参数
#     # 生成B矩阵
#     B[:r0, :] = 2 * UK   # 前r0行
#     B[r0:s, :] = 0.3   # 中间s-r0行
#     # B_revise[:s,:] += support*0.1
#     V[:s, :] = rng.choice([-1,1],size=(s, K))
#     # 生成delta
#     # 计算目标参数beta
#     # 生成源数据（K个源任务）
#         # 随机选择o个不同的索引位置
#         # 将这些位置的值设为outlier
#     # 生成目标数据
#     # 将这些位置的值设为outlier
#     }

def generate_data_with_outliers(K=5, s=40, n=150, N=800, o=50, support=1):
    """
    Redesigned Example 8.

    Here `support` is interpreted as the support-mismatch level r.

    For each source k:
        1. The baseline coefficient vector is supported on the first s coordinates.
        2. Randomly select r coordinates D_k(r) from the first s coordinates.
        3. Move their coefficient values one-to-one to coordinates
           {s+1, ..., s+r} in mathematical indexing.
        4. Require
               union_k S_k(r) = {1, ..., s},
           where S_k(r) is the set of retained coordinates.

    The target coefficient is
        beta(r) = B(r) w + delta(r),

    where supp(delta(r)) is sampled from
        {s+r+1, ..., p}
    in mathematical indexing.
    """

    # ============================================================
    # Fixed parameters
    # ============================================================
    p = 100
    h = 6

    # `support` in the outer code now represents r
    r = int(support)

    r0 = s // 3
    s_delta = s // 5

    if r < 0 or r > s:
        raise ValueError(f"r must satisfy 0 <= r <= s, but got r={r}.")

    # In Python indexing, delta is sampled from {s+r, ..., p-1}.
    # Make sure there are enough coordinates for s_delta nonzeros.
    if p - (s + r) < s_delta:
        raise ValueError(
            f"Not enough coordinates for delta: "
            f"p={p}, s={s}, r={r}, s_delta={s_delta}."
        )

    # Transfer weights
    w = np.random.uniform(low=-2.0, high=2.0, size=K)

    # ============================================================
    # Step 1: Generate the baseline B
    # Each column is nonzero on the first s coordinates
    # (almost surely for the SVD block) and zero afterward.
    # ============================================================
    Omega = np.random.randn(r0, K)
    U, _, _ = np.linalg.svd(Omega, full_matrices=False)
    UK = U

    B_base = np.zeros((p, K))
    B_base[:r0, :] = 2 * UK
    B_base[r0:s, :] = 0.3

    # ============================================================
    # Step 2: Construct B(r) by genuinely changing source supports
    # ============================================================
    B_revise = B_base.copy()

    all_first_s = np.arange(s)

    D_sets = []
    S_sets = []

    if r == 0:
        for k in range(K):
            D_sets.append(np.array([], dtype=int))
            S_sets.append(all_first_s.copy())

    else:
        # We need
        # union_k S_k(r) = {0, ..., s-1}.
        # Since S_k(r) = {0,...,s-1} \ D_k(r),
        # this is equivalent to
        # intersection_k D_k(r) = empty set.
        # Rejection sampling is extremely efficient for the current
        while True:
            candidate_D = [
                np.random.choice(all_first_s, size=r, replace=False)
                for _ in range(K)
            ]

            common_removed = set(candidate_D[0])
            for k in range(1, K):
                common_removed.intersection_update(candidate_D[k])

            if len(common_removed) == 0:
                D_sets = candidate_D
                break

        # Construct the retained sets S_k(r)
        for k in range(K):
            D_k = D_sets[k]
            S_k = np.setdiff1d(all_first_s, D_k)
            S_sets.append(S_k)

        # Move the r nonzero coefficient values one-to-one
        # Python coordinates:
        #   s, ..., s+r-1
        # correspond to mathematical coordinates:
        #   s+1, ..., s+r.
        destination = np.arange(s, s + r)

        for k in range(K):
            D_k = D_sets[k]

            # Save the original nonzero values before zeroing them
            moved_values = B_revise[D_k, k].copy()

            # Set original coordinates to zero
            B_revise[D_k, k] = 0.0

            # Move the values one-to-one to the new coordinates
            B_revise[destination, k] = moved_values

    # ============================================================
    # Optional internal verification of the designed supports
    # ============================================================

    # Verify that every S_k(r) contains exactly s-r indices
    for k in range(K):
        assert len(S_sets[k]) == s - r

    # Verify union_k S_k(r) = {0, ..., s-1}
    union_S = set()
    for S_k in S_sets:
        union_S.update(S_k.tolist())

    assert union_S == set(all_first_s.tolist())

    # Each source still has exactly s nonzero coefficients
    # (up to numerical zero, which occurs with probability zero
    # in the random SVD block).
    for k in range(K):
        assert np.count_nonzero(B_revise[:, k]) == s

    # Step 3: Generate delta(r)
    # Mathematical indexing:
    # Python indexing:
    delta_pool = np.arange(s + r, p)

    S_delta = np.random.choice(
        delta_pool,
        size=s_delta,
        replace=False
    )

    delta = np.zeros(p)

    delta[S_delta] = np.random.normal(
        loc=10,
        scale=np.sqrt(h / s_delta),
        size=s_delta
    )

    # ============================================================
    # Step 4: Target coefficient
    # ============================================================
    beta = B_revise @ w + delta

    # ============================================================
    # Step 5: Generate source data
    # Covariate distributions no longer depend on r.
    # Therefore r changes the support structure, rather than
    # simultaneously changing the mean of X_source.
    # ============================================================
    X_source = []
    Y_source = []

    for k in range(K):
        X_k = np.random.normal(0, 1, (N, p))
        eps_k = np.random.normal(0, 1, N)

        gamma = np.zeros(N)

        random_indices_source = np.random.choice(
            N,
            size=int(o * N),
            replace=False
        )

        for i in random_indices_source:
            mean = np.random.uniform(low=1, high=20, size=1)
            std_dev = np.random.uniform(low=0, high=5, size=1)
            gamma[i] = np.random.normal(
                mean,
                std_dev,
                size=1
            )[0]

        Y_k = X_k @ B_revise[:, k] + eps_k + gamma

        X_source.append(X_k)
        Y_source.append(Y_k)

    # ============================================================
    # Step 6: Generate target data
    # ============================================================
    X_target = np.random.normal(0, 1, (n, p))
    eps = np.random.normal(0, 1, n)

    gamma = np.zeros(n)

    random_indices = np.random.choice(
        n,
        size=int(o * n),
        replace=False
    )

    for i in random_indices:
        mean = np.random.uniform(low=1, high=20, size=1)
        std_dev = np.random.uniform(low=0, high=5, size=1)
        gamma[i] = np.random.normal(
            mean,
            std_dev,
            size=1
        )[0]

    Y_target = X_target @ beta + eps + gamma

    # ============================================================
    # Return
    # ============================================================
    return {
        "B": B_revise,
        "delta": delta,
        "beta": beta,
        "X_source": X_source,
        "Y_source": Y_source,
        "X_target": X_target,
        "Y_target": Y_target,
        "outlier_index": random_indices,

        # Extra information for checking Example 8.
        # The rest of your code does not need to use these.
        "r": r,
        "S_k": S_sets,
        "D_k": D_sets,
        "S_delta": S_delta
    }

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "ex8"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_SEED = 2026
B=50
s_=[75,25]
o=0.05
n = 200
N_=[1500]
supports = [9,6,3]
supports = [10,8,6,4,2]
K=5
for num_s,s in enumerate(s_):
    print("s",s)
    for num_N,N in enumerate(N_):
        print("N", N)
        for num_support,support in enumerate(supports):
            print("support", support)
            mse_results = []
            mse_results_ipod = []
            mse_results_Trans_CO = []
            results_list = []
            for b in range(B):
                setting_id = (num_s * len(N_) + num_N) * len(supports) + num_support
                rep_seed = BASE_SEED + setting_id * B + b
                set_random_seed(rep_seed)
                print(b, "seed =", rep_seed)
                # 生成模拟数据（假设generate_data返回X, y, beta）
                sparse_data = generate_data_with_outliers(K=K, s=s, n=n, N=N, o=o, support = support)
                X_source = sparse_data["X_source"]
                Y_source = sparse_data["Y_source"]
                X_target = sparse_data["X_target"]
                Y_target = sparse_data["Y_target"]
                true_outlier = sparse_data["outlier_index"]

                # 使用PTL估计器（假设已实现五折CV选择λ参数）
                beta_hat,wb_ptl = PTL_estimator(X_source, Y_source, X_target, Y_target)

                # 不使用迁移学习下的IPOD算法
                H = np.dot(np.dot(X_target, inv(np.dot(X_target.T, X_target))), X_target.T)
                IPOD1 = IPOD(X_target, Y_target, H, TOL=1e-4, eta=1/20)
                detect_index_IPOD = np.where(IPOD1["gamma"] != 0)[0]
                TP,FP,TN,FN,f1_score = score(Y_target, true_outlier, detect_index_IPOD)

                #使用迁移学习下的IPOD算法
                beta_hat_Trans_CO = IPODTL(X_source,Y_source, X_target, Y_target,  method="hard", TOL=1e-4, eta=1/20)
                detect_index_Trans_CO = np.where(beta_hat_Trans_CO["gamma"] != 0)[0]
                TP_, FP_, TN_, FN_, f1_score_ = score(Y_target, true_outlier, detect_index_Trans_CO)

                # ========== 新增 baseline 3: Multi-task Lasso ==========
                beta_hat_mtl, best_mtl_lambda, best_mtl_val_mse = multitask_lasso_estimator(
                    X_source,
                    Y_source,
                    X_target,
                    Y_target,
                    random_state=rep_seed,
                    tol=1e-4
                )

                # ========== R baselines: temporary CSV files, automatically deleted ==========
                r_baseline_results = run_r_baselines_temp(
                    X_source,
                    Y_source,
                    X_target,
                    Y_target,
                    seed = rep_seed
                )

                beta_hat_translasso = r_baseline_results["beta_hat_translasso"]
                beta_hat_sparse_lts = r_baseline_results["beta_hat_sparse_lts"]
                outlier_hat_sparse_lts = r_baseline_results["outlier_hat_sparse_lts"]
                beta_hat_transptlr = r_baseline_results["beta_hat_transptlr"]

                f1_sparse_lts = np.nan
                if outlier_hat_sparse_lts is not None:
                    detect_index_sparse_lts = np.where(outlier_hat_sparse_lts != 0)[0]
                    TP_slts, FP_slts, TN_slts, FN_slts, f1_sparse_lts = score(
                        Y_target,
                        true_outlier,
                        detect_index_sparse_lts
                    )


                # 计算MSE
                mse = np.log(np.mean((beta_hat - sparse_data["beta"]) ** 2))
                mse_results.append(mse)

                mse_ipod = np.log(np.mean((IPOD1["beta"] - sparse_data["beta"]) ** 2))
                mse_results_ipod.append(mse_ipod)

                mse_Trans_CO = np.log(np.mean((beta_hat_Trans_CO["beta"] - sparse_data["beta"]) ** 2))
                mse_results_Trans_CO.append(mse_Trans_CO)

                mse_mtl = np.log(np.mean((beta_hat_mtl - sparse_data["beta"]) ** 2))

                mse_translasso = np.nan
                if beta_hat_translasso is not None:
                    mse_translasso = np.log(np.mean((beta_hat_translasso - sparse_data["beta"]) ** 2))

                mse_sparse_lts = np.nan
                if beta_hat_sparse_lts is not None:
                    mse_sparse_lts = np.log(np.mean((beta_hat_sparse_lts - sparse_data["beta"]) ** 2))

                mse_transptlr = np.nan
                if beta_hat_transptlr is not None:
                    mse_transptlr = np.log(np.mean((beta_hat_transptlr - sparse_data["beta"]) ** 2))

                results = {
                    "seed": rep_seed,
                    "IPOD": mse_ipod,
                    "Sparse-LTS": mse_sparse_lts,

                    "Multi-task-Lasso": mse_mtl,

                    "PTL": mse,
                    "Trans-Lasso": mse_translasso,

                    "Trans-PtLR": mse_transptlr,
                    "Trans-CO": mse_Trans_CO,

                    "f1_score_IPOD": f1_score,
                    "f1_score_Trans-CO": f1_score_,
                    "f1_score_Sparse-LTS": f1_sparse_lts
                }

                results_list.append(results)
                print(results)
            df_results = pd.DataFrame(results_list)
            # 保存到 Excel
            with pd.ExcelWriter(RESULTS_DIR / f"out_simu_results_ex8_B{B}_s{s}_n{n}_N{N}_support{support}.xlsx") as writer:
                df_results.to_excel(writer, sheet_name="Model Results", index=False)

