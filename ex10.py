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
import numpy as np

def generate_r_orthogonal_to_B(B, seed=123):
    """
    Generate r = (I - P_B) z / ||(I - P_B) z||_2,
    where z ~ N(0, I_p) and P_B is the projection onto span(B).
    """
    rng = np.random.default_rng(seed)

    p, K = B.shape

    # Step 1: z ~ N(0, I_p)
    z = rng.normal(size=p)

    # Step 2: get orthonormal basis of span(B) by SVD
    U, svals, Vt = np.linalg.svd(B, full_matrices=False)

    # numerical rank of B
    tol = np.finfo(float).eps * max(B.shape) * (svals[0] if len(svals) > 0 else 1.0)
    rank = np.sum(svals > tol)

    # Step 3: compute (I - P_B) z
    if rank == 0:
        z_perp = z
    else:
        Q = U[:, :rank]          # columns of Q span col(B)
        z_perp = z - Q @ (Q.T @ z)

    # Step 4: normalize
    norm_z_perp = np.linalg.norm(z_perp)

    if norm_z_perp < 1e-12:
        raise ValueError("The projected vector is nearly zero. Try another random seed.")

    r = z_perp / norm_z_perp

    return r

def generate_data_with_outliers(K=5, s=40, n=150, N=800, o=50, tau = 0.25):
    # 固定参数
    p = 100
    h = 6
    w = np.random.uniform(low=-2.0, high=2.0, size=K)

    # 计算中间参数
    r0 = s // 3
    s_delta = s // 5

    # 生成B矩阵
    Omega = np.random.randn(r0, K)
    U, _, _ = np.linalg.svd(Omega, full_matrices=False)
    UK = U  # 前K左奇异向量

    B = np.zeros((p, K))
    B[:r0, :] = 2 * UK  # 前r0行
    B[r0:s, :] = 0.3  # 中间s-r0行

    # 生成delta
    S = np.random.choice(np.arange(s, p), size=s_delta, replace=False)
    delta = np.zeros(p)
    delta[S] = np.random.normal(10, np.sqrt(h / s_delta), s_delta)

    r = generate_r_orthogonal_to_B(B, seed=123)
    # 计算目标参数beta
    beta = B @ w + delta + tau * r


    # 生成源数据（K个源任务）
    X_source = []
    Y_source = []
    for k in range(K):
        X_k = np.random.normal(0, 1, (N, p))  # 协方差矩阵Σk=I
        eps_k = np.random.normal(0, 1, N)  # σk²=1
        gamma = np.zeros(N)
        # 随机选择o个不同的索引位置
        random_indices = np.random.choice(N, size=int(o*N), replace=False)
        # 将这些位置的值设为outlier
        for i in random_indices:
            mean = np.random.uniform(low=1, high=20, size=1)  # 在区间 (5, 10) 上生成一个随机数
            std_dev = np.random.uniform(low=0, high=5, size=1)
            gamma[i] = np.random.normal(mean, std_dev, size=1)[0]
        Y_k = X_k @ B[:, k] + eps_k + gamma
        X_source.append(X_k)
        Y_source.append(Y_k)

    # 生成目标数据
    X_target = np.random.normal(0, 1, (n, p))  # 协方差矩阵Σ=I
    eps = np.random.normal(0, 1, n)  # σ²=1
    gamma = np.zeros(n)
    numbers = np.array(range(s, n))
    random_indices = np.random.choice(n, size=int(o * n), replace=False)

    # 将这些位置的值设为outlier
    for i in random_indices:
        mean = np.random.uniform(low=1, high=20, size=1)  # 在区间 (5, 10) 上生成一个随机数
        std_dev = np.random.uniform(low=0, high=5, size=1)
        gamma[i] = np.random.normal(mean, std_dev, size=1)[0]
    Y_target = X_target @ beta + eps + gamma

    return {
        "B": B,
        "delta": delta,
        "beta": beta,
        "X_source": X_source,
        "Y_source": Y_source,
        "X_target": X_target,
        "Y_target": Y_target,
        "outlier_index": random_indices
    }

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "ex10"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_SEED = 2026
B=50
s_=[25,75]
o=0.05
n = 200
N_=[1500]
tau_ = [2,1,0.5,0.25]
K=5
for num_s,s in enumerate(s_):
    print("s",s)
    for num_N,N in enumerate(N_):
        print("N", N)
        for num_tau,tau in enumerate(tau_):
            print("tau", tau)
            mse_results = []
            mse_results_ipod = []
            mse_results_Trans_CO = []
            results_list = []
            for b in range(B):
                setting_id = (num_s * len(N_) + num_N) * len(tau_) + num_tau
                rep_seed = BASE_SEED + setting_id * B + b
                set_random_seed(rep_seed)
                print(b, "seed =", rep_seed)
                # 生成模拟数据（假设generate_data返回X, y, beta）
                sparse_data = generate_data_with_outliers(K=K, s=s, n=n, N=N, o=o, tau=tau)
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
                    seed=rep_seed,
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
            with pd.ExcelWriter(RESULTS_DIR / f"out_simu_results_ex10_B{B}_s{s}_tau{tau}_N{N}.xlsx") as writer:
                df_results.to_excel(writer, sheet_name="Model Results", index=False)

