import numpy as np
import random
from sklearn.linear_model import LassoCV,LassoLarsCV, LinearRegression
from numpy.linalg import pinv
from numpy.linalg import inv, norm
from scipy.stats import norm as stats_norm
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.robust.robust_linear_model import RLM
import math
import pandas as pd
from pathlib import Path
import math
from Trans_CO import *

def generate_data_with_outliers(K=5, s=40, n=150, N=800, o=50):
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

    # 计算目标参数beta
    beta = B @ w + delta


    # 生成源数据（K个源任务）
    X_source = []
    Y_source = []
    for j in range(K):
        k=j+1
        mu_k = np.zeros(p)
        first_col = np.zeros(p)
        first_col[0] = 1
        end_idx = min(2 * k, p)
        first_col[1:end_idx] = 1 / (k + 1)

        # Generate the Toeplitz matrix
        Sigma_k = toeplitz(first_col)
        X_k = np.random.multivariate_normal(mu_k, Sigma_k,N)  # 协方差矩阵Σk=I
        eps_k = np.random.normal(0,(k+1)/10, N)  # σk²=1
        gamma = np.zeros(N)
        # print(X_k.shape)
        # 随机选择o个不同的索引位置
        random_indices = np.random.choice(N, size=int(o*N), replace=False)
        # 将这些位置的值设为outlier
        for i in random_indices:
            mean = np.random.uniform(low=1, high=20, size=1)  # 在区间 (5, 10) 上生成一个随机数
            std_dev = np.random.uniform(low=0, high=5, size=1)
            gamma[i] = np.random.normal(mean, std_dev, size=1)[0]
        Y_k = X_k @ B[:, k-1] + eps_k + gamma
        X_source.append(X_k)
        Y_source.append(Y_k)

    # 生成目标数据
    X_target = np.random.normal(0, 1, (n, p))  # 协方差矩阵Σ=I
    eps = np.random.normal(0, 1, n)  # σ²=1
    gamma = np.zeros(n)
    numbers = np.array(range(1, n))
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

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "ex3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_SEED = 2026
B=50
s_=[75,25]
o=0.05
n_ = [150,200,300]
N_=[1000,1500,2000]
K=5
for num_s,s in enumerate(s_):
    print("s",s)
    for num_N,N in enumerate(N_):
        print("N", N)
        for num_n,n in enumerate(n_):
            print("n", n)
            mse_results = []
            mse_results_ipod = []
            mse_results_ipod_tr = []
            results_list = []
            for b in range(B):
                setting_id = (num_s * len(N_) + num_N) * len(n_) + num_n
                rep_seed = BASE_SEED + setting_id * B + b
                set_random_seed(rep_seed)
                print(b, "seed =", rep_seed)
                # 生成模拟数据（假设generate_data返回X, y, beta）
                # sparse_data = generate_data(s=40, n=600, N=1500)
                sparse_data = generate_data_with_outliers(K=K, s=s, n=n, N=N, o=o)
                X_source = sparse_data["X_source"]
                Y_source = sparse_data["Y_source"]
                X_target = sparse_data["X_target"]
                Y_target = sparse_data["Y_target"]
                true_outlier = sparse_data["outlier_index"]

                # 使用PTL估计器（假设已实现五折CV选择λ参数）
                beta_hat, wb_ptl = PTL_estimator(X_source, Y_source, X_target, Y_target)
                # print("wb_ptl",wb_ptl)

                # 不使用迁移学习下的IPOD算法
                H = np.dot(np.dot(X_target, inv(np.dot(X_target.T, X_target))), X_target.T)
                IPOD1 = IPOD(X_target, Y_target, H, TOL=1e-4, eta=1 / 20)
                # model = LassoLarsCV(cv=5, n_jobs=-1).fit(X_target, (Y_target-IPOD1["gamma"]))
                # beta_hat_IPOD = model.coef_
                detect_index_IPOD = np.where(IPOD1["gamma"] != 0)[0]
                TP, FP, TN, FN, f1_score = score(Y_target, true_outlier, detect_index_IPOD)

                # 使用迁移学习下的IPOD算法
                beta_hat_IPOD_tr = IPODTL(X_source, Y_source, X_target, Y_target, method="hard", TOL=1e-4, eta=1 / 20)
                detect_index_IPOD_tr = np.where(beta_hat_IPOD_tr["gamma"] != 0)[0]
                TP_, FP_, TN_, FN_, f1_score_ = score(Y_target, true_outlier, detect_index_IPOD_tr)

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

                mse_ipod_tr = np.log(np.mean((beta_hat_IPOD_tr["beta"] - sparse_data["beta"]) ** 2))
                mse_results_ipod_tr.append(mse_ipod_tr)

                # mse_pool = np.log(np.mean((beta_hat_pool - sparse_data["beta"]) ** 2))
                # mse_wpool = np.log(np.mean((beta_hat_wpool - sparse_data["beta"]) ** 2))
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
                    "Trans-CO": mse_ipod_tr,

                    "f1_score_IPOD": f1_score,
                    "f1_score_Trans-CO": f1_score_,
                    "f1_score_Sparse-LTS": f1_sparse_lts
                }


                results_list.append(results)
                print(results)
            df_results = pd.DataFrame(results_list)
            # 保存到 Excel
            with pd.ExcelWriter(RESULTS_DIR / f"out_simu_results_ex3_B{B}_s{s}_n{n}_N{N}.xlsx") as writer:
                df_results.to_excel(writer, sheet_name="Model Results", index=False)

