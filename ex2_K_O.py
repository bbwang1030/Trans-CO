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
from pathlib import Path
from Trans_CO import *

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "ex2_K_O"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_SEED = 2026
B=50
s_=[30,70]
o_=[0.05,0.1]
n = 200
N = 1000
K_=[2,4,6,8,10]
for num_s,s in enumerate(s_):
    for num_o,o in enumerate(o_):
        print("s",o)
        for num_K,K in enumerate(K_):
            print("K", K)
            mse_results = []
            mse_results_ipod = []
            mse_results_Trans_CO = []
            results_list = []
            for b in range(B):
                setting_id = (num_s * len(o_) + num_o) * len(K_) + num_K
                rep_seed = BASE_SEED + setting_id * B + b
                set_random_seed(rep_seed)
                print(b)
                # 生成模拟数据（假设generate_data返回X, y, beta）
                sparse_data = generate_data_with_outliers(K=K, s=s, n=n, N=N, o=o)
                X_source = sparse_data["X_source"]
                Y_source = sparse_data["Y_source"]
                X_target = sparse_data["X_target"]
                Y_target = sparse_data["Y_target"]
                true_outlier = sparse_data["outlier_index"]

                # 使用PTL估计器（假设已实现五折CV选择λ参数）
                beta_hat, wb_ptl = PTL_estimator(X_source, Y_source, X_target, Y_target)

                # 不使用迁移学习下的IPOD算法
                H = np.dot(np.dot(X_target, inv(np.dot(X_target.T, X_target))), X_target.T)
                IPOD1 = IPOD(X_target, Y_target, H, TOL=1e-4, eta=1 / 20)
                detect_index_IPOD = np.where(IPOD1["gamma"] != 0)[0]
                TP, FP, TN, FN, f1_score = score(Y_target, true_outlier, detect_index_IPOD)

                # 使用迁移学习下的IPOD算法
                beta_hat_Trans_CO = IPODTL(X_source, Y_source, X_target, Y_target, method="hard", TOL=1e-4, eta=1 / 20)
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
            with pd.ExcelWriter(RESULTS_DIR / f"out_simu_results_ex2_s{s}_o{o}_K{K}.xlsx") as writer:
                df_results.to_excel(writer, sheet_name="Model Results", index=False)

