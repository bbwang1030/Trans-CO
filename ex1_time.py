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
import time

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "ex1_time"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_SEED = 2026
B=50
s_=[25,75]
o=0.05
n_ = [150,200,300]
N_=[1000,1500,2000]
# N_=[1000,1500]
K=5
for num_s,s in enumerate(s_):
    print("s",s)
    for num_N,N in enumerate(N_):
        print("N", N)
        for num_n,n in enumerate(n_):
            print("n", n)
            mse_results = []
            mse_results_ipod = []
            mse_results_Trans_CO = []
            results_list = []
            for b in range(B):
                setting_id = (num_s * len(N_) + num_N) * len(n_) + num_n
                rep_seed = BASE_SEED + setting_id * B + b
                set_random_seed(rep_seed)
                print(b, "seed =", rep_seed)

                method_times = {}

                sparse_data = generate_data_with_outliers(K=K, s=s, n=n, N=N, o=o)
                X_source = sparse_data["X_source"]
                Y_source = sparse_data["Y_source"]
                X_target = sparse_data["X_target"]
                Y_target = sparse_data["Y_target"]
                true_outlier = sparse_data["outlier_index"]

                # ========== PTL ==========
                t0 = time.perf_counter()
                beta_hat, wb_ptl = PTL_estimator(X_source, Y_source, X_target, Y_target)
                method_times["time_PTL"] = time.perf_counter() - t0

                # ========== IPOD ==========
                t0 = time.perf_counter()
                H = np.dot(np.dot(X_target, inv(np.dot(X_target.T, X_target))), X_target.T)
                IPOD1 = IPOD(X_target, Y_target, H, TOL=1e-4, eta=1 / 20)
                method_times["time_IPOD"] = time.perf_counter() - t0

                detect_index_IPOD = np.where(IPOD1["gamma"] != 0)[0]
                TP, FP, TN, FN, f1_score = score(Y_target, true_outlier, detect_index_IPOD)

                # ========== Trans-CO / IPODTL ==========
                t0 = time.perf_counter()
                beta_hat_Trans_CO = IPODTL(
                    X_source, Y_source,
                    X_target, Y_target,
                    method="hard",
                    TOL=1e-4,
                    eta=1 / 20
                )
                method_times["time_Trans-CO"] = time.perf_counter() - t0

                detect_index_Trans_CO = np.where(beta_hat_Trans_CO["gamma"] != 0)[0]
                TP_, FP_, TN_, FN_, f1_score_ = score(Y_target, true_outlier, detect_index_Trans_CO)

                # ========== Multi-task Lasso ==========
                t0 = time.perf_counter()
                beta_hat_mtl, best_mtl_lambda, best_mtl_val_mse = multitask_lasso_estimator(
                    X_source,
                    Y_source,
                    X_target,
                    Y_target,
                    random_state=rep_seed,
                    tol=1e-4
                )
                method_times["time_Multi-task-Lasso"] = time.perf_counter() - t0

                # ========== R baselines ==========
                # 注意：如果 run_r_baselines_temp 内部一次性跑完 Trans-Lasso、Sparse-LTS、Trans-PtLR，
                # 这里统计的是三个 R baseline 的总运行时间。
                t0 = time.perf_counter()
                r_baseline_results = run_r_baselines_temp(
                    X_source,
                    Y_source,
                    X_target,
                    Y_target,
                    seed=rep_seed,
                )
                r_total_time = time.perf_counter() - t0

                beta_hat_translasso = r_baseline_results["beta_hat_translasso"]
                beta_hat_sparse_lts = r_baseline_results["beta_hat_sparse_lts"]
                outlier_hat_sparse_lts = r_baseline_results["outlier_hat_sparse_lts"]
                beta_hat_transptlr = r_baseline_results["beta_hat_transptlr"]

                # 如果你的 run_r_baselines_temp 暂时没有分别返回 R 中各方法耗时，
                # 先把 R baseline 总时间保存下来。
                method_times["time_R_baselines_total"] = r_total_time

                # 如果之后你在 run_r_baselines_temp 里加了单独计时，可以用下面这种写法自动读取：
                method_times["time_Trans-Lasso"] = r_baseline_results.get("time_translasso", np.nan)
                method_times["time_Sparse-LTS"] = r_baseline_results.get("time_sparse_lts", np.nan)
                method_times["time_Trans-PtLR"] = r_baseline_results.get("time_transptlr", np.nan)

                f1_sparse_lts = np.nan
                if outlier_hat_sparse_lts is not None:
                    detect_index_sparse_lts = np.where(outlier_hat_sparse_lts != 0)[0]
                    TP_slts, FP_slts, TN_slts, FN_slts, f1_sparse_lts = score(
                        Y_target,
                        true_outlier,
                        detect_index_sparse_lts
                    )

                # ========== MSE ==========
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

                # 把运行时间并入每次重复的结果
                results.update(method_times)

                results_list.append(results)
                print(results)

            df_results = pd.DataFrame(results_list)

            # 统计每个方法运行时间的平均值和标准差
            time_cols = [col for col in df_results.columns if col.startswith("time_")]

            df_time_summary = pd.DataFrame({
                "method": [col.replace("time_", "") for col in time_cols],
                "time_mean": [df_results[col].mean(skipna=True) for col in time_cols],
                "time_std": [df_results[col].std(skipna=True) for col in time_cols]
            })

            with pd.ExcelWriter(RESULTS_DIR / f"out_simu_results_ex1_time_B{B}_s{s}_n{n}_N{N}.xlsx") as writer:
                df_results.to_excel(writer, sheet_name="Model Results", index=False)
                df_time_summary.to_excel(writer, sheet_name="Runtime Summary", index=False)

