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
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "ex1_eta"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_SEED = 2026
B=50
s_=[25]
etas = [1e-2,1e-3,1e-4,1e-5]
o=0.05
n = 200
N_=[2000]

K=5
for num_s,s in enumerate(s_):
    print("s",s)
    for num_N,N in enumerate(N_):
        print("N", N)
        for num_eta,eta in enumerate(etas):
            print("eta", eta)
            mse_results = []
            mse_results_ipod = []
            mse_results_Trans_CO = []
            results_list = []
            for b in range(B):
                setting_id = (num_s * 3 + 2) * len(etas) + num_eta
                rep_seed = BASE_SEED + setting_id * B + b
                set_random_seed(rep_seed)
                print(b, "seed =", rep_seed)
                # 生成模拟数据（假设generate_data返回X, y, beta）
                sparse_data = generate_data_with_outliers(K=K, s=s, n=n, N=N, o=o)
                X_source = sparse_data["X_source"]
                Y_source = sparse_data["Y_source"]
                X_target = sparse_data["X_target"]
                Y_target = sparse_data["Y_target"]
                true_outlier = sparse_data["outlier_index"]

                # # 使用PTL估计器（假设已实现五折CV选择λ参数）
                # # 不使用迁移学习下的IPOD算法

                #使用迁移学习下的IPOD算法
                start_time = time.perf_counter()

                beta_hat_Trans_CO = IPODTL(
                    X_source, Y_source,
                    X_target, Y_target,
                    method="hard",
                    TOL=eta,
                    eta=1 / 20
                )

                end_time = time.perf_counter()
                runtime_trans_co = end_time - start_time

                detect_index_Trans_CO = np.where(beta_hat_Trans_CO["gamma"] != 0)[0]
                TP_, FP_, TN_, FN_, f1_score_ = score(Y_target, true_outlier, detect_index_Trans_CO)

                # )
                # # ========== R baselines: temporary CSV files, automatically deleted ==========
                #     Y_target
                # )

                #         detect_index_sparse_lts
                #     )


                # # 计算MSE

                mse_Trans_CO = np.log(np.mean((beta_hat_Trans_CO["beta"] - sparse_data["beta"]) ** 2))
                mse_results_Trans_CO.append(mse_Trans_CO)


                results = {
                    "seed": rep_seed,
                    "Trans-CO": mse_Trans_CO,
                    "f1_score_Trans-CO": f1_score_,
                    "runtime_Trans-CO": runtime_trans_co
                }

                results_list.append(results)
                print(results)
            df_results = pd.DataFrame(results_list)
            # 保存到 Excel
            with pd.ExcelWriter(RESULTS_DIR / f"out_simu_results_ex1_eta_B{B}_s{s}_eta{eta}_N{N}.xlsx") as writer:
                df_results.to_excel(writer, sheet_name="Model Results", index=False)

