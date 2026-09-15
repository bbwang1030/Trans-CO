from Trans_CO import *


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "ex1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 选择需要运行的方法
# ============================================================

ALL_METHODS = [
    "IPOD",
    "Sparse-LTS",
    "Multi-task-Lasso",
    "PTL",
    "Trans-Lasso",
    "Trans-PtLR",
    "Trans-CO"
]

# ------------------------------------------------------------
# 这里控制本次需要运行的方法
# ------------------------------------------------------------

RUN_METHODS = [
    # "IPOD",
    "Sparse-LTS",
    # "Multi-task-Lasso",
    # "PTL",
    # "Trans-Lasso",
    # "Trans-PtLR",
    # "Trans-CO"
]

# 例如：
# RUN_METHODS = ["IPOD", "Sparse-LTS"]
#
# 或者全部运行：
# RUN_METHODS = ALL_METHODS


# ============================================================
# Simulation settings
# ============================================================

BASE_SEED = 2026
B = 50

s_ = [25, 75]

o = 0.05

n_ = [150, 200, 300]

N_ = [1000, 1500, 2000]

K = 5


# ============================================================
# Simulation
# ============================================================

for num_s, s in enumerate(s_):

    print("s", s)

    for num_N, N in enumerate(N_):

        print("N", N)

        for num_n, n in enumerate(n_):

            print("n", n)

            # ====================================================
            # 如果该组模拟已经运行完成，则直接跳过
            # ====================================================

            output_file = (
                    RESULTS_DIR
                    / f"ex1_B{B}_s{s}_n{n}_N{N}.xlsx"
            )

            if output_file.exists():
                print(
                    f"[SKIP] : "
                    f"s={s}, n={n}, N={N}"
                )
                continue

            print(
                f"[RUN] : "
                f"s={s}, n={n}, N={N}"
            )

            mse_results = []
            mse_results_ipod = []
            mse_results_Trans_CO = []

            results_list = []

            for b in range(B):

                setting_id = (
                    num_s * len(N_) + num_N
                ) * len(n_) + num_n

                rep_seed = BASE_SEED + setting_id * B + b

                set_random_seed(rep_seed)

                print(b, "seed =", rep_seed)

                # ====================================================
                # 生成模拟数据
                # ====================================================

                sparse_data = generate_data_with_outliers(
                    K=K,
                    s=s,
                    n=n,
                    N=N,
                    o=o
                )

                X_source = sparse_data["X_source"]
                Y_source = sparse_data["Y_source"]

                X_target = sparse_data["X_target"]
                Y_target = sparse_data["Y_target"]

                true_outlier = sparse_data["outlier_index"]


                # ====================================================
                # 先初始化所有结果
                # 没有运行的方法最后就是 NaN
                # ====================================================

                mse = np.nan
                mse_ipod = np.nan
                mse_Trans_CO = np.nan
                mse_mtl = np.nan
                mse_translasso = np.nan
                mse_sparse_lts = np.nan
                mse_transptlr = np.nan

                f1_score = np.nan
                f1_score_ = np.nan
                f1_sparse_lts = np.nan


                # ====================================================
                # PTL
                # ====================================================

                if "PTL" in RUN_METHODS:

                    beta_hat, wb_ptl = PTL_estimator(
                        X_source,
                        Y_source,
                        X_target,
                        Y_target
                    )

                    mse = np.log(
                        np.mean(
                            (beta_hat - sparse_data["beta"]) ** 2
                        )
                    )

                    mse_results.append(mse)


                # ====================================================
                # IPOD
                # ====================================================

                if "IPOD" in RUN_METHODS:

                    H = np.dot(
                        np.dot(
                            X_target,
                            inv(
                                np.dot(
                                    X_target.T,
                                    X_target
                                )
                            )
                        ),
                        X_target.T
                    )

                    IPOD1 = IPOD(
                        X_target,
                        Y_target,
                        H,
                        TOL=1e-4,
                        eta=1/20
                    )

                    detect_index_IPOD = np.where(
                        IPOD1["gamma"] != 0
                    )[0]

                    TP, FP, TN, FN, f1_score = score(
                        Y_target,
                        true_outlier,
                        detect_index_IPOD
                    )

                    mse_ipod = np.log(
                        np.mean(
                            (
                                IPOD1["beta"]
                                - sparse_data["beta"]
                            ) ** 2
                        )
                    )

                    mse_results_ipod.append(mse_ipod)


                # ====================================================
                # Trans-CO
                # ====================================================

                if "Trans-CO" in RUN_METHODS:

                    beta_hat_Trans_CO = IPODTL(
                        X_source,
                        Y_source,
                        X_target,
                        Y_target,
                        method="hard",
                        TOL=1e-4,
                        eta=1/20
                    )

                    detect_index_Trans_CO = np.where(
                        beta_hat_Trans_CO["gamma"] != 0
                    )[0]

                    TP_, FP_, TN_, FN_, f1_score_ = score(
                        Y_target,
                        true_outlier,
                        detect_index_Trans_CO
                    )

                    mse_Trans_CO = np.log(
                        np.mean(
                            (
                                beta_hat_Trans_CO["beta"]
                                - sparse_data["beta"]
                            ) ** 2
                        )
                    )

                    mse_results_Trans_CO.append(
                        mse_Trans_CO
                    )


                # ====================================================
                # Multi-task Lasso
                # ====================================================

                if "Multi-task-Lasso" in RUN_METHODS:

                    (
                        beta_hat_mtl,
                        best_mtl_lambda,
                        best_mtl_val_mse
                    ) = multitask_lasso_estimator(
                        X_source,
                        Y_source,
                        X_target,
                        Y_target,
                        random_state=rep_seed,
                        tol=1e-4
                    )

                    mse_mtl = np.log(
                        np.mean(
                            (
                                beta_hat_mtl
                                - sparse_data["beta"]
                            ) ** 2
                        )
                    )


                # ====================================================
                # R baselines
                #
                # Trans-Lasso
                # Sparse-LTS
                # Trans-PtLR
                # ====================================================

                R_METHODS = [
                    "Trans-Lasso",
                    "Sparse-LTS",
                    "Trans-PtLR"
                ]

                if any(
                    method in RUN_METHODS
                    for method in R_METHODS
                ):

                    r_baseline_results = run_r_baselines_temp(
                        X_source,
                        Y_source,
                        X_target,
                        Y_target,
                        seed=rep_seed
                    )


                    # ================================================
                    # Trans-Lasso
                    # ================================================

                    if "Trans-Lasso" in RUN_METHODS:

                        beta_hat_translasso = (
                            r_baseline_results[
                                "beta_hat_translasso"
                            ]
                        )

                        if beta_hat_translasso is not None:

                            mse_translasso = np.log(
                                np.mean(
                                    (
                                        beta_hat_translasso
                                        - sparse_data["beta"]
                                    ) ** 2
                                )
                            )


                    # ================================================
                    # Sparse-LTS
                    # ================================================

                    if "Sparse-LTS" in RUN_METHODS:

                        beta_hat_sparse_lts = (
                            r_baseline_results[
                                "beta_hat_sparse_lts"
                            ]
                        )

                        outlier_hat_sparse_lts = (
                            r_baseline_results[
                                "outlier_hat_sparse_lts"
                            ]
                        )


                        # --------------------------------------------
                        # Sparse-LTS F1 score
                        # --------------------------------------------

                        if outlier_hat_sparse_lts is not None:

                            detect_index_sparse_lts = np.where(
                                outlier_hat_sparse_lts != 0
                            )[0]

                            (
                                TP_slts,
                                FP_slts,
                                TN_slts,
                                FN_slts,
                                f1_sparse_lts
                            ) = score(
                                Y_target,
                                true_outlier,
                                detect_index_sparse_lts
                            )


                        # --------------------------------------------
                        # Sparse-LTS MSE
                        # --------------------------------------------

                        if beta_hat_sparse_lts is not None:

                            mse_sparse_lts = np.log(
                                np.mean(
                                    (
                                        beta_hat_sparse_lts
                                        - sparse_data["beta"]
                                    ) ** 2
                                )
                            )


                    # ================================================
                    # Trans-PtLR
                    # ================================================

                    if "Trans-PtLR" in RUN_METHODS:

                        beta_hat_transptlr = (
                            r_baseline_results[
                                "beta_hat_transptlr"
                            ]
                        )

                        if beta_hat_transptlr is not None:

                            mse_transptlr = np.log(
                                np.mean(
                                    (
                                        beta_hat_transptlr
                                        - sparse_data["beta"]
                                    ) ** 2
                                )
                            )


                # ====================================================
                # 保存结果
                # ====================================================

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


            # ========================================================
            # 保存到 Excel
            # ========================================================

            df_results = pd.DataFrame(results_list)

            with pd.ExcelWriter(output_file) as writer:

                df_results.to_excel(
                    writer,
                    sheet_name="Model Results",
                    index=False
                )

