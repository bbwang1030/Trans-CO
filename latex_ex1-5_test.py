
from pathlib import Path
import math
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# 基本设置
# ============================================================
DATA_DIR = Path(__file__).resolve().parent / "results"

ERROR_TYPE = "se"


METHODS = [
    ("Sparse-LTS", "f1_score_Sparse-LTS"),
    (r"$\Theta$-IPOD", "f1_score_IPOD"),
    ("Trans-CO", "f1_score_Trans-CO"),
]


# ============================================================
# 通用函数
# ============================================================
def fmt_num(x):
    """
    用于文件名和 LaTeX 显示，避免 1.0 这种格式。
    """
    if isinstance(x, int):
        return str(x)

    if isinstance(x, float) and x.is_integer():
        return str(int(x))

    return str(x)


def read_model_results(path):
    """
    读取 xlsx 中的 Model Results 工作表。
    """
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    try:
        df = pd.read_excel(path, sheet_name="Model Results")
    except ValueError:
        df = pd.read_excel(path, sheet_name=0)

    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_f1_percent_values(df, col_name, path=None):
    """
    读取指定 F1-score 列，返回百分比尺度下的原始重复实验结果。
    """
    if col_name not in df.columns:
        if path is None:
            raise KeyError(
                f"找不到列 {col_name}。\n"
                f"当前列名为: {list(df.columns)}"
            )
        else:
            raise KeyError(
                f"文件 {path.name} 中找不到列 {col_name}。\n"
                f"当前列名为: {list(df.columns)}"
            )

    values = pd.to_numeric(df[col_name], errors="coerce") * 100
    return values


def mean_error_percent_from_values(values, error_type="se"):
    """
    给定百分比尺度下的一组 F1-score，返回 mean 和 error。

    error_type = "se"  : standard error = std / sqrt(number of valid runs)
    error_type = "std" : standard deviation
    """
    values = pd.Series(values).dropna()

    if len(values) == 0:
        raise ValueError("没有有效数值。")

    mean = values.mean()

    if len(values) == 1:
        error = 0.0
    else:
        std = values.std(ddof=1)

        if error_type == "se":
            error = std / math.sqrt(len(values))
        elif error_type == "std":
            error = std
        else:
            raise ValueError("ERROR_TYPE 只能是 'se' 或 'std'。")

    return mean, error


def mean_error_percent_from_xlsx(path, col_name, error_type="se"):
    """
    读取指定 F1-score 列，返回 mean ± error，单位为百分比。

    这个函数保留原功能。
    """
    df = read_model_results(path)
    values = get_f1_percent_values(df, col_name, path=path)
    mean, error = mean_error_percent_from_values(values, error_type)
    return rf"${mean:.2f}\pm{error:.2f}$"


def significance_marker(p_value):
    """
    根据 Wilcoxon signed-rank test 的 p-value 返回显著性星号。
    """
    if pd.isna(p_value):
        return ""

    if p_value < 0.001:
        return r"^{***}"
    elif p_value < 0.01:
        return r"^{**}"
    elif p_value < 0.05:
        return r"^{*}"
    else:
        return ""


def wilcoxon_signed_rank_pvalue(trans_values, baseline_values):
    """
    使用 50 次 paired repetitions 做 two-sided Wilcoxon signed-rank test。

    检验对象：
        diff = F1_Trans-CO - F1_best_baseline
    """
    paired = pd.concat(
        [
            pd.Series(trans_values, name="trans"),
            pd.Series(baseline_values, name="baseline"),
        ],
        axis=1,
    ).dropna()

    if len(paired) == 0:
        return float("nan")

    diff = paired["trans"] - paired["baseline"]

    # 如果所有 paired differences 都是 0，则没有显著差异
    if (diff.abs() < 1e-12).all():
        return 1.0

    try:
        result = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        return result.pvalue
    except ValueError:
        return float("nan")


def row_line(cells):
    return " & ".join(str(x) for x in cells) + r"\\"


def method_cells_from_file(path):
    """
    返回 Sparse-LTS, Theta-IPOD, Trans-CO 三个方法的结果，
    以及新增的 Delta_best 列。

    Delta_best =
        mean(F1_Trans-CO) -
        max(mean(F1_Sparse-LTS), mean(F1_Theta-IPOD))

    显著性检验：
        对 50 次 paired repetitions 做 Wilcoxon signed-rank test。
        星号加在 Delta_best 列。
    """
    df = read_model_results(path)

    values_dict = {}
    mean_error_dict = {}

    for method_name, col_name in METHODS:
        values = get_f1_percent_values(df, col_name, path=path)
        mean, error = mean_error_percent_from_values(values, ERROR_TYPE)

        values_dict[method_name] = values
        mean_error_dict[method_name] = (mean, error)

    sparse_mean, sparse_error = mean_error_dict["Sparse-LTS"]
    ipod_mean, ipod_error = mean_error_dict[r"$\Theta$-IPOD"]
    trans_mean, trans_error = mean_error_dict["Trans-CO"]

    # 在每个 setting 下，根据平均 F1 选择 strongest competing baseline
    if sparse_mean >= ipod_mean:
        best_baseline_name = "Sparse-LTS"
        best_baseline_mean = sparse_mean
    else:
        best_baseline_name = r"$\Theta$-IPOD"
        best_baseline_mean = ipod_mean

    delta_best = trans_mean - best_baseline_mean

    p_value = wilcoxon_signed_rank_pvalue(
        values_dict["Trans-CO"],
        values_dict[best_baseline_name],
    )

    star = significance_marker(p_value)

    return [
        rf"${sparse_mean:.2f}\pm{sparse_error:.2f}$",
        rf"${ipod_mean:.2f}\pm{ipod_error:.2f}$",
        rf"${trans_mean:.2f}\pm{trans_error:.2f}$",
        rf"${delta_best:.2f}{star}$",
    ]


# ============================================================
# ex1 / ex3 / ex4 / ex5:
# 表格结构为 s, N, n, Method, Delta_best
# ============================================================
def build_s_N_n_table(
    ex_id,
    B,
    s_list,
    N_list,
    n_list,
    label=None,
):
    if label is None:
        label = f"table:ex{ex_id}"

    lines = []

    lines.extend([
        r"\begin{table}[ht]",
        r"% \begin{center}",
        rf"\caption{{Evaluation results of different algorithms. The results include the mean and standard error of F1-score ($\%$) for outlier detection on the target dataset in Example \ref{{ex{ex_id}}}. "
        r"The column $\Delta_{\mathrm{best}}$ reports the absolute improvement in F1 percentage points of Trans-CO over the strongest competing baseline, defined as "
        r"$\Delta_{\mathrm{best}}=\mathrm{F1}_{\mathrm{Trans\text{-}CO}}-\max(\mathrm{F1}_{\mathrm{Sparse\text{-}LTS}},\mathrm{F1}_{\Theta\text{-IPOD}})$. "
        r"Statistical significance is assessed using a two-sided Wilcoxon signed-rank test over the 50 paired repetitions, comparing Trans-CO with the strongest competing baseline. "
        r"Significance levels are denoted by $^{*}p<0.05$, $^{**}p<0.01$, and $^{***}p<0.001$.}",
        rf"\label{{{label}}}",
        r"% \begin{threeparttable}",
        r"\small",
        r"\setlength{\tabcolsep}{15pt}",
        r"% \renewcommand{\arraystretch}{1}",
        r"\begin{tabular}{c c c c c c c}",
        r"\toprule",
        r"  \multirow{2}{*}{$s$}&\multirow{2}{*}{$N$}&\multirow{2}{*}{$n$}& \multicolumn{3}{c}{Method}&\multirow{2}{*}{$\Delta_{\mathrm{best}}$}\\",
        r"& & & Sparse-LTS & $\Theta$-IPOD & Trans-CO & \\",
        r"\midrule",
    ])

    rows_per_s = len(N_list) * len(n_list)
    rows_per_N = len(n_list)

    for s_idx, s in enumerate(s_list):
        first_row_of_s = True

        for N_idx, N in enumerate(N_list):
            first_row_of_N = True

            for n in n_list:
                filename = (
                    f"out_simu_results_ex{ex_id}_"
                    f"B{B}_s{fmt_num(s)}_n{fmt_num(n)}_N{fmt_num(N)}.xlsx"
                )
                path = DATA_DIR / f"ex{ex_id}" / filename

                cells = []

                if first_row_of_s:
                    cells.append(rf"\multirow{{{rows_per_s}}}{{*}}{{{fmt_num(s)}}}")
                    first_row_of_s = False
                else:
                    cells.append("")

                if first_row_of_N:
                    cells.append(rf"\multirow{{{rows_per_N}}}{{*}}{{{fmt_num(N)}}}")
                    first_row_of_N = False
                else:
                    cells.append("")

                cells.append(fmt_num(n))
                cells.extend(method_cells_from_file(path))

                lines.append(row_line(cells))

            if N_idx != len(N_list) - 1:
                lines.append(r"\cmidrule(lr){2-7}")

        if s_idx != len(s_list) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"% \end{center}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# ============================================================
# 表格结构为 s, o, K, Method, Delta_best
# 文件名格式：
# out_simu_results_ex2_s{s}_o{o}_K{K}.xlsx
# ============================================================
def build_ex2_table(
    s_list,
    o_list,
    K_list,
    label="table:ex2",
):
    ex_id = 2

    lines = []

    lines.extend([
        r"\begin{table}[ht]",
        r"% \begin{center}",
        rf"\caption{{Evaluation results of different algorithms. The results include the mean and standard error of F1-score ($\%$) for outlier detection on the target dataset in Example \ref{{ex{ex_id}}}. "
        r"The column $\Delta_{\mathrm{best}}$ reports the absolute improvement in F1 percentage points of Trans-CO over the strongest competing baseline, defined as "
        r"$\Delta_{\mathrm{best}}=\mathrm{F1}_{\mathrm{Trans\text{-}CO}}-\max(\mathrm{F1}_{\mathrm{Sparse\text{-}LTS}},\mathrm{F1}_{\Theta\text{-IPOD}})$. "
        r"Statistical significance is assessed using a two-sided Wilcoxon signed-rank test over the 50 paired repetitions, comparing Trans-CO with the strongest competing baseline. "
        r"Significance levels are denoted by $^{*}p<0.05$, $^{**}p<0.01$, and $^{***}p<0.001$.}",
        rf"\label{{{label}}}",
        r"% \begin{threeparttable}",
        r"\small",
        r"\setlength{\tabcolsep}{15pt}",
        r"% \renewcommand{\arraystretch}{1}",
        r"\begin{tabular}{c c c c c c c}",
        r"\toprule",
        r"  \multirow{2}{*}{$s$}&\multirow{2}{*}{$o$}&\multirow{2}{*}{$K$}& \multicolumn{3}{c}{Method}&\multirow{2}{*}{$\Delta_{\mathrm{best}}$}\\",
        r"& & & Sparse-LTS & $\Theta$-IPOD & Trans-CO & \\",
        r"\midrule",
    ])

    rows_per_s = len(o_list) * len(K_list)
    rows_per_o = len(K_list)

    for s_idx, s in enumerate(s_list):
        first_row_of_s = True

        for o_idx, o in enumerate(o_list):
            first_row_of_o = True

            for K in K_list:
                filename = (
                    f"out_simu_results_ex2_"
                    f"s{fmt_num(s)}_o{fmt_num(o)}_K{fmt_num(K)}.xlsx"
                )
                path = DATA_DIR / "ex2_K_O" / filename

                cells = []

                if first_row_of_s:
                    cells.append(rf"\multirow{{{rows_per_s}}}{{*}}{{{fmt_num(s)}}}")
                    first_row_of_s = False
                else:
                    cells.append("")

                if first_row_of_o:
                    cells.append(rf"\multirow{{{rows_per_o}}}{{*}}{{{fmt_num(o)}}}")
                    first_row_of_o = False
                else:
                    cells.append("")

                cells.append(fmt_num(K))
                cells.extend(method_cells_from_file(path))

                lines.append(row_line(cells))

            if o_idx != len(o_list) - 1:
                lines.append(r"\cmidrule(lr){2-7}")

        if s_idx != len(s_list) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"% \end{center}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# ============================================================
# 主程序：分别 print ex1 到 ex5
# ============================================================
def main():
    tables = []

    # # ex1
    #     )
    # )
    # # ex2
    #     )
    # )

    # -------------------------
    # ex1
    # -------------------------
    tables.append(
        build_s_N_n_table(
            ex_id=1,
            B=50,
            s_list=[25, 75],
            n_list=[150, 200, 300],
            N_list=[1000, 1500, 2000],
            label="table:ex1",
        )
    )

    # -------------------------
    # ex2
    # -------------------------
    tables.append(
        build_ex2_table(
            s_list=[30, 70],
            o_list=[0.05, 0.1],
            K_list=[2, 4, 6, 8, 10],
            label="table:ex2",
        )
    )

    # -------------------------
    # ex3
    # 注意：你给的是 s_ = [75, 25]，这里保留该顺序
    # -------------------------
    tables.append(
        build_s_N_n_table(
            ex_id=3,
            B=50,
            s_list=[75, 25],
            n_list=[150, 200, 300],
            N_list=[1000, 1500, 2000],
            label="table:ex3",
        )
    )

    # -------------------------
    # ex4
    # -------------------------
    tables.append(
        build_s_N_n_table(
            ex_id=4,
            B=50,
            s_list=[25, 75],
            n_list=[150, 200, 300],
            N_list=[1000, 1500, 2000],
            label="table:ex4",
        )
    )

    # -------------------------
    # ex5
    # -------------------------
    tables.append(
        build_s_N_n_table(
            ex_id=5,
            B=50,
            s_list=[25, 75],
            n_list=[10, 30, 50, 70, 90],
            N_list=[1000, 1500, 2000],
            label="table:ex5",
        )
    )

    for i, latex in enumerate(tables, start=1):
        print(f"% ==================== Example {i} ====================")
        print(latex)
        print("\n")


if __name__ == "__main__":
    main()
