from pathlib import Path
import math
import pandas as pd
from scipy.stats import wilcoxon


# =========================
# 基本设置
# =========================
DATA_DIR = Path(__file__).resolve().parent / "results"
OUT_TEX = Path("table_6_10.tex")

B = 50
s_list = [25, 75]
n = 200
N_list = [1500]

# Example 6 / 7 中的 o，对应表格中的 rho 或 rho_(k)
o_list = [0.05, 0.1, 0.2]

# Example 8 中 support 对应表格中的 mu
support_list = [2,4,6,8,10]

# Example 9 中 K_irrelevant 对应表格中的 K_I
K_irrelevant_list = [6, 4, 2]

# Example 10 中 tau
tau_list = [0.25, 0.5, 1, 2]


# =========================
# 方法列对应关系
# =========================
METHODS = [
    ("Sparse-LTS", "f1_score_Sparse-LTS"),
    (r"$\Theta$-IPOD", "f1_score_IPOD"),
    ("Trans-CO", "f1_score_Trans-CO"),
]


def fmt_float(x):
    """
    用于生成文件名，避免 1.0 这类格式。
    例如：0.05 -> '0.05', 0.1 -> '0.1', 1 -> '1'
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


def mean_se_percent_from_values(values):
    """
    给定百分比尺度下的一组 F1-score，返回 mean 和 standard error。
    """
    values = pd.Series(values).dropna()

    if len(values) == 0:
        raise ValueError("没有有效数值。")

    mean = values.mean()

    if len(values) == 1:
        se = 0.0
    else:
        se = values.std(ddof=1) / math.sqrt(len(values))

    return mean, se


def mean_se_percent_from_xlsx(path, col_name):
    """
    读取一个 xlsx 文件中指定 F1-score 列，
    返回 mean ± standard error，单位为百分比。

    这个函数保留原功能。
    """
    df = read_model_results(path)
    values = get_f1_percent_values(df, col_name, path=path)
    mean, se = mean_se_percent_from_values(values)
    return rf"${mean:.2f}\pm{se:.2f}$"


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

    # 如果所有 paired differences 都为 0，则没有显著差异
    if (diff.abs() < 1e-12).all():
        return 1.0

    try:
        result = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        return result.pvalue
    except ValueError:
        return float("nan")


def make_result_cells(path):
    """
    按表格中方法顺序返回：
    Sparse-LTS, Theta-IPOD, Trans-CO, Delta_best

    Delta_best =
        mean(F1_Trans-CO)
        - max(mean(F1_Sparse-LTS), mean(F1_Theta-IPOD))

    显著性检验：
        对 50 次 paired repetitions 做 Wilcoxon signed-rank test。
        星号加在 Delta_best 列。
    """
    df = read_model_results(path)

    values_dict = {}
    mean_se_dict = {}

    for method_name, col_name in METHODS:
        values = get_f1_percent_values(df, col_name, path=path)
        mean, se = mean_se_percent_from_values(values)

        values_dict[method_name] = values
        mean_se_dict[method_name] = (mean, se)

    sparse_mean, sparse_se = mean_se_dict["Sparse-LTS"]
    ipod_mean, ipod_se = mean_se_dict[r"$\Theta$-IPOD"]
    trans_mean, trans_se = mean_se_dict["Trans-CO"]

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
        rf"${sparse_mean:.2f}\pm{sparse_se:.2f}$",
        rf"${ipod_mean:.2f}\pm{ipod_se:.2f}$",
        rf"${trans_mean:.2f}\pm{trans_se:.2f}$",
        rf"${delta_best:.2f}{star}$",
    ]


def row_line(cells):
    return "&".join(cells) + r"\\"


def build_rows_for_one_s(s, N):
    """
    生成某一个 s 下 Example 6--10 的所有表格行。
    """
    lines = []
    total_rows = 18
    first_row_of_s = True

    examples = [
        {
            "ex": 6,
            "n_rows": 3,
            "params": [
                (
                    rf"$\rho={fmt_float(o)}$",
                    DATA_DIR / "ex6" / f"out_simu_results_ex6_B{B}_s{s}_o{fmt_float(o)}_N{N}.xlsx",
                )
                for o in o_list
            ],
        },
        {
            "ex": 7,
            "n_rows": 3,
            "params": [
                (
                    rf"$\rho_{{(k)}}={fmt_float(o)}$",
                    DATA_DIR / "ex7" / f"out_simu_results_ex7_B{B}_s{s}_o{fmt_float(o)}_N{N}.xlsx",
                )
                for o in o_list
            ],
        },
        {
            "ex": 8,
            "n_rows": 5,
            "params": [
                (
                    rf"$\mu={fmt_float(support)}$",
                    DATA_DIR / "ex8"
                    / f"out_simu_results_ex8_B{B}_s{s}_n{n}_N{N}_support{fmt_float(support)}.xlsx",
                )
                for support in support_list
            ],
        },
        {
            "ex": 9,
            "n_rows": 3,
            "params": [
                (
                    rf"$K_{{I}}={fmt_float(K_irrelevant)}$",
                    DATA_DIR / "ex9"
                    / f"out_simu_results_ex9_B{B}_s{s}_n{n}_N{N}_{fmt_float(K_irrelevant)}.xlsx",
                )
                for K_irrelevant in K_irrelevant_list
            ],
        },
        {
            "ex": 10,
            "n_rows": 4,
            "params": [
                (
                    rf"$\tau={fmt_float(tau)}$",
                    DATA_DIR / "ex10" / f"out_simu_results_ex10_B{B}_s{s}_tau{fmt_float(tau)}_N{N}.xlsx",
                )
                for tau in tau_list
            ],
        },
    ]

    for ex_idx, ex_info in enumerate(examples):
        ex = ex_info["ex"]
        params = ex_info["params"]
        n_rows = ex_info["n_rows"]

        for param_idx, (param_label, path) in enumerate(params):
            cells = []

            if first_row_of_s:
                cells.append(rf"\multirow{{{total_rows}}}{{*}}{{{s}}}")
                first_row_of_s = False
            else:
                cells.append("")

            if param_idx == 0:
                cells.append(rf"\multirow{{{n_rows}}}{{*}}{{\ref{{ex{ex}}}}}")
            else:
                cells.append("")

            cells.append(param_label)
            cells.extend(make_result_cells(path))

            lines.append(row_line(cells))

        if ex_idx != len(examples) - 1:
            lines.append(r"\cmidrule(lr){2-7}")

    return lines


def build_latex_table(N):
    lines = []

    lines.extend([
        r"\begin{table}[ht]",
        r"% \begin{center}",
        r"\caption{Evaluation results of different algorithms. The results include the mean and standard error of F1-score ($\%$) for outlier detection on the target dataset in Example \ref{ex6}-\ref{ex10}. "
        r"The column $\Delta_{\mathrm{best}}$ reports the absolute improvement in F1 percentage points of Trans-CO over the strongest competing baseline, defined as "
        r"$\Delta_{\mathrm{best}}=\mathrm{F1}_{\mathrm{Trans\text{-}CO}}-\max(\mathrm{F1}_{\mathrm{Sparse\text{-}LTS}},\mathrm{F1}_{\Theta\text{-IPOD}})$. "
        r"Statistical significance is assessed using a two-sided Wilcoxon signed-rank test over the 50 paired repetitions, comparing Trans-CO with the strongest competing baseline. "
        r"Significance levels are denoted by $^{*}p<0.05$, $^{**}p<0.01$, and $^{***}p<0.001$.}",
        r"\label{table:6-10}",
        r"% \begin{threeparttable}",
        r"\small",
        r"\begin{tabular}{c c c c c c c}",
        r"\toprule",
        r"  \multirow{2}{*}{$s$}&\multirow{2}{*}{Example}&\multirow{2}{*}{Parameter}& \multicolumn{3}{c}{Method}&\multirow{2}{*}{$\Delta_{\mathrm{best}}$}\\",
        r"& & & Sparse-LTS & $\Theta$-IPOD &Trans-CO & \\",
        r"\midrule",
    ])

    for s_idx, s in enumerate(s_list):
        lines.extend(build_rows_for_one_s(s, N))

        if s_idx != len(s_list) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"% \end{center}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    if len(N_list) != 1:
        raise ValueError("当前 LaTeX 模版没有 N 列。若有多个 N，请分别生成多个表格或增加 N 列。")

    N = N_list[0]
    latex = build_latex_table(N)

    OUT_TEX.write_text(latex, encoding="utf-8")

    print(f"LaTeX table has been written to: {OUT_TEX.resolve()}")
    print(latex)


if __name__ == "__main__":
    main()

