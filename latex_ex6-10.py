from pathlib import Path
import math
import pandas as pd


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
support_list = [1, 2, 3, 4, 5]

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


def mean_se_percent_from_xlsx(path, col_name):
    """
    读取一个 xlsx 文件中指定 F1-score 列，
    返回 mean ± standard error，单位为百分比。
    """
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    try:
        df = pd.read_excel(path, sheet_name="Model Results")
    except ValueError:
        df = pd.read_excel(path, sheet_name=0)

    col_map = {str(c).strip(): c for c in df.columns}

    if col_name not in col_map:
        raise KeyError(
            f"文件 {path.name} 中找不到列 {col_name}。\n"
            f"当前列名为: {list(df.columns)}"
        )

    values = pd.to_numeric(df[col_map[col_name]], errors="coerce").dropna()

    if len(values) == 0:
        raise ValueError(f"文件 {path.name} 的列 {col_name} 没有有效数值。")

    mean = values.mean() * 100

    if len(values) == 1:
        se = 0.0
    else:
        se = values.std(ddof=1) / math.sqrt(len(values)) * 100

    return rf"${mean:.2f}\pm{se:.2f}$"


def make_result_cells(path):
    """
    按表格中方法顺序返回：
    Sparse-LTS, Theta-IPOD, Trans-CO
    """
    return [mean_se_percent_from_xlsx(path, col) for _, col in METHODS]


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
            lines.append(r"\cmidrule(lr){2-6}")

    return lines


def build_latex_table(N):
    lines = []

    lines.extend([
        r"\begin{table}[ht]",
        r"% \begin{center}",
        r"\caption{Evaluation results of different algorithms. The results include the mean and standard error of F1-score ($\%$) for outlier detection on the target dataset in Example \ref{ex6}-\ref{ex10}.}",
        r"\label{table:6-10}",
        r"% \begin{threeparttable}",
        r"\small",
        r"\begin{tabular}{c c c c c c}",
        r"\toprule",
        r"  \multirow{2}{*}{$s$}&\multirow{2}{*}{Example}&\multirow{2}{*}{Parameter}& \multicolumn{3}{c}{Method}\\",
        r"& & & Sparse-LTS & $\Theta$-IPOD &Trans-CO\\",
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
