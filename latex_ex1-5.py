from pathlib import Path
import math
import pandas as pd


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


def mean_error_percent_from_xlsx(path, col_name, error_type="se"):
    """
    读取指定 F1-score 列，返回 mean ± error，单位为百分比。

    error_type = "se"  : standard error = std / sqrt(number of valid runs)
    error_type = "std" : standard deviation
    """
    df = read_model_results(path)

    if col_name not in df.columns:
        raise KeyError(
            f"文件 {path.name} 中找不到列 {col_name}。\n"
            f"当前列名为: {list(df.columns)}"
        )

    values = pd.to_numeric(df[col_name], errors="coerce").dropna()

    if len(values) == 0:
        raise ValueError(f"文件 {path.name} 的列 {col_name} 没有有效数值。")

    values = values * 100

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

    return rf"${mean:.2f}\pm{error:.2f}$"


def row_line(cells):
    return " & ".join(str(x) for x in cells) + r"\\"


def method_cells_from_file(path):
    """
    返回 Sparse-LTS, Theta-IPOD, Trans-CO 三个方法的结果。
    """
    return [
        mean_error_percent_from_xlsx(path, col_name, ERROR_TYPE)
        for _, col_name in METHODS
    ]


# ============================================================
# ex1 / ex3 / ex4 / ex5:
# 表格结构为 s, N, n, Method
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
        rf"\caption{{Evaluation results of different algorithms. The results include the mean and standard error of F1-score ($\%$) for outlier detection on the target dataset in Example \ref{{ex{ex_id}}}.}}",
        rf"\label{{{label}}}",
        r"% \begin{threeparttable}",
        r"\small",
        r"\setlength{\tabcolsep}{15pt}",
        r"% \renewcommand{\arraystretch}{1}",
        r"\begin{tabular}{c c c c c c}",
        r"\toprule",
        r"  \multirow{2}{*}{$s$}&\multirow{2}{*}{$N$}&\multirow{2}{*}{$n$}& \multicolumn{3}{c}{Method}\\",
        r"& & & Sparse-LTS & $\Theta$-IPOD & Trans-CO\\",
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
                lines.append(r"\cmidrule(lr){2-6}")

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
# 表格结构为 s, o, K, Method
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
        rf"\caption{{Evaluation results of different algorithms. The results include the mean and standard error of F1-score ($\%$) for outlier detection on the target dataset in Example \ref{{ex{ex_id}}}.}}",
        rf"\label{{{label}}}",
        r"% \begin{threeparttable}",
        r"\small",
        r"\setlength{\tabcolsep}{15pt}",
        r"% \renewcommand{\arraystretch}{1}",
        r"\begin{tabular}{c c c c c c}",
        r"\toprule",
        r"  \multirow{2}{*}{$s$}&\multirow{2}{*}{$o$}&\multirow{2}{*}{$K$}& \multicolumn{3}{c}{Method}\\",
        r"& & & Sparse-LTS & $\Theta$-IPOD & Trans-CO\\",
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
                lines.append(r"\cmidrule(lr){2-6}")

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
