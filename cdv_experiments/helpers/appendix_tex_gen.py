"""Generate LaTeX tabular fragments for the appendix full-grid tables.
Reads the CSVs produced by appendix_grids.py and writes one .tex fragment
per table into cdv_experiments/helpers/_appendix_tex/.
"""
import os
import re
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(os.path.dirname(__file__), "_appendix_tex")
os.makedirs(OUT_DIR, exist_ok=True)

LEARNER_HEADS = {
    "DR_RF": "DR (RF)", "S_RF": "S (RF)", "S_Linear": "S (Linear)",
    "T_RF": "T (RF)", "X_RF": "X (RF)",
}

_SCI_RE = re.compile(r"(-?\d+\.?\d*)e([+-]\d+)")


def _fix_sci(cell: str) -> str:
    """Turn '1.23e+06' into '1.23{\\times}10^{6}' and wrap the whole cell in math mode."""
    def repl(m):
        mant, exp = m.group(1), int(m.group(2))
        return rf"{mant}{{\times}}10^{{{exp}}}"
    body = _SCI_RE.sub(repl, cell)
    body = body.replace(r"\pm", r"\pm").replace("[", r"\,[").replace("$", "")
    return f"${body}$"


def csv_to_tabular(path, caption, label):
    df = pd.read_csv(path, index_col=0)
    cols = list(df.columns)
    header = " & ".join(LEARNER_HEADS[c] for c in cols)
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\tiny")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "c" * len(cols) + "}")
    lines.append(r"\hline")
    lines.append(r"Method & " + header + r" \\")
    lines.append(r"\hline")
    for method, row in df.iterrows():
        cells = " & ".join(_fix_sci(str(row[c])) for c in cols)
        lines.append(f"{method} & {cells} " + r"\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    # Sepsis
    for metric, mlabel in [("ate_mse", "ATE MSE"), ("cate_mse", "CATE MSE")]:
        path = os.path.join(_ROOT, "cdv_experiments", "sepsis", "artifacts", f"appendix_full_grid_{metric}.csv")
        cap = (f"Sepsis: full method $\\times$ learner grid, {mlabel} "
               f"(mean $\\pm$ std across 20 outer seeds; bracketed values are the paired "
               f"95\\% CI of $\\Delta=$ CDV Separate $-$ method, two-sided bootstrap).")
        tex = csv_to_tabular(path, cap, f"tab:appendix_sepsis_{metric}")
        with open(os.path.join(OUT_DIR, f"sepsis_{metric}.tex"), "w", encoding="utf-8") as f:
            f.write(tex)

    # Synthetic, per alpha
    for alpha in ["0.00", "0.25", "0.50", "0.75", "1.00"]:
        for metric, mlabel in [("ate_mse", "ATE MSE"), ("cate_mse", "CATE MSE")]:
            path = os.path.join(_ROOT, "cdv_experiments", "synthetic", "artifacts",
                                 f"appendix_full_grid_{metric}_alpha_{alpha}.csv")
            cap = (f"Synthetic ($\\alpha={alpha}$): full method $\\times$ learner grid, {mlabel} "
                   f"(mean $\\pm$ std across 20 outer seeds; bracketed values are the paired "
                   f"95\\% CI of $\\Delta=$ CDV Separate $-$ method, two-sided bootstrap).")
            tex = csv_to_tabular(path, cap, f"tab:appendix_synth_{metric}_{alpha}")
            with open(os.path.join(OUT_DIR, f"synth_{metric}_{alpha}.tex"), "w", encoding="utf-8") as f:
                f.write(tex)

    print("done", os.listdir(OUT_DIR))


if __name__ == "__main__":
    main()
