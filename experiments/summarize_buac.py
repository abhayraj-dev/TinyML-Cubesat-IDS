"""Aggregate per-architecture BUAC gate evaluations into buac_summary.csv."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import RESULTS_DIR

ARCH_TAGS = [("MLP", "mlp"), ("GRU", "gru"), ("CNN1D", "cnn1d")]


def main():
    rows = []
    for arch, tag in ARCH_TAGS:
        path = os.path.join(RESULTS_DIR, f"buac_{tag}_gate_eval.csv")
        if not os.path.exists(path):
            print(f"skip missing {path}")
            continue
        d = pd.read_csv(path)
        if "scope" not in d.columns:
            d["scope"] = "all"
        for _, r in d.iterrows():
            rows.append(dict(
                arch=arch, scope=r["scope"], gate=r["gate"],
                safe_accept=r["gate_safe_accept"],
                unsafe_accept=r["gate_unsafe_accept"],
                balanced_acc=r["gate_balanced_acc"],
                precision=r["gate_precision"], auroc=r["gate_auroc"],
                n=int(r["gate_n"]), safe_n=int(r["gate_safe_n"]),
                unsafe_n=int(r["gate_unsafe_n"]),
                mem_budget_KB=r.get("mem_budget_KB"),
                up_budget_KB=r.get("up_budget_KB")))
    if not rows:
        raise SystemExit("no BUAC gate_eval files found")
    out = pd.DataFrame(rows).sort_values(
        ["arch", "scope", "gate"], ignore_index=True)
    path = os.path.join(RESULTS_DIR, "buac_summary.csv")
    out.to_csv(path, index=False)
    print(f"Wrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
