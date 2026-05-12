#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot ECM measurement cycle vs DCIR cycle used.")
    ap.add_argument("--table_csv", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--group_tag", default="HYCL")
    ap.add_argument("--max_cycle", type=float, default=50)
    args = ap.parse_args()

    df = pd.read_csv(args.table_csv)
    df = df[df["group_tag"] == args.group_tag].copy()
    df = df[pd.to_numeric(df["cycle_t"], errors="coerce") <= args.max_cycle].copy()
    df["cycle_t"] = pd.to_numeric(df["cycle_t"], errors="coerce")
    df["feat_dcir_cycle_used"] = pd.to_numeric(df.get("feat_dcir_cycle_used"), errors="coerce")
    df["feat_ecm_measurement_cycle"] = pd.to_numeric(df.get("feat_ecm_measurement_cycle"), errors="coerce")

    both = df.dropna(subset=["cycle_t", "feat_dcir_cycle_used", "feat_ecm_measurement_cycle"]).copy()
    if both.empty:
        raise ValueError("No rows with both DCIR cycle and ECM measurement cycle.")

    counts = (
        both.groupby(["feat_dcir_cycle_used", "feat_ecm_measurement_cycle"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sizes = 80 + counts["count"].to_numpy(dtype=float) * 8
    sc = ax.scatter(
        counts["feat_dcir_cycle_used"],
        counts["feat_ecm_measurement_cycle"],
        s=sizes,
        c=counts["count"],
        cmap="Blues",
        edgecolor="#1f2937",
        linewidth=0.8,
        alpha=0.9,
    )

    all_cycles = pd.concat([both["feat_dcir_cycle_used"], both["feat_ecm_measurement_cycle"], both["cycle_t"]]).dropna()
    lo = max(0, float(all_cycles.min()) - 3)
    hi = float(all_cycles.max()) + 3
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#dc2626", linewidth=1.3, label="exact alignment (y = x)")

    for _, r in counts.iterrows():
        ax.text(
            r["feat_dcir_cycle_used"],
            r["feat_ecm_measurement_cycle"] + 0.8,
            f"n={int(r['count'])}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("DCIR cycle used")
    ax.set_ylabel("ECM measurement cycle")
    ax.set_title(f"ECM Cycle vs DCIR Cycle Used ({args.group_tag}, cycle_t <= {args.max_cycle:g})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row count")

    exact = int((both["feat_dcir_cycle_used"] == both["feat_ecm_measurement_cycle"]).sum())
    summary = (
        f"Rows with both cycles: {len(both)}\n"
        f"Exact ECM=DCIR cycle: {exact}/{len(both)}\n"
        f"Unique cycle pairs: {len(counts)}"
    )
    ax.text(
        0.98,
        0.02,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.94},
    )

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close(fig)
    print(f"[INFO] Saved cycle-pair plot: {out}")


if __name__ == "__main__":
    main()
