#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot simple ECM/DCIR cycle-alignment summary.")
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

    dcir = df.dropna(subset=["cycle_t", "feat_dcir_cycle_used"]).copy()
    ecm = df.dropna(subset=["cycle_t", "feat_ecm_measurement_cycle"]).copy()
    both = df.dropna(subset=["feat_dcir_cycle_used", "feat_ecm_measurement_cycle"]).copy()

    checks = [
        ("sample cycle =\nDCIR cycle", int((dcir["cycle_t"] == dcir["feat_dcir_cycle_used"]).sum()), len(dcir)),
        ("sample cycle =\nECM cycle", int((ecm["cycle_t"] == ecm["feat_ecm_measurement_cycle"]).sum()), len(ecm)),
        ("ECM cycle =\nDCIR cycle", int((both["feat_ecm_measurement_cycle"] == both["feat_dcir_cycle_used"]).sum()), len(both)),
    ]

    labels = [x[0] for x in checks]
    matches = [x[1] for x in checks]
    totals = [x[2] for x in checks]
    percents = [(m / t * 100.0) if t else 0.0 for m, t in zip(matches, totals)]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(labels, percents, color=["#2563eb", "#dc2626", "#7c3aed"], alpha=0.85)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Exact cycle-alignment rate (%)")
    ax.set_title(f"ECM/DCIR Exact Cycle Alignment ({args.group_tag}, cycle_t <= {args.max_cycle:g})")
    ax.grid(axis="y", alpha=0.25)

    for bar, m, t, pct in zip(bars, matches, totals, percents):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(2.0, pct + 2.0),
            f"{m}/{t}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    note = (
        "Interpretation: 0% exact alignment means the compared quantities are\n"
        "not measured/selected at the same cycle in the current feature table."
    )
    ax.text(
        0.5,
        -0.22,
        note,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved alignment summary plot: {out}")


if __name__ == "__main__":
    main()
