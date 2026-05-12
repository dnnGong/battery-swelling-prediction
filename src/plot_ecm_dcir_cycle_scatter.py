#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot cycle alignment between model sample cycle, DCIR cycle, and ECM measurement cycle."
    )
    ap.add_argument("--table_csv", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--group_tag", default="HYCL")
    ap.add_argument("--max_cycle", type=float, default=50)
    args = ap.parse_args()

    df = pd.read_csv(args.table_csv)
    df = df[df["group_tag"] == args.group_tag].copy()
    df = df[pd.to_numeric(df["cycle_t"], errors="coerce") <= args.max_cycle].copy()
    df = df.dropna(subset=["cycle_t", "feat_dcir_cycle_used"]).copy()
    if df.empty:
        raise ValueError("No rows with cycle_t and feat_dcir_cycle_used after filtering.")

    df["cycle_t"] = pd.to_numeric(df["cycle_t"], errors="coerce")
    df["feat_dcir_cycle_used"] = pd.to_numeric(df["feat_dcir_cycle_used"], errors="coerce")
    df["feat_ecm_measurement_cycle"] = pd.to_numeric(df.get("feat_ecm_measurement_cycle"), errors="coerce")
    df = df.dropna(subset=["cycle_t", "feat_dcir_cycle_used"]).copy()
    df = df.sort_values(["cell_key", "cycle_t"]).reset_index(drop=True)
    df["sample_idx"] = np.arange(len(df))

    ecm_nonnull = df["feat_ecm_measurement_cycle"].notna()
    dcir_exact = df["cycle_t"] == df["feat_dcir_cycle_used"]
    ecm_exact = ecm_nonnull & (df["cycle_t"] == df["feat_ecm_measurement_cycle"])
    ecm_dcir_exact = ecm_nonnull & (df["feat_ecm_measurement_cycle"] == df["feat_dcir_cycle_used"])

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 8.5),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        sharex=False,
    )

    ax1.scatter(df["sample_idx"], df["cycle_t"], s=14, color="#111827", alpha=0.45, label="sample cycle_t")
    ax1.scatter(
        df["sample_idx"],
        df["feat_dcir_cycle_used"],
        s=18,
        color="#2563eb",
        alpha=0.75,
        label="DCIR cycle used",
    )
    if ecm_nonnull.any():
        ax1.scatter(
            df.loc[ecm_nonnull, "sample_idx"],
            df.loc[ecm_nonnull, "feat_ecm_measurement_cycle"],
            s=28,
            marker="x",
            color="#dc2626",
            alpha=0.9,
            label="ECM measurement cycle",
        )
    ax1.set_ylabel("Cycle")
    ax1.set_title(f"ECM/DCIR Cycle Alignment ({args.group_tag}, cycle_t <= {args.max_cycle:g})")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left", frameon=False, ncol=3)

    dcir_delta = df["cycle_t"] - df["feat_dcir_cycle_used"]
    ax2.hist(dcir_delta.dropna(), bins=np.arange(dcir_delta.min() - 0.5, dcir_delta.max() + 1.5, 1), color="#60a5fa")
    ax2.axvline(0, color="#111827", linewidth=1)
    ax2.set_xlabel("cycle_t - DCIR cycle used")
    ax2.set_ylabel("Count")
    ax2.set_title("DCIR cycle lag relative to sample cycle")
    ax2.grid(alpha=0.25, axis="y")

    summary = (
        f"Rows with DCIR: {len(df)} | Cells: {df['cell_key'].nunique()}\n"
        f"cycle_t == DCIR cycle: {int(dcir_exact.sum())}/{len(df)}\n"
        f"Rows with ECM measurement cycle: {int(ecm_nonnull.sum())}/{len(df)}\n"
        f"cycle_t == ECM cycle: {int(ecm_exact.sum())}/{int(ecm_nonnull.sum()) if ecm_nonnull.any() else 0}\n"
        f"ECM cycle == DCIR cycle: {int(ecm_dcir_exact.sum())}/{int(ecm_nonnull.sum()) if ecm_nonnull.any() else 0}"
    )
    ax1.text(
        0.99,
        0.02,
        summary,
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.92},
    )

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close(fig)
    print(f"[INFO] Saved cycle alignment plot: {out}")


if __name__ == "__main__":
    main()
