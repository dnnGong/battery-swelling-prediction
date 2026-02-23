#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pick_numeric_feature_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table_csv", required=True, help="Path to feature_table.csv")
    ap.add_argument("--out_png", required=True, help="Output heatmap png path")
    ap.add_argument("--feature_prefix", default="feat_", help="Feature column prefix")
    ap.add_argument("--method", choices=["pearson", "spearman", "kendall"], default="pearson")
    ap.add_argument("--max_features", type=int, default=40, help="Keep top-N features by variance for readability")
    ap.add_argument("--annot", action="store_true", help="Show correlation values in cells")
    ap.add_argument("--include_targets", action="store_true", help="Also include target columns in matrix")
    args = ap.parse_args()

    df = pd.read_csv(args.table_csv)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    feat_cols = pick_numeric_feature_cols(df, args.feature_prefix)
    if not feat_cols:
        raise ValueError(f"No numeric columns found with prefix: {args.feature_prefix}")

    # Remove columns with <=1 unique value.
    feat_cols = [c for c in feat_cols if df[c].nunique(dropna=True) > 1]
    if not feat_cols:
        raise ValueError("No usable feature columns after removing constant columns.")

    # Keep top variance features for readable heatmap.
    if len(feat_cols) > args.max_features:
        var = df[feat_cols].var(numeric_only=True).sort_values(ascending=False)
        feat_cols = var.head(args.max_features).index.tolist()

    cols = list(feat_cols)
    if args.include_targets:
        target_candidates = [
            "y_abs_thickness_t",
            "y_delta_thickness_baseline_t",
            "y_future_abs_thickness_tk",
            "y_future_delta_thickness_tk",
        ]
        for c in target_candidates:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)

    sub = df[cols].copy()
    corr = sub.corr(method=args.method)

    # Plot
    n = len(corr.columns)
    figsize = (max(8, 0.55 * n), max(6, 0.55 * n))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title(f"Feature Correlation Matrix ({args.method})")

    if args.annot and n <= 30:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", rotation=90)

    plt.tight_layout()
    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=220)
    plt.close(fig)

    print(f"[INFO] Saved correlation heatmap: {out}")
    print(f"[INFO] Features used: {len(corr.columns)}")


if __name__ == "__main__":
    main()
