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


def build_out_path(base: Path, tag: str) -> Path:
    return base.with_name(f"{base.stem}__{tag}{base.suffix or '.png'}")


def parse_columns(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def plot_corr(corr: pd.DataFrame, out_path: Path, method: str, annot: bool, title: str) -> None:
    n = len(corr.columns)
    figsize = (max(8, 0.55 * n), max(6, 0.55 * n))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title(title)

    if annot and n <= 30:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", rotation=90)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Plot correlation heatmaps from feature_table.csv for feature-only and/or "
            "feature-plus-target analysis."
        ),
        epilog=(
            "Example:\n"
            "  python src/plot_feature_corr.py --table_csv ./data/ml/feature_table.csv "
            "--out_png ./data/ml/feature_corr.png --method pearson --max_features 40 --annot\n\n"
            "Outputs with --mode both:\n"
            "  feature_corr__features.png\n"
            "  feature_corr__features_targets.png"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--table_csv", required=True, help="Input feature_table.csv path.")
    ap.add_argument("--out_png", required=True, help="Base output png path; actual outputs append suffixes like __features.png")
    ap.add_argument("--feature_prefix", default="feat_", help="Feature column prefix to include in the matrix.")
    ap.add_argument("--columns", default="", help="Optional comma-separated columns to plot in the given order.")
    ap.add_argument("--group_tag", default="", help="Optional group_tag filter, e.g. HYCL.")
    ap.add_argument("--max_cycle", type=float, default=None, help="Optional filter: keep rows with cycle_t <= max_cycle.")
    ap.add_argument("--method", choices=["pearson", "spearman", "kendall"], default="pearson", help="Correlation method.")
    ap.add_argument("--max_features", type=int, default=40, help="Keep top-N features by variance for readability.")
    ap.add_argument("--annot", action="store_true", help="Show correlation values in cells for small matrices.")
    ap.add_argument(
        "--mode",
        choices=["both", "features", "features_targets"],
        default="both",
        help=(
            "Which heatmap(s) to save:\n"
            "  both             : save feature-only and feature+target heatmaps\n"
            "  features         : save feature-only heatmap\n"
            "  features_targets : save feature+target heatmap"
        ),
    )
    ap.add_argument(
        "--include_targets",
        action="store_true",
        help="Legacy alias for --mode features_targets",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.table_csv)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    target_candidates = [
        "y_abs_thickness_t",
        "y_delta_thickness_baseline_t",
        "y_future_abs_thickness_tk",
        "y_future_delta_thickness_tk",
    ]

    if args.group_tag:
        if "group_tag" not in df.columns:
            raise ValueError("--group_tag was provided, but input table has no group_tag column.")
        df = df[df["group_tag"] == args.group_tag].copy()
    if args.max_cycle is not None:
        if "cycle_t" not in df.columns:
            raise ValueError("--max_cycle was provided, but input table has no cycle_t column.")
        df = df[pd.to_numeric(df["cycle_t"], errors="coerce") <= args.max_cycle].copy()
    if df.empty:
        raise ValueError("No rows left after filtering.")

    if args.columns:
        cols = parse_columns(args.columns)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing requested columns: {missing}")
        non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise ValueError(f"Requested columns are not numeric: {non_numeric}")
        cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
        if not cols:
            raise ValueError("No usable requested columns after removing constant columns.")
        feat_cols = [c for c in cols if c.startswith(args.feature_prefix)]
        target_cols = [c for c in cols if c in target_candidates]
    else:
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

        target_cols = [c for c in target_candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    mode = "features_targets" if args.include_targets else args.mode
    out_base = Path(args.out_png)

    if mode in {"both", "features"}:
        if not feat_cols:
            raise ValueError("No feature columns available for feature-only heatmap.")
        corr_feat = df[feat_cols].copy().corr(method=args.method)
        out_feat = build_out_path(out_base, "features")
        plot_corr(
            corr=corr_feat,
            out_path=out_feat,
            method=args.method,
            annot=args.annot,
            title=f"Feature Correlation Matrix ({args.method})",
        )
        print(f"[INFO] Saved feature-only heatmap: {out_feat}")
        print(f"[INFO] Feature-only columns used: {len(corr_feat.columns)}")

    if mode in {"both", "features_targets"}:
        cols = list(feat_cols) + target_cols
        if args.columns:
            cols = parse_columns(args.columns)
            cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1]
        if not cols:
            raise ValueError("No columns available for correlation heatmap.")
        if not target_cols and not args.columns:
            raise ValueError("No numeric target columns found for feature+target correlation heatmap.")
        corr_all = df[cols].copy().corr(method=args.method)
        out_all = build_out_path(out_base, "features_targets")
        plot_corr(
            corr=corr_all,
            out_path=out_all,
            method=args.method,
            annot=args.annot,
            title=f"Feature + Target Correlation Matrix ({args.method})",
        )
        print(f"[INFO] Saved feature+target heatmap: {out_all}")
        print(f"[INFO] Feature+target columns used: {len(corr_all.columns)}")


if __name__ == "__main__":
    main()
