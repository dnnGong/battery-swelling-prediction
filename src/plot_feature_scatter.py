#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix


def build_out_path(base: Path, tag: str) -> Path:
    return base.with_name(f"{base.stem}__{tag}{base.suffix or '.png'}")


def pick_anchor_rows_fixed_T(df: pd.DataFrame, T: int, max_input_cycle: int) -> pd.DataFrame:
    out = []
    for _, g in df.groupby("cell_key"):
        g = g.sort_values("cycle_t")
        anchor = g[g["cycle_t"] <= max_input_cycle]
        if anchor.empty:
            continue
        anchor_row = anchor.iloc[-1].copy()

        tgt = g[g["cycle_t"] <= T]
        if tgt.empty:
            continue
        tgt_row = tgt.iloc[-1]
        anchor_row["target_abs"] = float(tgt_row["y_abs_thickness_t"])
        anchor_row["target_delta"] = float(tgt_row["y_delta_thickness_baseline_t"])
        anchor_row["target_cycle"] = int(tgt_row["cycle_t"])
        out.append(anchor_row)

    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out)


def pick_rows_future_delta_TK(df: pd.DataFrame, max_input_cycle: int) -> pd.DataFrame:
    sub = df[(df["cycle_t"] <= max_input_cycle) & (df["has_future_k"] == 1)].copy()
    if sub.empty:
        return sub
    sub["target_abs"] = sub["y_future_abs_thickness_tk"].astype(float)
    sub["target_delta"] = sub["y_future_delta_thickness_tk"].astype(float)
    sub["target_cycle"] = (sub["cycle_t"] + sub["future_k"]).astype(int)
    return sub


def pick_numeric_feature_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols


def select_top_features_by_label_corr(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    top_n: int,
    corr_method: str,
) -> pd.DataFrame:
    rows = []
    for c in feature_cols:
        s = df[[c, label_col]].dropna()
        if len(s) < 3 or s[c].nunique(dropna=True) <= 1:
            continue
        corr = s[c].corr(s[label_col], method=corr_method)
        if pd.isna(corr):
            continue
        rows.append({"feature": c, "corr": float(corr), "abs_corr": float(abs(corr))})

    if not rows:
        return pd.DataFrame(columns=["feature", "corr", "abs_corr"])
    rank = pd.DataFrame(rows).sort_values(["abs_corr", "feature"], ascending=[False, True]).reset_index(drop=True)
    return rank.head(top_n)


def save_feature_feature_scatter_matrix(
    df: pd.DataFrame,
    feat_cols: List[str],
    out_path: Path,
    max_points: int,
) -> None:
    sub = df[feat_cols].copy().dropna()
    if sub.empty:
        raise ValueError("No valid rows to draw feature-feature scatter matrix.")

    if len(sub) > max_points:
        sub = sub.sample(n=max_points, random_state=42)

    n = len(feat_cols)
    fig_w = max(8.0, n * 2.0)
    fig_h = max(8.0, n * 2.0)
    axes = scatter_matrix(
        sub,
        alpha=0.65,
        figsize=(fig_w, fig_h),
        diagonal="hist",
        marker="o",
        s=12,
        range_padding=0.05,
    )
    for ax_row in axes:
        for ax in np.atleast_1d(ax_row):
            ax.grid(alpha=0.2)

    plt.suptitle(f"Feature-Feature Scatter Matrix (N={len(sub)})", y=0.995)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_feature_label_scatter_grid(
    df: pd.DataFrame,
    label_col: str,
    rank_df: pd.DataFrame,
    out_path: Path,
    max_points: int,
) -> None:
    if rank_df.empty:
        raise ValueError("No ranked features available for feature-label scatter plot.")

    n = len(rank_df)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.8 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax in axes_flat[n:]:
        ax.axis("off")

    for ax, row in zip(axes_flat, rank_df.itertuples(index=False)):
        feat = row.feature
        corr = row.corr
        sub = df[[feat, label_col]].dropna()
        if len(sub) > max_points:
            sub = sub.sample(n=max_points, random_state=42)

        ax.scatter(sub[feat], sub[label_col], s=16, alpha=0.7, edgecolors="none")
        ax.set_xlabel(feat)
        ax.set_ylabel(label_col)
        ax.set_title(f"{feat}\ncorr={corr:.4f}, N={len(sub)}", fontsize=10)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Auto-generate feature-feature and feature-label scatter plots from feature_table.csv, "
            "with top features ranked by absolute correlation to the selected label."
        ),
        epilog=(
            "Example:\n"
            "  python src/plot_feature_scatter.py "
            "--table_csv ./data/ml/feature_table.csv "
            "--out_png ./data/ml/feature_scatter.png "
            "--target_mode fixed_T --label_mode absolute --T 100 --max_input_cycle 50 "
            "--group_tag HYCL --top_n 8 --corr_method spearman\n\n"
            "Outputs:\n"
            "  feature_scatter__feature_feature.png\n"
            "  feature_scatter__feature_label.png\n"
            "  feature_scatter__top_features.csv"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--table_csv", required=True, help="Input feature_table.csv path.")
    ap.add_argument("--out_png", required=True, help="Base output png path; actual outputs append suffixes.")
    ap.add_argument("--feature_prefix", default="feat_", help="Feature column prefix.")
    ap.add_argument("--target_mode", choices=["fixed_T", "future_delta_TK"], default="fixed_T")
    ap.add_argument("--label_mode", choices=["absolute", "delta"], default="absolute")
    ap.add_argument("--T", type=int, default=100, help="Target cycle for fixed_T mode.")
    ap.add_argument("--max_input_cycle", type=int, default=50, help="Maximum cycle used as model input.")
    ap.add_argument("--group_tag", default="", help="Optional group filter such as CL/FLC/HYCL.")
    ap.add_argument("--top_n", type=int, default=8, help="Top-N features to plot, ranked by |corr(feature, label)|.")
    ap.add_argument("--corr_method", choices=["pearson", "spearman", "kendall"], default="spearman")
    ap.add_argument("--max_points", type=int, default=1200, help="Max sampled points per plot for readability.")
    args = ap.parse_args()

    raw = pd.read_csv(args.table_csv)
    if raw.empty:
        raise ValueError("Input CSV is empty.")

    if args.target_mode == "fixed_T":
        data = pick_anchor_rows_fixed_T(raw, T=args.T, max_input_cycle=args.max_input_cycle)
    else:
        data = pick_rows_future_delta_TK(raw, max_input_cycle=args.max_input_cycle)

    if data.empty:
        raise ValueError("No rows after target-mode filtering.")

    if args.group_tag:
        data = data[data["group_tag"] == args.group_tag].copy()
        if data.empty:
            raise ValueError(f"No rows for group_tag={args.group_tag}")

    label_col = "target_abs" if args.label_mode == "absolute" else "target_delta"
    if label_col not in data.columns:
        raise ValueError(f"Label column not found: {label_col}")

    feat_cols = pick_numeric_feature_cols(data, args.feature_prefix)
    feat_cols = [c for c in feat_cols if data[c].nunique(dropna=True) > 1]
    if not feat_cols:
        raise ValueError(f"No usable numeric features found with prefix: {args.feature_prefix}")

    rank = select_top_features_by_label_corr(
        df=data,
        feature_cols=feat_cols,
        label_col=label_col,
        top_n=max(1, args.top_n),
        corr_method=args.corr_method,
    )
    if rank.empty:
        raise ValueError("No feature-label correlation could be computed.")

    top_feats = rank["feature"].tolist()
    out_base = Path(args.out_png)
    out_ff = build_out_path(out_base, "feature_feature")
    out_fl = build_out_path(out_base, "feature_label")
    out_rank = out_base.with_name(f"{out_base.stem}__top_features.csv")

    save_feature_feature_scatter_matrix(
        df=data,
        feat_cols=top_feats,
        out_path=out_ff,
        max_points=max(200, args.max_points),
    )
    save_feature_label_scatter_grid(
        df=data,
        label_col=label_col,
        rank_df=rank,
        out_path=out_fl,
        max_points=max(200, args.max_points),
    )
    rank.to_csv(out_rank, index=False)

    print(f"[INFO] Rows used: {len(data)}")
    print(f"[INFO] Label: {label_col}")
    print(f"[INFO] Top features ({len(top_feats)}): {top_feats}")
    print(f"[INFO] Saved feature-feature plot: {out_ff}")
    print(f"[INFO] Saved feature-label plot: {out_fl}")
    print(f"[INFO] Saved top-feature ranking: {out_rank}")


if __name__ == "__main__":
    main()
