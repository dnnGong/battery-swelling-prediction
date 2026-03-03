#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform


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
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return cols


def feature_clustering_plot(
    df: pd.DataFrame,
    feat_cols: List[str],
    corr_method: str,
    linkage_method: str,
    n_feature_clusters: int,
    out_png: Path,
    out_csv: Path,
) -> None:
    corr = df[feat_cols].corr(method=corr_method).fillna(0.0)
    dist = 1.0 - np.abs(corr.values)
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=linkage_method)
    leaf_order = dendrogram(Z, no_plot=True)["leaves"]
    ordered_cols = [corr.columns[i] for i in leaf_order]
    corr_ord = corr.loc[ordered_cols, ordered_cols]

    feature_cluster_labels = fcluster(Z, t=max(1, n_feature_clusters), criterion="maxclust")
    cluster_df = pd.DataFrame(
        {
            "feature": corr.columns.tolist(),
            "cluster_id": feature_cluster_labels.tolist(),
        }
    ).sort_values(["cluster_id", "feature"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cluster_df.to_csv(out_csv, index=False)

    n = len(ordered_cols)
    fig = plt.figure(figsize=(max(10, n * 0.45), max(7, n * 0.45)))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.1, 4.0], hspace=0.05)

    ax_d = fig.add_subplot(gs[0, 0])
    dendrogram(
        Z,
        labels=corr.columns.tolist(),
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax_d,
        color_threshold=None,
    )
    ax_d.set_title(f"Feature Hierarchical Clustering ({linkage_method}, corr={corr_method})")
    ax_d.set_ylabel("Distance")

    ax_h = fig.add_subplot(gs[1, 0])
    im = ax_h.imshow(corr_ord.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax_h.set_xticks(np.arange(n))
    ax_h.set_yticks(np.arange(n))
    ax_h.set_xticklabels(ordered_cols, rotation=90, fontsize=7)
    ax_h.set_yticklabels(ordered_cols, fontsize=7)
    ax_h.set_title("Ordered Feature Correlation Matrix")
    cbar = fig.colorbar(im, ax=ax_h, fraction=0.02, pad=0.02)
    cbar.set_label("Correlation")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close(fig)


def sample_clustering_plot(
    df: pd.DataFrame,
    feat_cols: List[str],
    n_clusters: int,
    seed: int,
    out_png: Path,
    out_csv: Path,
) -> None:
    _ = seed  # reserved for reproducibility extensions
    work = df.copy()
    X = work[feat_cols].copy()
    med = X.median(numeric_only=True)
    X = X.fillna(med).fillna(0.0)
    Xv = X.to_numpy(dtype=float, copy=True)

    mu = np.mean(Xv, axis=0, keepdims=True)
    sd = np.std(Xv, axis=0, keepdims=True)
    sd = np.where(sd == 0.0, 1.0, sd)
    Xs = (Xv - mu) / sd

    Zs = linkage(Xs, method="ward")
    labels = fcluster(Zs, t=max(2, n_clusters), criterion="maxclust") - 1

    # PCA from SVD to avoid heavy runtime deps.
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    XY = U[:, :2] * S[:2]
    var = np.var(Xs, axis=0, ddof=0).sum()
    evr = np.array([0.0, 0.0], dtype=float)
    if var > 0 and len(S) >= 2:
        evr[0] = (S[0] ** 2 / max(1, Xs.shape[0])) / var
        evr[1] = (S[1] ** 2 / max(1, Xs.shape[0])) / var

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        XY[:, 0],
        XY[:, 1],
        c=labels,
        cmap="tab10",
        s=22,
        alpha=0.8,
        edgecolors="none",
    )
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% var)")
    ax.set_title(f"Sample Clustering via Hierarchical (k={max(2, n_clusters)}) + PCA")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cluster ID")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close(fig)

    out = work.copy()
    out["sample_cluster_id"] = labels
    group_col = "group_tag" if "group_tag" in out.columns else None
    if group_col:
        summary = (
            out.groupby(["sample_cluster_id", group_col], dropna=False)
            .size()
            .rename("n_rows")
            .reset_index()
            .sort_values(["sample_cluster_id", "n_rows"], ascending=[True, False])
        )
    else:
        summary = (
            out.groupby(["sample_cluster_id"], dropna=False)
            .size()
            .rename("n_rows")
            .reset_index()
            .sort_values(["sample_cluster_id"])
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Feature clustering utility for non-raw ML feature tables. "
            "Outputs feature hierarchical clustering and sample clustering visualizations."
        ),
        epilog=(
            "Example:\n"
            "  python src/plot_feature_clustering.py "
            "--table_csv ./data/ml/feature_table.csv "
            "--out_png ./data/ml/feature_cluster.png "
            "--sample_mode future_delta_TK --max_input_cycle 50 --group_tag HYCL "
            "--max_features 30 --n_feature_clusters 6 --n_sample_clusters 5\n\n"
            "Outputs:\n"
            "  feature_cluster__feature_clustermap.png\n"
            "  feature_cluster__sample_pca_clusters.png\n"
            "  feature_cluster__feature_cluster_labels.csv\n"
            "  feature_cluster__sample_cluster_summary.csv"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--table_csv", required=True, help="Input feature table CSV.")
    ap.add_argument("--out_png", required=True, help="Base output PNG path.")
    ap.add_argument("--feature_prefix", default="feat_", help="Feature column prefix.")
    ap.add_argument("--sample_mode", choices=["raw", "fixed_T", "future_delta_TK"], default="raw")
    ap.add_argument("--T", type=int, default=100, help="Target cycle for fixed_T sampling mode.")
    ap.add_argument("--max_input_cycle", type=int, default=50, help="Max input cycle for fixed_T/future sampling.")
    ap.add_argument("--group_tag", default="", help="Optional group filter: CL/FLC/HYCL.")
    ap.add_argument("--max_features", type=int, default=35, help="Keep top-N features by variance for readability.")
    ap.add_argument("--corr_method", choices=["pearson", "spearman", "kendall"], default="spearman")
    ap.add_argument("--linkage_method", choices=["average", "complete", "single", "ward"], default="average")
    ap.add_argument("--n_feature_clusters", type=int, default=6, help="Cluster count for feature label CSV.")
    ap.add_argument("--n_sample_clusters", type=int, default=5, help="KMeans cluster count for samples.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = ap.parse_args()

    raw = pd.read_csv(args.table_csv)
    if raw.empty:
        raise ValueError("Input CSV is empty.")

    if args.sample_mode == "fixed_T":
        data = pick_anchor_rows_fixed_T(raw, T=args.T, max_input_cycle=args.max_input_cycle)
    elif args.sample_mode == "future_delta_TK":
        data = pick_rows_future_delta_TK(raw, max_input_cycle=args.max_input_cycle)
    else:
        data = raw.copy()

    if data.empty:
        raise ValueError("No rows after sample_mode filtering.")

    if args.group_tag:
        data = data[data["group_tag"] == args.group_tag].copy()
        if data.empty:
            raise ValueError(f"No rows for group_tag={args.group_tag}")

    feat_cols = pick_numeric_feature_cols(data, args.feature_prefix)
    if not feat_cols:
        raise ValueError(f"No usable numeric features with prefix '{args.feature_prefix}'.")

    if len(feat_cols) > args.max_features:
        var = data[feat_cols].var(numeric_only=True).sort_values(ascending=False)
        feat_cols = var.head(args.max_features).index.tolist()

    out_base = Path(args.out_png)
    out_feature_png = build_out_path(out_base, "feature_clustermap")
    out_sample_png = build_out_path(out_base, "sample_pca_clusters")
    out_feature_csv = out_base.with_name(f"{out_base.stem}__feature_cluster_labels.csv")
    out_sample_csv = out_base.with_name(f"{out_base.stem}__sample_cluster_summary.csv")

    feature_clustering_plot(
        df=data,
        feat_cols=feat_cols,
        corr_method=args.corr_method,
        linkage_method=args.linkage_method,
        n_feature_clusters=args.n_feature_clusters,
        out_png=out_feature_png,
        out_csv=out_feature_csv,
    )
    sample_clustering_plot(
        df=data,
        feat_cols=feat_cols,
        n_clusters=args.n_sample_clusters,
        seed=args.seed,
        out_png=out_sample_png,
        out_csv=out_sample_csv,
    )

    print(f"[INFO] Rows used: {len(data)}")
    print(f"[INFO] Feature count used: {len(feat_cols)}")
    print(f"[INFO] Saved feature clustering plot: {out_feature_png}")
    print(f"[INFO] Saved sample clustering plot: {out_sample_png}")
    print(f"[INFO] Saved feature cluster labels: {out_feature_csv}")
    print(f"[INFO] Saved sample cluster summary: {out_sample_csv}")


if __name__ == "__main__":
    main()
