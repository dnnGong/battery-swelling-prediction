#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_out_path(base: Path, tag: str) -> Path:
    return base.with_name(f"{base.stem}__{tag}{base.suffix or '.png'}")


def plot_one(ax: plt.Axes, sub: pd.DataFrame, title: str) -> None:
    y_true = sub["y_true"].to_numpy(dtype=float)
    y_pred = sub["y_pred"].to_numpy(dtype=float)
    ax.scatter(y_true, y_pred, s=36, alpha=0.8, edgecolors="none")

    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    pad = max((hi - lo) * 0.05, 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1.2, color="gray")

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ax.set_title(f"{title}\nN={len(sub)} RMSE={rmse:.4g} MAE={mae:.4g}", fontsize=10)
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    ax.grid(alpha=0.25)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)


def save_single_plot(sub: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    plot_one(ax, sub, title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def save_grid(df: pd.DataFrame, out_path: Path, split_by: str) -> None:
    groups = list(df.groupby(split_by))
    n = len(groups)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.6 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax in axes_flat[n:]:
        ax.axis("off")

    for ax, (tag, sub) in zip(axes_flat, groups):
        plot_one(ax, sub, f"{split_by}={tag}")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True, help="Path to predictions__*.csv")
    ap.add_argument("--out_png", required=True, help="Base output png path")
    ap.add_argument(
        "--mode",
        choices=["combined", "by_model", "by_group", "all"],
        default="all",
        help="Which scatter plot(s) to save. Default: all",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)
    if df.empty:
        raise ValueError("Predictions CSV is empty.")
    required = {"model", "group_tag", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions CSV missing required columns: {sorted(missing)}")

    out_base = Path(args.out_png)

    if args.mode in {"combined", "all"}:
        save_single_plot(df, build_out_path(out_base, "combined"), "All predictions")
        print(f"[INFO] Saved combined scatter: {build_out_path(out_base, 'combined')}")

    if args.mode in {"by_model", "all"}:
        save_grid(df, build_out_path(out_base, "by_model"), "model")
        print(f"[INFO] Saved model-split scatter: {build_out_path(out_base, 'by_model')}")

    if args.mode in {"by_group", "all"}:
        save_grid(df, build_out_path(out_base, "by_group"), "group_tag")
        print(f"[INFO] Saved group-split scatter: {build_out_path(out_base, 'by_group')}")


if __name__ == "__main__":
    main()
