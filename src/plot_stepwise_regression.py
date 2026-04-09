#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_out_path(base: Path, tag: str) -> Path:
    return base.with_name(f"{base.stem}__{tag}{base.suffix or '.png'}")


def shorten_feature_name(name: str) -> str:
    text = str(name)
    if text.startswith("feat_"):
        text = text[5:]
    return text.replace("_", "\n")


def sorted_panels(df: pd.DataFrame) -> List[Tuple[Tuple[str, str], pd.DataFrame]]:
    keys = ["group_tag", "model"]
    if not set(keys).issubset(df.columns):
        raise ValueError(f"Stepwise trace must include columns: {keys}")
    panels = []
    for key, sub in df.groupby(keys, dropna=False):
        sub = sub.sort_values("step").reset_index(drop=True)
        panels.append((key, sub))
    panels.sort(key=lambda x: (str(x[0][0]), str(x[0][1])))
    return panels


def plot_path(ax: plt.Axes, sub: pd.DataFrame, title: str) -> None:
    steps = sub["step"].to_numpy(dtype=int)
    cv_mae = sub["cv_mae"].to_numpy(dtype=float)
    ax.plot(steps, cv_mae, marker="o", linewidth=2.0, color="#1f77b4")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("CV MAE")
    ax.set_xticks(steps)
    ax.grid(alpha=0.25)

    for _, row in sub.iterrows():
        ax.annotate(
            shorten_feature_name(row["feature_name"]),
            (int(row["step"]), float(row["cv_mae"])),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )


def save_path_grid(df: pd.DataFrame, out_path: Path) -> None:
    panels = sorted_panels(df)
    n = len(panels)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.8 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax in axes_flat[n:]:
        ax.axis("off")

    for ax, ((group_tag, model), sub) in zip(axes_flat, panels):
        plot_path(ax, sub, f"{group_tag} | {model}")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_improvement(ax: plt.Axes, sub: pd.DataFrame, title: str) -> None:
    labels = [shorten_feature_name(x) for x in sub["feature_name"].tolist()]
    improvements = sub["improvement"].to_numpy(dtype=float)
    y = np.arange(len(labels))
    ax.barh(y, improvements, color="#2ca02c", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("CV MAE Improvement")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, axis="x")
    ax.invert_yaxis()

    xmax = float(max(np.max(improvements), 1e-9))
    ax.set_xlim(0.0, xmax * 1.15)
    for yi, val in zip(y, improvements):
        ax.text(float(val) + xmax * 0.03, yi, f"{val:.4g}", va="center", fontsize=8)


def save_improvement_grid(df: pd.DataFrame, out_path: Path) -> None:
    panels = sorted_panels(df)
    n = len(panels)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.6 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax in axes_flat[n:]:
        ax.axis("off")

    for ax, ((group_tag, model), sub) in zip(axes_flat, panels):
        plot_improvement(ax, sub, f"{group_tag} | {model}")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def save_entry_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    panels = sorted_panels(df)
    frames = []
    for (group_tag, model), sub in panels:
        tag = f"{group_tag} | {model}"
        one = sub[["feature_name", "step"]].copy()
        one["panel"] = tag
        frames.append(one)
    mat = pd.concat(frames, ignore_index=True).pivot(index="feature_name", columns="panel", values="step")
    mat = mat.sort_values(by=list(mat.columns), key=lambda s: s.fillna(999), ascending=True)

    fig_w = max(7.0, 1.4 * len(mat.columns) + 2.5)
    fig_h = max(5.0, 0.45 * len(mat.index) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    arr = mat.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(arr)
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="#f2f2f2")
    im = ax.imshow(masked, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels([shorten_feature_name(x).replace("\n", " ") for x in mat.index], fontsize=8)
    ax.set_title("Stepwise Feature Entry Order")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not np.isnan(arr[i, j]):
                ax.text(j, i, f"{int(arr[i, j])}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Entry Step", rotation=90)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot stepwise regression trace visualizations from stepwise_trace__*.csv.",
        epilog=(
            "Example:\n"
            "  python src/plot_stepwise_regression.py "
            "--trace_csv ./data/ml3/.../stepwise_trace__fixed_T__absolute__fixedT_100__stepwise_v1.csv "
            "--out_png ./data/ml3/.../stepwise.png --mode all"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--trace_csv", required=True, help="Input stepwise_trace__*.csv path.")
    ap.add_argument("--out_png", required=True, help="Base output PNG path.")
    ap.add_argument(
        "--mode",
        choices=["path", "improvement", "heatmap", "all"],
        default="all",
        help=(
            "Which plot(s) to save:\n"
            "  path        : CV-MAE vs step with feature annotations\n"
            "  improvement : horizontal bars of per-step improvement\n"
            "  heatmap     : feature entry-order matrix across group/model panels\n"
            "  all         : save all three views"
        ),
    )
    args = ap.parse_args()

    df = pd.read_csv(args.trace_csv)
    if df.empty:
        raise ValueError("Stepwise trace CSV is empty.")

    required = {"group_tag", "model", "step", "feature_name", "cv_mae", "improvement"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Stepwise trace CSV missing required columns: {sorted(missing)}")

    out_base = Path(args.out_png)

    if args.mode in {"path", "all"}:
        out_path = build_out_path(out_base, "path")
        save_path_grid(df, out_path)
        print(f"[INFO] Saved stepwise path plot: {out_path}")

    if args.mode in {"improvement", "all"}:
        out_path = build_out_path(out_base, "improvement")
        save_improvement_grid(df, out_path)
        print(f"[INFO] Saved stepwise improvement plot: {out_path}")

    if args.mode in {"heatmap", "all"}:
        out_path = build_out_path(out_base, "heatmap")
        save_entry_heatmap(df, out_path)
        print(f"[INFO] Saved stepwise heatmap: {out_path}")


if __name__ == "__main__":
    main()
