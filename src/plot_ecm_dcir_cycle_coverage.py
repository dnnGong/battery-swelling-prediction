#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def parse_cycle_list(val) -> List[int]:
    if pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except Exception:
            continue
    return sorted(set(out))


def build_long_df(overview: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in overview.iterrows():
        cell_key = r["cell_key"]
        short_cell = str(r["serial"])
        for cyc in parse_cycle_list(r.get("ecm_cycles")):
            rows.append(
                {
                    "cell_key": cell_key,
                    "serial": short_cell,
                    "source": "ECM",
                    "cycle": cyc,
                }
            )
        for cyc in parse_cycle_list(r.get("dcir_cycles")):
            rows.append(
                {
                    "cell_key": cell_key,
                    "serial": short_cell,
                    "source": "DCIR",
                    "cycle": cyc,
                }
            )
    return pd.DataFrame(rows)


def plot_timeline(long_df: pd.DataFrame, overview: pd.DataFrame, out_png: Path, title: str) -> None:
    order = overview.sort_values(["exact_match_count", "dcir_cycle_count", "ecm_cycle_count", "serial"], ascending=[False, False, False, True])["cell_key"].tolist()
    y_map = {k: i for i, k in enumerate(order)}

    fig_h = max(6, min(24, 0.22 * max(len(order), 1)))
    fig, ax = plt.subplots(figsize=(14, fig_h))

    for src, color, marker, alpha in [
        ("DCIR", "#D95F02", "x", 0.8),
        ("ECM", "#1F78B4", "o", 0.95),
    ]:
        sub = long_df[long_df["source"] == src].copy()
        if sub.empty:
            continue
        sub["y"] = sub["cell_key"].map(y_map)
        ax.scatter(
            sub["cycle"],
            sub["y"],
            label=src,
            s=24 if src == "ECM" else 20,
            c=color,
            marker=marker,
            alpha=alpha,
            linewidths=1.0 if src == "DCIR" else 0.0,
        )

    ax.set_title(title)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Cell")
    ax.set_yticks(list(range(len(order))))
    ax.set_yticklabels([x.split("::")[-1] for x in order], fontsize=7)
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cycle_counts(long_df: pd.DataFrame, out_png: Path, title: str) -> None:
    if long_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No cycle records to plot", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    counts = (
        long_df.groupby(["cycle", "source"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    for src in ["ECM", "DCIR"]:
        if src not in counts.columns:
            counts[src] = 0
    counts = counts[["ECM", "DCIR"]]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = counts.index.to_list()
    ax.bar(x, counts["DCIR"], width=6, color="#FDB863", alpha=0.7, label="DCIR")
    ax.bar(x, counts["ECM"], width=3, color="#5E3C99", alpha=0.9, label="ECM")
    ax.set_title(title)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Count of cells with measurement")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize ECM/DCIR cycle coverage from alignment overview CSV.")
    ap.add_argument("--overview_csv", required=True, help="Alignment overview CSV from check_ecm_dcir_alignment.py")
    ap.add_argument("--out_dir", required=True, help="Directory to save plots and long-form CSV")
    ap.add_argument("--title_prefix", default="ECM vs DCIR Cycle Coverage", help="Title prefix for output figures")
    args = ap.parse_args()

    overview_csv = Path(args.overview_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overview = pd.read_csv(overview_csv)
    long_df = build_long_df(overview)

    stem = overview_csv.stem.replace("__overview", "")
    long_csv = out_dir / f"{stem}__long.csv"
    timeline_png = out_dir / f"{stem}__timeline.png"
    counts_png = out_dir / f"{stem}__cycle_counts.png"

    long_df.to_csv(long_csv, index=False)
    plot_timeline(long_df, overview, timeline_png, f"{args.title_prefix}: per-cell timeline")
    plot_cycle_counts(long_df, counts_png, f"{args.title_prefix}: aggregate cycle counts")

    print(f"[INFO] cells={len(overview)}")
    print(f"[INFO] long_rows={len(long_df)}")
    print(f"[INFO] saved long csv: {long_csv}")
    print(f"[INFO] saved timeline: {timeline_png}")
    print(f"[INFO] saved cycle-count plot: {counts_png}")


if __name__ == "__main__":
    main()
