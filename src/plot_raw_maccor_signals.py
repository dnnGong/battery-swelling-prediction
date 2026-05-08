#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parse_raw_maccor import parse_one_raw_file


SIGNALS: List[Tuple[str, str]] = [
    ("capacity_ahr", "Capacity (Ah)"),
    ("energy_whr", "Energy (Wh)"),
    ("current_a", "Current (A)"),
    ("voltage_v", "Voltage (V)"),
]


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def choose_x_axis(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    if "test_time_s" in df.columns:
        x = pd.to_numeric(df["test_time_s"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(x).any():
            return x / 3600.0, "Test Time (h)"
    if "rec_num" in df.columns:
        x = pd.to_numeric(df["rec_num"], errors="coerce").to_numpy(dtype=float)
        return x, "Record Number"
    return np.arange(len(df), dtype=float), "Row Index"


def downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.copy()
    idx = np.linspace(0, len(df) - 1, max_points, dtype=int)
    return df.iloc[idx].copy()


def plot_one_file(df: pd.DataFrame, out_png: Path, title: str, max_points: int) -> Dict[str, float]:
    work = downsample(df, max_points=max_points)
    x, xlabel = choose_x_axis(work)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()

    summary: Dict[str, float] = {"n_rows": float(len(df))}
    for ax, (col, label) in zip(axes, SIGNALS):
        y = pd.to_numeric(work.get(col), errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.any():
            ax.plot(x[valid], y[valid], linewidth=1.0, color="#2563eb")
            summary[f"{col}_min"] = float(np.nanmin(y[valid]))
            summary[f"{col}_max"] = float(np.nanmax(y[valid]))
            summary[f"{col}_last"] = float(y[valid][-1])
        else:
            summary[f"{col}_min"] = math.nan
            summary[f"{col}_max"] = math.nan
            summary[f"{col}_last"] = math.nan
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_overlay(rows: List[pd.DataFrame], labels: List[str], out_png: Path, max_points: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    axes = axes.ravel()

    for df, label in zip(rows, labels):
        work = downsample(df, max_points=max_points)
        x, xlabel = choose_x_axis(work)
        for ax, (col, signal_label) in zip(axes, SIGNALS):
            y = pd.to_numeric(work.get(col), errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.any():
                ax.plot(x[valid], y[valid], linewidth=0.8, alpha=0.65, label=label)
            ax.set_title(signal_label)
            ax.set_xlabel(xlabel)
            ax.grid(alpha=0.25)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        uniq = {}
        for h, l in zip(handles, legend_labels):
            uniq.setdefault(l, h)
        fig.legend(
            uniq.values(),
            uniq.keys(),
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=8,
            frameon=False,
        )
    fig.suptitle("HYCL Raw Data Overlay", fontsize=13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_signal_overlays(rows: List[pd.DataFrame], labels: List[str], out_dir: Path, max_points: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for col, signal_label in SIGNALS:
        fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
        xlabel = "Index"
        for df, label in zip(rows, labels):
            work = downsample(df, max_points=max_points)
            x, xlabel = choose_x_axis(work)
            y = pd.to_numeric(work.get(col), errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.any():
                ax.plot(x[valid], y[valid], linewidth=0.9, alpha=0.7, label=label)

        ax.set_title(f"{signal_label} Overlay by File")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            uniq = {}
            for h, l in zip(handles, legend_labels):
                uniq.setdefault(l, h)
            ax.legend(
                uniq.values(),
                uniq.keys(),
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=8,
                frameon=False,
            )
        out_png = out_dir / f"{sanitize_name(col)}_overlay.png"
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)


def plot_range_summary(summary_df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()

    x = np.arange(len(summary_df))
    labels = summary_df["label"].tolist()
    for ax, (col, signal_label) in zip(axes, SIGNALS):
        mins = pd.to_numeric(summary_df[f"{col}_min"], errors="coerce")
        maxs = pd.to_numeric(summary_df[f"{col}_max"], errors="coerce")
        mids = pd.to_numeric(summary_df[f"{col}_last"], errors="coerce")
        valid = mins.notna() & maxs.notna()
        ax.vlines(x[valid], mins[valid], maxs[valid], color="#93c5fd", linewidth=3, alpha=0.9)
        ax.scatter(x[valid], mids[valid], color="#1d4ed8", s=22, zorder=3)
        ax.set_title(f"{signal_label} Range by File")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("HYCL Raw Data Signal Ranges", fontsize=13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualize raw Maccor signals (Capacity, Energy, Current, Voltage) for a folder of text exports."
    )
    ap.add_argument("--raw_dir", required=True, help="Directory containing raw Maccor exports.")
    ap.add_argument("--out_dir", required=True, help="Output directory for plots and summaries.")
    ap.add_argument("--max_points_per_file", type=int, default=5000, help="Downsample per-file plots to at most this many rows.")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    per_file_dir = plots_dir / "per_file"

    files = [p for p in sorted(raw_dir.iterdir()) if p.is_file() and not p.name.startswith(".")]
    if not files:
        raise ValueError(f"No raw files found under: {raw_dir}")

    parsed_rows: List[pd.DataFrame] = []
    file_summaries: List[Dict[str, float]] = []
    labels: List[str] = []
    bad_files: List[Tuple[str, str]] = []

    for path in files:
        try:
            df = parse_one_raw_file(path)
            if df.empty:
                bad_files.append((path.name, "empty after parsing"))
                continue

            parsed_rows.append(df)
            serial = str(df.get("serial_norm", pd.Series(["UNKNOWN"])).iloc[0])
            phase = str(df.get("phase", pd.Series([""])).iloc[0] or "")
            label = f"{serial}-{phase}" if phase else serial
            labels.append(label)

            title = f"{path.name}\nserial={serial} | phase={phase or 'NA'} | rows={len(df):,}"
            out_png = per_file_dir / f"{sanitize_name(path.stem)}.png"
            summary = plot_one_file(df, out_png=out_png, title=title, max_points=args.max_points_per_file)
            summary.update(
                {
                    "file_name": path.name,
                    "label": label,
                    "serial_norm": serial,
                    "phase": phase,
                }
            )
            file_summaries.append(summary)
        except Exception as exc:
            bad_files.append((path.name, str(exc)))

    if not parsed_rows:
        raise ValueError("No valid raw files were parsed.")

    row_df = pd.concat(parsed_rows, ignore_index=True)
    summary_df = pd.DataFrame(file_summaries).sort_values(["serial_norm", "phase", "file_name"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    row_df.to_csv(out_dir / "raw_rows.csv", index=False)
    summary_df.to_csv(out_dir / "file_signal_summary.csv", index=False)

    plot_overlay(parsed_rows, labels, out_png=plots_dir / "overlay_all_files.png", max_points=1500)
    plot_signal_overlays(parsed_rows, labels, out_dir=plots_dir / "overlay_by_signal", max_points=1500)
    plot_range_summary(summary_df, out_png=plots_dir / "signal_ranges_by_file.png")

    report_lines = [
        f"raw_dir: {raw_dir}",
        f"parsed_files: {len(parsed_rows)}",
        f"bad_files: {len(bad_files)}",
        f"total_rows: {len(row_df)}",
        f"output_dir: {out_dir}",
    ]
    if bad_files:
        report_lines.append("")
        report_lines.append("bad_files_detail:")
        report_lines.extend([f"- {name}: {err}" for name, err in bad_files])
    (out_dir / "run_summary.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[INFO] Parsed files: {len(parsed_rows)} / {len(files)}")
    print(f"[INFO] Bad files: {len(bad_files)}")
    print(f"[INFO] Total rows: {len(row_df)}")
    print(f"[INFO] Saved row CSV: {out_dir / 'raw_rows.csv'}")
    print(f"[INFO] Saved file summary: {out_dir / 'file_signal_summary.csv'}")
    print(f"[INFO] Saved overlay plot: {plots_dir / 'overlay_all_files.png'}")
    print(f"[INFO] Saved signal overlay plots under: {plots_dir / 'overlay_by_signal'}")
    print(f"[INFO] Saved range plot: {plots_dir / 'signal_ranges_by_file.png'}")
    print(f"[INFO] Saved per-file plots under: {per_file_dir}")


if __name__ == "__main__":
    main()
