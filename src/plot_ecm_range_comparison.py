#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MATCHED5: List[Tuple[str, str, str, str | None]] = [
    ("R0", "prior_R0_lb", "prior_R0_ub", "feat_dev_r0_abs_proxy_ohm"),
    ("Rsei", "prior_Rsei_lb", "prior_Rsei_ub", "feat_dev_td_Rsei_ohm"),
    ("Rct", "prior_Rct_lb", "prior_Rct_ub", "feat_dev_td_Rct_ohm"),
    ("Rw1", "prior_Rw1_lb", "prior_Rw1_ub", "feat_dev_td_Rw1_ohm"),
    ("Rw2", "prior_Rw2_lb", "prior_Rw2_ub", "feat_dev_td_Rw2_ohm"),
]


def build_range_tables(priors: pd.DataFrame, train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    freq_rows = []
    time_rows = []
    for label, lb_col, ub_col, td_col in MATCHED5:
        freq_lb = pd.to_numeric(priors.get(lb_col), errors="coerce")
        freq_ub = pd.to_numeric(priors.get(ub_col), errors="coerce")
        freq_rows.append(
            {
                "parameter": label,
                "lb": float(freq_lb.min()) if freq_lb.notna().any() else np.nan,
                "ub": float(freq_ub.max()) if freq_ub.notna().any() else np.nan,
                "status": "ok" if freq_lb.notna().any() and freq_ub.notna().any() else "missing",
            }
        )

        if td_col is None:
            time_rows.append({"parameter": label, "lb": np.nan, "ub": np.nan, "status": "not_fitted_in_current_impl"})
        else:
            td_series = train_df[td_col] if td_col in train_df.columns else pd.Series(dtype=float)
            td_vals = pd.to_numeric(td_series, errors="coerce")
            if td_vals.notna().any():
                time_rows.append(
                    {
                        "parameter": label,
                        "lb": float(td_vals.min()),
                        "ub": float(td_vals.max()),
                        "status": "ok",
                    }
                )
            else:
                status = "no_valid_fit"
                if "feat_dev_td_fit_status" in train_df.columns and train_df["feat_dev_td_fit_status"].notna().any():
                    status = str(train_df["feat_dev_td_fit_status"].dropna().iloc[0])
                time_rows.append({"parameter": label, "lb": np.nan, "ub": np.nan, "status": status})

    return pd.DataFrame(freq_rows), pd.DataFrame(time_rows)


def _plot_range_df(df: pd.DataFrame, title: str, out_path: Path, color: str, xlim: tuple[float, float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(df))
    for i, row in df.iterrows():
        lb = row["lb"]
        ub = row["ub"]
        if pd.notna(lb) and pd.notna(ub):
            left = min(lb, ub)
            width = abs(ub - lb)
            if width == 0:
                width = 1e-12
            ax.barh(i, width, left=left, height=0.55, color=color, alpha=0.8)
            ax.text(ub, i, f" [{lb:.4g}, {ub:.4g}]", va="center", ha="left", fontsize=9)
        else:
            ax.text(0.02, i, f'N/A ({row["status"]})', va="center", ha="left", fontsize=9, transform=ax.get_yaxis_transform())
    ax.set_yticks(y)
    ax.set_yticklabels(df["parameter"])
    ax.set_title(title)
    ax.set_xlabel("Resistance (ohm)")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    if xlim is not None:
        ax.set_xlim(*xlim)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)


def _plot_compare(freq_df: pd.DataFrame, time_df: pd.DataFrame, out_path: Path, share_x: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, sharex=share_x)
    xlim = None
    if share_x:
        vals = []
        for df in (freq_df, time_df):
            vals.extend(pd.to_numeric(df["lb"], errors="coerce").dropna().tolist())
            vals.extend(pd.to_numeric(df["ub"], errors="coerce").dropna().tolist())
        if vals:
            lo = min(vals)
            hi = max(vals)
            if lo == hi:
                pad = max(abs(lo) * 0.1, 1e-6)
                lo -= pad
                hi += pad
            else:
                pad = 0.05 * (hi - lo)
                lo -= pad
                hi += pad
            xlim = (lo, hi)

    for ax, df, title, color in [
        (axes[0], freq_df, "Frequency-domain prior bounds (matched5)", "#4c78a8"),
        (axes[1], time_df, "Time-domain fitted ranges (matched5)", "#f58518"),
    ]:
        y = np.arange(len(df))
        for i, row in df.iterrows():
            lb = row["lb"]
            ub = row["ub"]
            if pd.notna(lb) and pd.notna(ub):
                left = min(lb, ub)
                width = abs(ub - lb)
                if width == 0:
                    width = 1e-12
                ax.barh(i, width, left=left, height=0.55, color=color, alpha=0.8)
                ax.text(ub, i, f" [{lb:.4g}, {ub:.4g}]", va="center", ha="left", fontsize=8)
            else:
                ax.text(0.02, i, "N/A", va="center", ha="left", fontsize=9, transform=ax.get_yaxis_transform())
        ax.set_title(title)
        ax.set_xlabel("Resistance (ohm)")
        ax.grid(axis="x", alpha=0.25)
        ax.set_yticks(y)
        ax.set_yticklabels(df["parameter"])
        ax.invert_yaxis()
        if xlim is not None:
            ax.set_xlim(*xlim)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot matched frequency-domain prior bounds and time-domain fitted resistance ranges.")
    ap.add_argument("--prior_csv", required=True, help="CSV containing frequency-domain prior bounds.")
    ap.add_argument("--train_csv", required=True, help="CSV containing merged/training table with time-domain fitted values.")
    ap.add_argument("--out_dir", required=True, help="Directory to save plots and CSV summaries.")
    args = ap.parse_args()

    priors = pd.read_csv(args.prior_csv)
    train_df = pd.read_csv(args.train_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    freq_df, time_df = build_range_tables(priors, train_df)
    freq_csv = out_dir / "ecm_range__frequency_domain_prior_bounds__matched5.csv"
    time_csv = out_dir / "ecm_range__time_domain_fitted_ranges__matched5.csv"
    freq_df.to_csv(freq_csv, index=False)
    time_df.to_csv(time_csv, index=False)
    print("saved", freq_csv)
    print("saved", time_csv)

    freq_png = out_dir / "ecm_range__frequency_domain_prior_bounds__matched5.png"
    time_png = out_dir / "ecm_range__time_domain_fitted_ranges__matched5.png"
    compare_png = out_dir / "ecm_range__freq_vs_time__matched5.png"
    compare_shared_png = out_dir / "ecm_range__freq_vs_time__matched5__shared_x.png"

    _plot_range_df(freq_df, "Frequency-domain prior bounds (matched5)", freq_png, "#4c78a8")
    _plot_range_df(time_df, "Time-domain fitted ranges (matched5)", time_png, "#f58518")
    _plot_compare(freq_df, time_df, compare_png, share_x=False)
    _plot_compare(freq_df, time_df, compare_shared_png, share_x=True)


if __name__ == "__main__":
    main()
