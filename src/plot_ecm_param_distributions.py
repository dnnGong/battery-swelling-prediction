#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMMON_2CPE_PARAM_SPECS = [
    ("Rs_ohm", 0, "mOhm", 1000.0),
    ("Rsei_ohm", 1, "mOhm", 1000.0),
    ("Qsei", 2, "", 1.0),
    ("nsei", 3, "", 1.0),
    ("Rdl_ohm", 4, "mOhm", 1000.0),
    ("Qdl", 5, "", 1.0),
    ("ndl", 6, "", 1.0),
    ("sigma", 7, "", 1.0),
    ("warburg_tau", 8, "", 1.0),
]


def _safe_float(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def load_ecm_results(
    ecm_dir: Path,
    group_tag: str | None,
    rmse_max: float | None,
    sheet_filter: str | None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for p in sorted(ecm_dir.rglob("fit_result__*.json")):
        rel_parts = p.relative_to(ecm_dir).parts
        group = rel_parts[0] if rel_parts else ""

        if group_tag and group != group_tag:
            continue

        d = json.loads(p.read_text())
        rmse = _safe_float(d.get("rmse_complex_ohm"))
        if rmse_max is not None and np.isfinite(rmse) and rmse > rmse_max:
            continue

        sheet = str(d.get("sheet") or "")
        if sheet_filter and sheet != sheet_filter:
            continue

        row: Dict[str, object] = {
            "group_tag": group,
            "file_name": d.get("file_name"),
            "serial": d.get("serial"),
            "sheet": sheet,
            "measurement_cycle": _safe_float(d.get("measurement_cycle")),
            "circuit": str(d.get("circuit") or ""),
            "rmse_complex_ohm": rmse,
            "source_path": str(p),
        }

        params = d.get("params") or []
        for name, idx, _unit, scale in COMMON_2CPE_PARAM_SPECS:
            if idx < len(params):
                row[name] = _safe_float(params[idx]) * scale
            else:
                row[name] = np.nan

        if np.isfinite(row.get("Rs_ohm", np.nan)) and np.isfinite(row.get("Rsei_ohm", np.nan)) and np.isfinite(row.get("Rdl_ohm", np.nan)):
            row["R_total_ohm"] = row["Rs_ohm"] + row["Rsei_ohm"] + row["Rdl_ohm"]
        else:
            row["R_total_ohm"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append(
            {
                "parameter": c,
                "count": int(s.shape[0]),
                "min": float(s.min()),
                "p01": float(s.quantile(0.01)),
                "p05": float(s.quantile(0.05)),
                "median": float(s.median()),
                "mean": float(s.mean()),
                "p95": float(s.quantile(0.95)),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows)


def build_range_table_text(summary: pd.DataFrame) -> str:
    lines = ["Para.      Range (p05 - p95)"]
    for _, r in summary.iterrows():
        lines.append(f"{r['parameter']:<10} {r['p05']:.4g} - {r['p95']:.4g}")
    return "\n".join(lines)


def _pretty_axis_label(name: str) -> str:
    if name.endswith("_ohm"):
        return f"{name} (mOhm)"
    return name


def plot_distributions(
    df: pd.DataFrame,
    cols: List[str],
    summary: pd.DataFrame,
    out_png: Path,
    title: str,
):
    n = len(cols)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(14, 4.5 * nrows))
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols + 1, width_ratios=[0.65, 1, 1], wspace=0.35, hspace=0.35)

    ax_text = fig.add_subplot(gs[:, 0])
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        build_range_table_text(summary),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    for i, c in enumerate(cols):
        r = i // ncols
        cc = (i % ncols) + 1
        ax = fig.add_subplot(gs[r, cc])
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            ax.set_visible(False)
            continue

        p01 = float(s.quantile(0.01))
        p99 = float(s.quantile(0.99))
        x = s.clip(lower=p01, upper=p99)
        bins = min(30, max(12, int(np.sqrt(len(x)))))

        ax.hist(x, bins=bins, color="#2C7FB8", edgecolor="white", linewidth=0.4)
        ax.set_title(c)
        ax.set_xlabel(_pretty_axis_label(c))
        ax.set_ylabel("count")

        row = summary.loc[summary["parameter"] == c].iloc[0]
        ax.axvline(row["median"], color="#D95F0E", linestyle="--", linewidth=1.2, label="median")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.text(
            0.98,
            0.96,
            f"min={row['min']:.3g}\np95={row['p95']:.3g}\nmax={row['max']:.3g}",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

    fig.suptitle(title, fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot ECM parameter distributions from fit_result JSON files.")
    ap.add_argument("--ecm_dir", required=True, help="Directory containing fit_result__*.json outputs.")
    ap.add_argument("--out_dir", required=True, help="Directory to save summary CSV and figure.")
    ap.add_argument("--group_tag", default=None, help="Optional top-level subgroup under ecm_dir, e.g. HYCL/CL/FLC.")
    ap.add_argument("--sheet", default="03-4_EIS", help="Optional sheet filter. Use '' to disable.")
    ap.add_argument("--rmse_max", type=float, default=1.0, help="Optional RMSE upper bound to filter obviously bad fits.")
    ap.add_argument(
        "--params",
        default="Rs_ohm,Rsei_ohm,Rdl_ohm,R_total_ohm,nsei,ndl,sigma",
        help="Comma-separated parameter names to plot.",
    )
    ap.add_argument("--title", default="ECM Parameter Distributions", help="Figure title.")
    args = ap.parse_args()

    ecm_dir = Path(args.ecm_dir)
    out_dir = Path(args.out_dir)
    params = [x.strip() for x in args.params.split(",") if x.strip()]
    sheet_filter = args.sheet or None

    df = load_ecm_results(
        ecm_dir=ecm_dir,
        group_tag=args.group_tag,
        rmse_max=args.rmse_max,
        sheet_filter=sheet_filter,
    )
    if df.empty:
        raise SystemExit("No ECM fit results matched the requested filters.")

    missing = [c for c in params if c not in df.columns]
    if missing:
        raise SystemExit(f"Unknown parameter(s): {missing}")

    summary = summarize_columns(df, params)
    stem_parts = ["ecm_param_distribution"]
    if args.group_tag:
        stem_parts.append(args.group_tag)
    if sheet_filter:
        stem_parts.append(sheet_filter.replace("/", "_"))
    stem = "__".join(stem_parts)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{stem}.csv"
    out_png = out_dir / f"{stem}.png"
    summary.to_csv(out_csv, index=False)

    title = args.title
    if args.group_tag:
        title += f" ({args.group_tag})"
    plot_distributions(df=df, cols=params, summary=summary, out_png=out_png, title=title)

    print(f"[INFO] matched_fits={len(df)}")
    print(f"[INFO] saved summary: {out_csv}")
    print(f"[INFO] saved figure: {out_png}")


if __name__ == "__main__":
    main()
