#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


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


def pick_numeric_feature_cols(df: pd.DataFrame, feature_prefix: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(feature_prefix)]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return cols


def robust_z_scores(x: pd.DataFrame) -> pd.DataFrame:
    med = x.median(numeric_only=True)
    mad = (x - med).abs().median(numeric_only=True)
    mad = mad.replace(0.0, np.nan)
    z = 0.6745 * (x - med) / mad
    return z


def iqr_outlier_counts(x: pd.DataFrame, iqr_k: float) -> pd.Series:
    counts = pd.Series(0, index=x.index, dtype=int)
    for c in x.columns:
        s = x[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            continue
        lo = q1 - iqr_k * iqr
        hi = q3 + iqr_k * iqr
        counts += ((s < lo) | (s > hi)).astype(int)
    return counts


def mahalanobis_d2(x: pd.DataFrame) -> np.ndarray:
    xv = x.to_numpy(dtype=float)
    mu = xv.mean(axis=0)
    cov = np.cov(xv, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=float)
    cov = cov + np.eye(cov.shape[0]) * 1e-8
    cov_inv = np.linalg.pinv(cov)
    d2 = np.einsum("ij,jk,ik->i", xv - mu, cov_inv, xv - mu)
    return d2


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Detect (and optionally remove) extreme outliers from feature_table.csv. "
            "Default behavior is report-only; rows are removed only when --apply_drop is set."
        ),
        epilog=(
            "Example (report only):\n"
            "  python src/filter_feature_table_outliers.py "
            "--table_csv ./data/ml/test15/feature_table_test15.csv "
            "--out_dir ./data/ml/test15/outlier_report "
            "--sample_mode future_delta_TK --max_input_cycle 50 --group_tag HYCL\n\n"
            "Example (apply drop):\n"
            "  python src/filter_feature_table_outliers.py "
            "--table_csv ./data/ml/test15/feature_table_test15.csv "
            "--out_dir ./data/ml/test15/outlier_report "
            "--sample_mode future_delta_TK --max_input_cycle 50 --group_tag HYCL "
            "--apply_drop"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--table_csv", required=True, help="Input feature table CSV.")
    ap.add_argument("--out_dir", required=True, help="Output directory for report and optional cleaned table.")
    ap.add_argument("--feature_prefix", default="feat_", help="Prefix for feature columns.")
    ap.add_argument("--sample_mode", choices=["raw", "fixed_T", "future_delta_TK"], default="raw")
    ap.add_argument("--T", type=int, default=100, help="Target cycle for fixed_T mode.")
    ap.add_argument("--max_input_cycle", type=int, default=50, help="Max cycle for fixed_T/future modes.")
    ap.add_argument("--group_tag", default="", help="Optional group filter: CL/FLC/HYCL.")
    ap.add_argument("--method", choices=["robust", "iqr", "combined"], default="combined")
    ap.add_argument("--robust_z_thresh", type=float, default=6.0, help="Threshold on max absolute robust z-score.")
    ap.add_argument("--iqr_k", type=float, default=3.0, help="IQR multiplier for extreme fence.")
    ap.add_argument("--iqr_min_count", type=int, default=1, help="Min number of violated features to flag by IQR.")
    ap.add_argument("--mahal_q", type=float, default=0.995, help="Mahalanobis quantile threshold for combined mode.")
    ap.add_argument("--max_features", type=int, default=40, help="Use top-N variance features for stable detection.")
    ap.add_argument("--apply_drop", action="store_true", help="Actually remove flagged rows from original table.")
    ap.add_argument("--out_clean_csv", default="", help="Optional cleaned output CSV path.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full = pd.read_csv(args.table_csv)
    if full.empty:
        raise ValueError("Input table is empty.")
    full["_rowid__"] = np.arange(len(full))

    if args.sample_mode == "fixed_T":
        data = pick_anchor_rows_fixed_T(full, T=args.T, max_input_cycle=args.max_input_cycle)
    elif args.sample_mode == "future_delta_TK":
        data = pick_rows_future_delta_TK(full, max_input_cycle=args.max_input_cycle)
    else:
        data = full.copy()

    if data.empty:
        raise ValueError("No rows after sample_mode filtering.")

    if args.group_tag:
        data = data[data["group_tag"] == args.group_tag].copy()
        if data.empty:
            raise ValueError(f"No rows for group_tag={args.group_tag}")

    feat_cols = pick_numeric_feature_cols(data, args.feature_prefix)
    if not feat_cols:
        raise ValueError(f"No usable numeric features found with prefix: {args.feature_prefix}")

    if len(feat_cols) > args.max_features:
        var = data[feat_cols].var(numeric_only=True).sort_values(ascending=False)
        feat_cols = var.head(args.max_features).index.tolist()

    x = data[feat_cols].apply(pd.to_numeric, errors="coerce")
    valid_mask = x.notna().all(axis=1)
    x = x[valid_mask].copy()
    work = data.loc[valid_mask].copy()
    if x.empty:
        raise ValueError("No rows with complete numeric feature values.")

    rz = robust_z_scores(x)
    max_abs_rz = rz.abs().max(axis=1).fillna(0.0)
    iqr_counts = iqr_outlier_counts(x, iqr_k=float(args.iqr_k))
    d2 = pd.Series(mahalanobis_d2(x), index=x.index)
    d2_thr = float(np.quantile(d2, min(max(args.mahal_q, 0.5), 0.9999)))

    if args.method == "robust":
        flag = max_abs_rz >= args.robust_z_thresh
    elif args.method == "iqr":
        flag = iqr_counts >= max(1, args.iqr_min_count)
    else:
        flag = (
            (max_abs_rz >= args.robust_z_thresh)
            | (iqr_counts >= max(1, args.iqr_min_count))
            | (d2 >= d2_thr)
        )

    report = work.copy()
    report["outlier_flag"] = flag.values
    report["max_abs_robust_z"] = max_abs_rz.values
    report["iqr_outlier_feature_count"] = iqr_counts.values
    report["mahalanobis_d2"] = d2.values

    id_cols = [c for c in ["serial", "cell_key", "group_tag", "cycle_t", "target_cycle"] if c in report.columns]
    outlier_cols = id_cols + ["outlier_flag", "max_abs_robust_z", "iqr_outlier_feature_count", "mahalanobis_d2"] + feat_cols
    outlier_csv = out_dir / "outlier_rows.csv"
    report[outlier_cols].sort_values(
        ["outlier_flag", "mahalanobis_d2", "max_abs_robust_z"], ascending=[False, False, False]
    ).to_csv(outlier_csv, index=False)

    summary: Dict[str, object] = {
        "table_csv": str(args.table_csv),
        "sample_mode": args.sample_mode,
        "group_tag": args.group_tag,
        "method": args.method,
        "rows_input_total": int(len(full)),
        "rows_used_for_detection": int(len(report)),
        "features_used_count": int(len(feat_cols)),
        "features_used": feat_cols,
        "outlier_count": int(report["outlier_flag"].sum()),
        "outlier_ratio": float(report["outlier_flag"].mean()),
        "robust_z_thresh": float(args.robust_z_thresh),
        "iqr_k": float(args.iqr_k),
        "iqr_min_count": int(args.iqr_min_count),
        "mahal_q": float(args.mahal_q),
        "mahal_d2_threshold": float(d2_thr),
        "apply_drop": bool(args.apply_drop),
        "outlier_csv": str(outlier_csv),
    }

    clean_csv = None
    if args.apply_drop:
        drop_ids = set(report.loc[report["outlier_flag"], "_rowid__"].astype(int).tolist())
        cleaned = full[~full["_rowid__"].isin(drop_ids)].copy()
        cleaned = cleaned.drop(columns=["_rowid__"])
        clean_csv = Path(args.out_clean_csv) if args.out_clean_csv else out_dir / "feature_table_cleaned.csv"
        clean_csv.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(clean_csv, index=False)
        summary["clean_csv"] = str(clean_csv)
        summary["rows_after_drop"] = int(len(cleaned))
        summary["dropped_rows"] = int(len(full) - len(cleaned))
    else:
        full = full.drop(columns=["_rowid__"])

    summary_json = out_dir / "outlier_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"[INFO] Rows used for detection: {summary['rows_used_for_detection']}")
    print(f"[INFO] Feature count used: {summary['features_used_count']}")
    print(f"[INFO] Outlier count: {summary['outlier_count']} ({summary['outlier_ratio']:.2%})")
    print(f"[INFO] Saved outlier rows: {outlier_csv}")
    print(f"[INFO] Saved summary: {summary_json}")
    if clean_csv is not None:
        print(f"[INFO] Saved cleaned table: {clean_csv}")


if __name__ == "__main__":
    main()
