#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_FEATURES = [
    "feat_cycle_t",
    "feat_Rs_ohm",
    "feat_nsei",
    "feat_ndl",
    "feat_R_total_ohm",
    "feat_sigma",
    "feat_capacity_t",
    "feat_capacity_slope_10",
    "feat_dcir_soc_t",
]

DEFAULT_CORR_COLUMNS = [
    "feat_Rs_ohm",
    "feat_dcir_soc_t",
    "feat_R_total_ohm",
    "feat_capacity_t",
    "feat_capacity_slope_10",
    "feat_cycle_t",
    "feat_nsei",
    "feat_ndl",
    "feat_sigma",
    "y_abs_thickness_t",
]

ECM_COMPLETE_COLS = [
    "feat_Rs_ohm",
    "feat_nsei",
    "feat_ndl",
    "feat_R_total_ohm",
    "feat_sigma",
]


def parse_socs(value: str) -> List[int]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(float(part)))
    if not out:
        raise ValueError("No SOC values provided.")
    return out


def run(cmd: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[RUN]", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}. See log: {log_path}")


def write_ecm_complete(src_csv: Path, dst_csv: Path) -> Dict[str, int]:
    df = pd.read_csv(src_csv)
    before = len(df)
    df = df.dropna(subset=ECM_COMPLETE_COLS).copy()
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_csv, index=False)
    return {"rows_before": int(before), "rows_after": int(len(df))}


def summarize_soc(table_csv: Path, soc: int) -> Dict[str, float]:
    df = pd.read_csv(table_csv)
    df = df[(df["group_tag"] == "HYCL") & (pd.to_numeric(df["cycle_t"], errors="coerce") <= 50)].copy()
    pair = df.dropna(subset=["feat_Rs_ohm", "feat_R_total_ohm", "feat_dcir_soc_t"]).copy()

    row: Dict[str, float] = {
        "soc_target": int(soc),
        "hycl_rows_le50": int(len(df)),
        "hycl_cells_le50": int(df["cell_key"].nunique()) if "cell_key" in df else 0,
        "dcir_non_null": int(df["feat_dcir_soc_t"].notna().sum()) if "feat_dcir_soc_t" in df else 0,
        "pair_rows": int(len(pair)),
        "pair_cells": int(pair["cell_key"].nunique()) if "cell_key" in pair else 0,
    }

    if not pair.empty:
        spearman = pair[["feat_Rs_ohm", "feat_R_total_ohm", "feat_dcir_soc_t"]].corr(method="spearman")
        pearson = pair[["feat_Rs_ohm", "feat_R_total_ohm", "feat_dcir_soc_t"]].corr(method="pearson")
        row.update(
            {
                "spearman_Rs_dcir": float(spearman.loc["feat_Rs_ohm", "feat_dcir_soc_t"]),
                "spearman_Rtotal_dcir": float(spearman.loc["feat_R_total_ohm", "feat_dcir_soc_t"]),
                "pearson_Rs_dcir": float(pearson.loc["feat_Rs_ohm", "feat_dcir_soc_t"]),
                "pearson_Rtotal_dcir": float(pearson.loc["feat_R_total_ohm", "feat_dcir_soc_t"]),
                "dcir_mean": float(pair["feat_dcir_soc_t"].mean()),
                "dcir_std": float(pair["feat_dcir_soc_t"].std()),
            }
        )
        if "feat_dcir_cycle_used" in pair:
            row["dcir_cycle_used_unique"] = ",".join(
                str(int(x)) for x in sorted(pair["feat_dcir_cycle_used"].dropna().unique())
            )
    return row


def read_hycl_result(results_csv: Path) -> Dict[str, float]:
    df = pd.read_csv(results_csv)
    if "group_tag" in df.columns:
        df = df[df["group_tag"] == "HYCL"].copy()
    if df.empty:
        return {}
    r = df.iloc[0]
    return {
        "mae": float(r["mae"]),
        "rmse": float(r["rmse"]),
        "n_train": int(r["n_train"]),
        "n_test": int(r["n_test"]),
        "n_cells_train": int(r["n_cells_train"]),
        "n_cells_test": int(r["n_cells_test"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run current-cycle XGBoost + permutation/correlation analysis across DCIR SOC targets."
    )
    ap.add_argument("--xlsx_dir", default="./dataset/OneDrive_1_2-20-2026")
    ap.add_argument("--ecm_dir", default="./data/latest_test_26_4_21/ecm_w_cycle")
    ap.add_argument(
        "--out_root",
        default="./data/latest_test_26_4_22_diff_soc",
        help="Root output directory. Each SOC gets a soc_XXX subdirectory.",
    )
    ap.add_argument("--socs", default="0,10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--min_cycle", type=int, default=5)
    ap.add_argument("--max_cycle", type=int, default=200)
    ap.add_argument("--max_input_cycle", type=int, default=50)
    ap.add_argument("--future_k", type=int, default=20)
    ap.add_argument("--group_tag", default="HYCL")
    ap.add_argument("--n_repeats", type=int, default=30)
    ap.add_argument("--skip_build", action="store_true", help="Reuse existing feature_table.csv files if present.")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir = out_root / "logs"
    features_csv = ",".join(DEFAULT_FEATURES)
    corr_cols = ",".join(DEFAULT_CORR_COLUMNS)

    summary_rows: List[Dict[str, float]] = []
    for soc in parse_socs(args.socs):
        soc_dir = out_root / f"soc_{soc:03d}"
        soc_dir.mkdir(parents=True, exist_ok=True)
        table_csv = soc_dir / "feature_table.csv"
        complete_csv = soc_dir / "feature_table_ecm_complete.csv"
        model_dir = soc_dir / "xgb_t11_lighter_reg_current"

        if not args.skip_build or not table_csv.exists():
            run(
                [
                    sys.executable,
                    "src/build_feature_table.py",
                    "--xlsx_dir",
                    args.xlsx_dir,
                    "--ecm_dir",
                    args.ecm_dir,
                    "--out_csv",
                    str(table_csv),
                    "--min_cycle",
                    str(args.min_cycle),
                    "--max_cycle",
                    str(args.max_cycle),
                    "--future_k",
                    str(args.future_k),
                    "--soc_target",
                    str(soc),
                    "--dcir_align_mode",
                    "last_le",
                    "--log_file",
                    str(logs_dir / f"build_feature_table_soc{soc:03d}.log"),
                ],
                logs_dir / f"build_feature_table_soc{soc:03d}.stdout.log",
            )

        complete_info = write_ecm_complete(table_csv, complete_csv)

        run_tag = f"xgb_t11_lighter_reg_soc{soc:03d}_current"
        run(
            [
                sys.executable,
                "src/train_swelling_models.py",
                "--table_csv",
                str(complete_csv),
                "--out_dir",
                str(model_dir),
                "--target_mode",
                "current",
                "--sample_mode",
                "rowwise",
                "--label_mode",
                "absolute",
                "--target_transform",
                "log",
                "--max_input_cycle",
                str(args.max_input_cycle),
                "--model_set",
                "basic",
                "--models",
                "XGBoost",
                "--feature_set",
                "custom",
                "--custom_features",
                features_csv,
                "--xgb_n_estimators",
                "1200",
                "--xgb_max_depth",
                "4",
                "--xgb_learning_rate",
                "0.015",
                "--xgb_subsample",
                "0.85",
                "--xgb_colsample_bytree",
                "0.85",
                "--xgb_min_child_weight",
                "2",
                "--xgb_reg_alpha",
                "0.05",
                "--xgb_reg_lambda",
                "2.0",
                "--run_tag",
                run_tag,
                "--log_file",
                str(logs_dir / f"train_soc{soc:03d}.log"),
            ],
            logs_dir / f"train_soc{soc:03d}.stdout.log",
        )

        run(
            [
                sys.executable,
                "src/plot_permutation_importance.py",
                "--table_csv",
                str(complete_csv),
                "--out_dir",
                str(model_dir),
                "--target_mode",
                "current",
                "--sample_mode",
                "rowwise",
                "--label_mode",
                "absolute",
                "--target_transform",
                "log",
                "--group_tag",
                args.group_tag,
                "--model",
                "XGBoost",
                "--custom_features",
                features_csv,
                "--max_input_cycle",
                str(args.max_input_cycle),
                "--xgb_n_estimators",
                "1200",
                "--xgb_max_depth",
                "4",
                "--xgb_learning_rate",
                "0.015",
                "--xgb_subsample",
                "0.85",
                "--xgb_colsample_bytree",
                "0.85",
                "--xgb_min_child_weight",
                "2",
                "--xgb_reg_alpha",
                "0.05",
                "--xgb_reg_lambda",
                "2.0",
                "--n_repeats",
                str(args.n_repeats),
                "--metric",
                "mae",
            ],
            logs_dir / f"perm_importance_soc{soc:03d}.log",
        )

        run(
            [
                sys.executable,
                "src/plot_feature_corr.py",
                "--table_csv",
                str(complete_csv),
                "--out_png",
                str(soc_dir / f"corr_selected_ecm_cap_dcir_soc{soc:03d}_current.png"),
                "--columns",
                corr_cols,
                "--group_tag",
                args.group_tag,
                "--max_cycle",
                str(args.max_input_cycle),
                "--method",
                "spearman",
                "--annot",
                "--mode",
                "features_targets",
            ],
            logs_dir / f"corr_soc{soc:03d}.log",
        )

        row = summarize_soc(complete_csv, soc)
        row.update(complete_info)
        results_csv = model_dir / f"results__current__absolute__current_cycle__{run_tag}.csv"
        row.update(read_hycl_result(results_csv))
        summary_rows.append(row)

        summary_path = out_root / "soc_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"[INFO] Updated summary: {summary_path}")

    manifest = {
        "xlsx_dir": args.xlsx_dir,
        "ecm_dir": args.ecm_dir,
        "out_root": str(out_root),
        "socs": parse_socs(args.socs),
        "target_mode": "current",
        "label_mode": "absolute",
        "features": DEFAULT_FEATURES,
        "corr_columns": DEFAULT_CORR_COLUMNS,
    }
    (out_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[INFO] Done. Summary -> {out_root / 'soc_summary.csv'}")


if __name__ == "__main__":
    main()
