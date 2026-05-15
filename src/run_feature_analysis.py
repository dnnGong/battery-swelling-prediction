#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd: List[str], env: dict) -> None:
    print("[INFO] Running:")
    print("       " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)


def infer_predictions_csv(pred_csv: str, results_dir: str) -> Path:
    if pred_csv:
        path = Path(pred_csv)
        if not path.exists():
            raise FileNotFoundError(f"--pred_csv not found: {path}")
        return path
    if not results_dir:
        raise ValueError("Provide either --pred_csv or --results_dir.")
    root = Path(results_dir)
    matches = sorted(root.glob("predictions__*.csv"))
    if not matches:
        raise FileNotFoundError(f"No predictions__*.csv found under: {root}")
    if len(matches) > 1:
        print(f"[WARN] Multiple predictions CSVs found under {root}; using: {matches[0]}")
    return matches[0]


def build_columns_arg(custom_features: str, target_col: str) -> str:
    feat_cols = [x.strip() for x in custom_features.split(",") if x.strip()]
    return ",".join(feat_cols + [target_col])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run a bundled feature-analysis workflow for one trained experiment: "
            "prediction-vs-ground-truth scatter, correlation matrix, and permutation importance."
        )
    )
    ap.add_argument("--table_csv", required=True, help="Feature table CSV used for training.")
    ap.add_argument("--out_dir", required=True, help="Directory to save all analysis outputs.")
    ap.add_argument("--pred_csv", default="", help="Optional predictions CSV. If omitted, infer from --results_dir.")
    ap.add_argument("--results_dir", default="", help="Optional trained-results directory used to infer predictions CSV.")
    ap.add_argument("--custom_features", required=True, help="Comma-separated training feature columns.")
    ap.add_argument("--target_col", default="y_abs_thickness_t", help="Target column to include in correlation matrix.")
    ap.add_argument("--group_tag", default="HYCL", help="Group tag filter, e.g. HYCL.")
    ap.add_argument("--target_mode", choices=["current", "fixed_T", "future_delta_TK"], default="current")
    ap.add_argument("--sample_mode", choices=["anchor", "rowwise"], default="rowwise")
    ap.add_argument("--label_mode", choices=["absolute", "delta"], default="absolute")
    ap.add_argument("--target_transform", choices=["none", "log"], default="log")
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--future_k", type=int, default=20)
    ap.add_argument("--max_input_cycle", type=int, default=120)
    ap.add_argument("--model", default="XGBoost", help="Model name for permutation importance.")
    ap.add_argument("--metric", choices=["mae", "rmse"], default="mae")
    ap.add_argument("--n_repeats", type=int, default=30)
    ap.add_argument("--corr_method", choices=["pearson", "spearman", "kendall"], default="spearman")
    ap.add_argument("--xgb_n_estimators", type=int, default=1200)
    ap.add_argument("--xgb_max_depth", type=int, default=4)
    ap.add_argument("--xgb_learning_rate", type=float, default=0.015)
    ap.add_argument("--xgb_subsample", type=float, default=0.85)
    ap.add_argument("--xgb_colsample_bytree", type=float, default=0.85)
    ap.add_argument("--xgb_min_child_weight", type=float, default=2.0)
    ap.add_argument("--xgb_reg_alpha", type=float, default=0.05)
    ap.add_argument("--xgb_reg_lambda", type=float, default=2.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = infer_predictions_csv(args.pred_csv, args.results_dir)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))

    scatter_base = out_dir / "pred_scatter.png"
    corr_base = out_dir / "feature_corr.png"

    run_cmd(
        [
            sys.executable,
            "src/plot_predictions_scatter.py",
            "--pred_csv",
            str(pred_csv),
            "--out_png",
            str(scatter_base),
            "--mode",
            "combined",
        ],
        env=env,
    )

    run_cmd(
        [
            sys.executable,
            "src/plot_feature_corr.py",
            "--table_csv",
            args.table_csv,
            "--out_png",
            str(corr_base),
            "--columns",
            build_columns_arg(args.custom_features, args.target_col),
            "--method",
            args.corr_method,
            "--annot",
            "--mode",
            "features_targets",
        ],
        env=env,
    )

    perm_cmd = [
        sys.executable,
        "src/plot_permutation_importance.py",
        "--table_csv",
        args.table_csv,
        "--out_dir",
        str(out_dir),
        "--target_mode",
        args.target_mode,
        "--sample_mode",
        args.sample_mode,
        "--label_mode",
        args.label_mode,
        "--target_transform",
        args.target_transform,
        "--group_tag",
        args.group_tag,
        "--model",
        args.model,
        "--custom_features",
        args.custom_features,
        "--max_input_cycle",
        str(args.max_input_cycle),
        "--n_repeats",
        str(args.n_repeats),
        "--metric",
        args.metric,
    ]
    if args.target_mode == "fixed_T":
        perm_cmd.extend(["--T", str(args.T)])
    if args.target_mode == "future_delta_TK":
        perm_cmd.extend(["--future_k", str(args.future_k)])
    if args.model == "XGBoost":
        perm_cmd.extend(
            [
                "--xgb_n_estimators",
                str(args.xgb_n_estimators),
                "--xgb_max_depth",
                str(args.xgb_max_depth),
                "--xgb_learning_rate",
                str(args.xgb_learning_rate),
                "--xgb_subsample",
                str(args.xgb_subsample),
                "--xgb_colsample_bytree",
                str(args.xgb_colsample_bytree),
                "--xgb_min_child_weight",
                str(args.xgb_min_child_weight),
                "--xgb_reg_alpha",
                str(args.xgb_reg_alpha),
                "--xgb_reg_lambda",
                str(args.xgb_reg_lambda),
            ]
        )
    run_cmd(perm_cmd, env=env)

    print(f"[INFO] Completed feature analysis bundle in: {out_dir}")


if __name__ == "__main__":
    main()
