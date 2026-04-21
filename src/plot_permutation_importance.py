#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, List
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is importable when launched from outside repo dir.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_swelling_models import (
    build_models,
    pick_anchor_rows_fixed_T,
    pick_rowwise_rows_current,
    pick_rows_future_delta_TK,
    train_test_group_split,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


def predict_with_transform(model: object, X: np.ndarray, target_transform: str) -> np.ndarray:
    pred_model = np.asarray(model.predict(X)).reshape(-1)
    return np.exp(pred_model) if target_transform == "log" else pred_model


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.table_csv)
    if args.target_mode == "fixed_T":
        if args.sample_mode == "anchor":
            data = pick_anchor_rows_fixed_T(df, T=args.T, max_input_cycle=args.max_input_cycle)
        else:
            data = pick_rowwise_rows_current(df, max_input_cycle=args.max_input_cycle)
    else:
        data = pick_rows_future_delta_TK(df, max_input_cycle=args.max_input_cycle)
    if data.empty:
        raise ValueError("No rows after target-mode filtering.")
    return data


def plot_importance(df_imp: pd.DataFrame, out_path: Path, metric_col: str) -> None:
    sub = df_imp.sort_values(metric_col, ascending=True).reset_index(drop=True)
    labels = [x.replace("feat_", "") for x in sub["feature"].tolist()]
    vals = sub[metric_col].to_numpy(dtype=float)
    errs = sub[f"{metric_col}_std"].to_numpy(dtype=float)

    fig_h = max(4.8, 0.55 * len(sub) + 1.5)
    fig, ax = plt.subplots(figsize=(8.4, fig_h))
    y = np.arange(len(sub))
    ax.barh(y, vals, xerr=errs, color="#3b82f6", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f"Permutation Importance ({metric_col} increase)")
    ax.set_title("Feature Permutation Importance")
    ax.grid(alpha=0.25, axis="x")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute held-out permutation importance for one regression model.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--table_csv", required=True, help="Input feature table CSV path.")
    ap.add_argument("--out_dir", required=True, help="Output directory for CSV and PNG.")
    ap.add_argument("--target_mode", choices=["fixed_T", "future_delta_TK"], required=True)
    ap.add_argument("--label_mode", choices=["absolute", "delta"], required=True)
    ap.add_argument("--target_transform", choices=["none", "log"], default="none")
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--future_k", type=int, default=20)
    ap.add_argument("--max_input_cycle", type=int, default=50)
    ap.add_argument(
        "--sample_mode",
        choices=["anchor", "rowwise"],
        default="anchor",
        help="Sampling mode for fixed_T. Use rowwise to match rowwise training experiments.",
    )
    ap.add_argument("--group_tag", default="HYCL", help="One of CL/FLC/HYCL.")
    ap.add_argument("--model", default="Ridge", help="Single model name, e.g. Ridge or RandomForest.")
    ap.add_argument("--custom_features", required=True, help="Comma list of feature columns.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--n_repeats", type=int, default=30)
    ap.add_argument("--metric", choices=["mae", "rmse"], default="mae", help="Metric used for ranking and plot axis.")
    ap.add_argument("--xgb_n_estimators", type=int, default=600, help="XGBoost n_estimators.")
    ap.add_argument("--xgb_max_depth", type=int, default=6, help="XGBoost max_depth.")
    ap.add_argument("--xgb_learning_rate", type=float, default=0.03, help="XGBoost learning_rate.")
    ap.add_argument("--xgb_subsample", type=float, default=0.9, help="XGBoost subsample.")
    ap.add_argument("--xgb_colsample_bytree", type=float, default=0.9, help="XGBoost colsample_bytree.")
    ap.add_argument("--xgb_min_child_weight", type=float, default=1.0, help="XGBoost min_child_weight.")
    ap.add_argument("--xgb_reg_alpha", type=float, default=0.0, help="XGBoost reg_alpha.")
    ap.add_argument("--xgb_reg_lambda", type=float, default=1.0, help="XGBoost reg_lambda.")
    args = ap.parse_args()

    if args.target_transform == "log" and args.label_mode != "absolute":
        raise ValueError("target_transform=log is only supported with label_mode=absolute.")

    data = build_dataset(args)
    sub = data[data["group_tag"] == args.group_tag].copy()
    label_col = "target_abs" if args.label_mode == "absolute" else "target_delta"
    sub = sub.dropna(subset=[label_col, "cell_key"]).copy()
    if sub.empty:
        raise ValueError(f"No rows left for group_tag={args.group_tag}")

    feature_cols = [x.strip() for x in args.custom_features.split(",") if x.strip()]
    missing = [c for c in feature_cols if c not in sub.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    valid_cols = [c for c in feature_cols if sub[c].notna().sum() > 0]
    tr_idx, te_idx = train_test_group_split(sub, test_size=args.test_size, seed=args.seed)
    tr_df = sub.iloc[tr_idx].copy()
    te_df = sub.iloc[te_idx].copy()

    med = tr_df[valid_cols].median(numeric_only=True)
    tr_df[valid_cols] = tr_df[valid_cols].fillna(med).fillna(0.0)
    te_df[valid_cols] = te_df[valid_cols].fillna(med).fillna(0.0)

    X_tr = tr_df[valid_cols].to_numpy(dtype=float)
    X_te = te_df[valid_cols].to_numpy(dtype=float)
    y_tr = tr_df[label_col].to_numpy(dtype=float).reshape(-1)
    y_te = te_df[label_col].to_numpy(dtype=float).reshape(-1)
    y_tr_model = np.log(y_tr) if args.target_transform == "log" else y_tr

    models = build_models(
        seed=args.seed,
        model_set="all",
        include_models=[args.model],
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
        xgb_min_child_weight=args.xgb_min_child_weight,
        xgb_reg_alpha=args.xgb_reg_alpha,
        xgb_reg_lambda=args.xgb_reg_lambda,
    )
    if args.model not in models:
        raise ValueError(f"Unknown/unavailable model: {args.model}")
    model = models[args.model]
    model.fit(X_tr, y_tr_model)

    base_pred = predict_with_transform(model, X_te, args.target_transform)
    base_metrics = compute_metrics(y_te, base_pred)

    rng = np.random.default_rng(args.seed)
    records: List[Dict[str, float]] = []
    for j, feature in enumerate(valid_cols):
        maes: List[float] = []
        rmses: List[float] = []
        for _ in range(args.n_repeats):
            X_perm = X_te.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            pred = predict_with_transform(model, X_perm, args.target_transform)
            met = compute_metrics(y_te, pred)
            maes.append(met["mae"] - base_metrics["mae"])
            rmses.append(met["rmse"] - base_metrics["rmse"])
        records.append(
            {
                "feature": feature,
                "baseline_mae": base_metrics["mae"],
                "baseline_rmse": base_metrics["rmse"],
                "importance_mae": float(np.mean(maes)),
                "importance_mae_std": float(np.std(maes, ddof=0)),
                "importance_rmse": float(np.mean(rmses)),
                "importance_rmse_std": float(np.std(rmses, ddof=0)),
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_parts = ["perm_importance", args.target_mode, args.label_mode]
    if args.target_mode == "fixed_T" and args.sample_mode != "anchor":
        stem_parts.append(args.sample_mode)
    stem_parts.extend([args.group_tag, args.model])
    stem = "__".join(stem_parts)
    csv_path = out_dir / f"{stem}.csv"
    png_path = out_dir / f"{stem}__{args.metric}.png"

    df_imp = pd.DataFrame(records).sort_values(f"importance_{args.metric}", ascending=False).reset_index(drop=True)
    df_imp.to_csv(csv_path, index=False)
    plot_importance(df_imp, png_path, f"importance_{args.metric}")

    print(f"[INFO] Saved permutation importance CSV: {csv_path}")
    print(f"[INFO] Saved permutation importance plot: {png_path}")
    print(df_imp[["feature", f"importance_{args.metric}", f"importance_{args.metric}_std"]])


if __name__ == "__main__":
    main()
