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

from src.train_swelling_models import pick_anchor_rows_fixed_T, pick_rows_future_delta_TK


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.table_csv)
    if args.target_mode == "fixed_T":
        data = pick_anchor_rows_fixed_T(df, T=args.T, max_input_cycle=args.max_input_cycle)
    else:
        data = pick_rows_future_delta_TK(df, max_input_cycle=args.max_input_cycle)
    if data.empty:
        raise ValueError("No rows after target-mode filtering.")
    return data


def cv_mae_for_features(
    sub: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    model_name: str,
    target_transform: str,
    seed: int,
    cv_splits: int,
) -> Dict[str, float]:
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import PLSRegression

    df_use = sub.dropna(subset=feature_cols + [label_col]).copy()
    if len(df_use) < max(4, cv_splits):
        raise ValueError(f"Too few rows for CV after dropping NaNs: {len(df_use)}")

    X = df_use[feature_cols].to_numpy(dtype=float)
    y = df_use[label_col].to_numpy(dtype=float).reshape(-1)
    if target_transform == "log":
        if np.any(y <= 0):
            raise ValueError("target_transform=log requires strictly positive target values.")
        y_model = np.log(y)
    else:
        y_model = y

    if model_name == "Linear":
        model = make_pipeline(StandardScaler(), LinearRegression())
    elif model_name == "Ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed))
    elif model_name == "PCR":
        model = make_pipeline(StandardScaler(), PCA(n_components=0.95, svd_solver="full"), LinearRegression())
    elif model_name == "PLSR":
        n_comp = max(1, min(8, X.shape[1], max(1, len(X) - 1)))
        model = make_pipeline(StandardScaler(), PLSRegression(n_components=n_comp))
    else:
        raise ValueError(f"Unsupported model for incremental CV: {model_name}")

    kf = KFold(n_splits=min(cv_splits, len(df_use)), shuffle=True, random_state=seed)
    maes: List[float] = []
    rmses: List[float] = []
    for tr_idx, va_idx in kf.split(X):
        model.fit(X[tr_idx], y_model[tr_idx])
        pred_model = np.asarray(model.predict(X[va_idx])).reshape(-1)
        pred = np.exp(pred_model) if target_transform == "log" else pred_model
        err = y[va_idx] - pred
        maes.append(float(np.mean(np.abs(err))))
        rmses.append(float(np.sqrt(np.mean(err ** 2))))
    return {
        "cv_mae_mean": float(np.mean(maes)),
        "cv_mae_std": float(np.std(maes, ddof=0)),
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_rmse_std": float(np.std(rmses, ddof=0)),
        "n_rows": int(len(df_use)),
    }


def plot_metric(df: pd.DataFrame, out_path: Path, metric_col: str, metric_std_col: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    x = df["n_features"].to_numpy(dtype=int)
    y = df[metric_col].to_numpy(dtype=float)
    yerr = df[metric_std_col].to_numpy(dtype=float)
    ax.plot(x, y, marker="o", linewidth=2.0, color="#2563eb")
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.0, capsize=3)
    ax.set_xticks(x)
    ax.set_xlabel("Number of Features Added")
    ax.set_ylabel(metric_col.replace("_", " "))
    ax.set_title("Incremental CV Performance")
    ax.grid(alpha=0.25)

    for _, row in df.iterrows():
        ax.annotate(
            row["feature_added"].replace("feat_", ""),
            (int(row["n_features"]), float(row[metric_col])),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute incremental CV MAE/RMSE by adding features in a fixed order.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--table_csv", required=True, help="Input feature table CSV path.")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--target_mode", choices=["fixed_T", "future_delta_TK"], required=True)
    ap.add_argument("--label_mode", choices=["absolute", "delta"], required=True)
    ap.add_argument("--target_transform", choices=["none", "log"], default="none")
    ap.add_argument("--group_tag", default="HYCL", help="One of CL/FLC/HYCL.")
    ap.add_argument("--model", choices=["Linear", "Ridge", "PCR", "PLSR"], default="Ridge")
    ap.add_argument("--custom_features", required=True, help="Comma list defining the fixed addition order.")
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--future_k", type=int, default=20)
    ap.add_argument("--max_input_cycle", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv_splits", type=int, default=5)
    args = ap.parse_args()

    if args.target_transform == "log" and args.label_mode != "absolute":
        raise ValueError("target_transform=log is only supported with label_mode=absolute.")

    data = build_dataset(args)
    sub = data[data["group_tag"] == args.group_tag].copy()
    label_col = "target_abs" if args.label_mode == "absolute" else "target_delta"
    sub = sub.dropna(subset=[label_col]).copy()
    if sub.empty:
        raise ValueError(f"No rows left for group_tag={args.group_tag}")

    feature_order = [x.strip() for x in args.custom_features.split(",") if x.strip()]
    missing = [c for c in feature_order if c not in sub.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    records: List[Dict[str, object]] = []
    for i in range(1, len(feature_order) + 1):
        cols = feature_order[:i]
        metrics = cv_mae_for_features(
            sub=sub,
            feature_cols=cols,
            label_col=label_col,
            model_name=args.model,
            target_transform=args.target_transform,
            seed=args.seed,
            cv_splits=args.cv_splits,
        )
        records.append(
            {
                "n_features": int(i),
                "feature_added": cols[-1],
                "features_used": ",".join(cols),
                **metrics,
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"incremental_cv__{args.target_mode}__{args.label_mode}__{args.group_tag}__{args.model}"
    csv_path = out_dir / f"{stem}.csv"
    png_mae = out_dir / f"{stem}__mae.png"
    png_rmse = out_dir / f"{stem}__rmse.png"

    df_out = pd.DataFrame(records)
    df_out.to_csv(csv_path, index=False)
    plot_metric(df_out, png_mae, "cv_mae_mean", "cv_mae_std")
    plot_metric(df_out, png_rmse, "cv_rmse_mean", "cv_rmse_std")

    print(f"[INFO] Saved incremental CV CSV: {csv_path}")
    print(f"[INFO] Saved incremental CV MAE plot: {png_mae}")
    print(f"[INFO] Saved incremental CV RMSE plot: {png_rmse}")
    print(df_out[["n_features", "feature_added", "cv_mae_mean", "cv_mae_std"]])


if __name__ == "__main__":
    main()
