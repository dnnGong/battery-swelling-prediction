#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pick_anchor_rows_fixed_T(df: pd.DataFrame, T: int, max_input_cycle: int) -> pd.DataFrame:
    """
    Build one sample per cell, using latest row at cycle<=max_input_cycle as input,
    and target from cycle<=T latest row.
    """
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
    """
    Use row at cycle t as input and future columns (t->t+K) as target.
    """
    sub = df[(df["cycle_t"] <= max_input_cycle) & (df["has_future_k"] == 1)].copy()
    if sub.empty:
        return sub
    sub["target_abs"] = sub["y_future_abs_thickness_tk"].astype(float)
    sub["target_delta"] = sub["y_future_delta_thickness_tk"].astype(float)
    sub["target_cycle"] = (sub["cycle_t"] + sub["future_k"]).astype(int)
    return sub


def train_test_group_split(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.model_selection import GroupShuffleSplit
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required. Please install it first: pip install scikit-learn"
        ) from e

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df["cell_key"].to_numpy()
    idx = np.arange(len(df))
    tr, te = next(gss.split(idx, groups=groups))
    return tr, te


def build_models(seed: int) -> Dict[str, object]:
    try:
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required. Please install it first: pip install scikit-learn"
        ) from e

    models: Dict[str, object] = {
        "Ridge": Ridge(alpha=1.0, random_state=seed),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=4,
            objective="reg:squarederror",
        )
    except Exception:
        pass
    return models


def fit_eval_one_group(
    df_group: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    seed: int,
    test_size: float,
    min_rows: int,
    min_cells: int,
) -> Tuple[List[Dict], List[Dict]]:
    records: List[Dict] = []
    pred_rows: List[Dict] = []
    sub = df_group.dropna(subset=[label_col, "cell_key"]).copy()
    if len(sub) < min_rows or sub["cell_key"].nunique() < min_cells:
        return records, pred_rows

    # Drop feature columns that are entirely NaN in this group.
    valid_cols = [c for c in feature_cols if sub[c].notna().sum() > 0]
    if len(valid_cols) < 3:
        return records, pred_rows

    # Split first, then impute by train medians to avoid leakage.
    tr_idx, te_idx = train_test_group_split(sub, test_size=test_size, seed=seed)
    tr_df = sub.iloc[tr_idx].copy()
    te_df = sub.iloc[te_idx].copy()

    med = tr_df[valid_cols].median(numeric_only=True)
    tr_df[valid_cols] = tr_df[valid_cols].fillna(med)
    te_df[valid_cols] = te_df[valid_cols].fillna(med)

    # Any still-NaN columns (e.g., all-NaN in train) -> fill 0.
    tr_df[valid_cols] = tr_df[valid_cols].fillna(0.0)
    te_df[valid_cols] = te_df[valid_cols].fillna(0.0)

    X = sub[valid_cols].to_numpy(dtype=float)
    y = sub[label_col].to_numpy(dtype=float)
    X_tr = tr_df[valid_cols].to_numpy(dtype=float)
    X_te = te_df[valid_cols].to_numpy(dtype=float)
    y_tr = tr_df[label_col].to_numpy(dtype=float)
    y_te = te_df[label_col].to_numpy(dtype=float)

    models = build_models(seed)
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        records.append(
            {
                "model": name,
                "n_train": int(len(X_tr)),
                "n_test": int(len(X_te)),
                "n_cells_train": int(sub.iloc[tr_idx]["cell_key"].nunique()),
                "n_cells_test": int(sub.iloc[te_idx]["cell_key"].nunique()),
                "n_features_used": int(len(valid_cols)),
                "rmse": safe_rmse(y_te, pred),
                "mae": float(np.mean(np.abs(y_te - pred))),
            }
        )
        for row, y_true_i, y_pred_i in zip(te_df.itertuples(index=False), y_te, pred):
            pred_rows.append(
                {
                    "model": name,
                    "cell_key": row.cell_key,
                    "serial": getattr(row, "serial", ""),
                    "group_tag": getattr(row, "group_tag", ""),
                    "cycle_t": int(row.cycle_t),
                    "target_cycle": int(row.target_cycle),
                    "label_col": label_col,
                    "y_true": float(y_true_i),
                    "y_pred": float(y_pred_i),
                    "abs_error": float(abs(y_true_i - y_pred_i)),
                }
            )
    return records, pred_rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Train grouped swelling prediction models from feature_table.csv. "
            "The script trains separate models for CL, FLC, and HYCL."
        ),
        epilog=(
            "Examples:\n"
            "  Fixed-T absolute thickness:\n"
            "    python src/train_swelling_models.py --table_csv ./data/ml/feature_table.csv "
            "--out_dir ./data/ml/results --target_mode fixed_T --label_mode absolute --T 100 --max_input_cycle 50\n\n"
            "  Future delta thickness:\n"
            "    python src/train_swelling_models.py --table_csv ./data/ml/feature_table.csv "
            "--out_dir ./data/ml/results --target_mode future_delta_TK --label_mode delta --future_k 20 --max_input_cycle 50"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--table_csv", required=True, help="Input feature table CSV generated by src/build_feature_table.py")
    ap.add_argument("--out_dir", required=True, help="Output directory for results__*.csv, predictions__*.csv, and run_meta__*.json")
    ap.add_argument(
        "--target_mode",
        choices=["fixed_T", "future_delta_TK"],
        required=True,
        help=(
            "Prediction target definition:\n"
            "  fixed_T         : predict thickness at a fixed cycle T\n"
            "  future_delta_TK : predict future thickness at t+K or delta from t to t+K"
        ),
    )
    ap.add_argument(
        "--label_mode",
        choices=["absolute", "delta"],
        required=True,
        help=(
            "Label definition:\n"
            "  absolute : absolute thickness target\n"
            "  delta    : thickness change target"
        ),
    )
    ap.add_argument("--T", type=int, default=100, help="Target cycle for fixed_T mode.")
    ap.add_argument("--max_input_cycle", type=int, default=50, help="Maximum cycle allowed in input features.")
    ap.add_argument("--future_k", type=int, default=20, help="Future horizon K for future_delta_TK mode.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for grouped train/test split and models.")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test split ratio at the cell_key group level.")
    ap.add_argument("--min_rows_per_group", type=int, default=6, help="Minimum rows required to train one group.")
    ap.add_argument("--min_cells_per_group", type=int, default=4, help="Minimum unique cells required to train one group.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.table_csv)
    if df.empty:
        raise ValueError("Input table is empty.")

    if args.target_mode == "fixed_T":
        data = pick_anchor_rows_fixed_T(df, T=args.T, max_input_cycle=args.max_input_cycle)
        mode_tag = f"fixedT_{args.T}"
    else:
        data = pick_rows_future_delta_TK(df, max_input_cycle=args.max_input_cycle)
        mode_tag = f"futureK_{args.future_k}"

    if data.empty:
        raise ValueError("No training rows after target-mode filtering.")

    label_col = "target_abs" if args.label_mode == "absolute" else "target_delta"

    # Numeric feature columns only, prefixed by feat_.
    feature_cols = [
        c for c in data.columns
        if c.startswith("feat_") and pd.api.types.is_numeric_dtype(data[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")

    summary_rows: List[Dict] = []
    pred_rows_all: List[Dict] = []
    for group in ["CL", "FLC", "HYCL"]:
        dg = data[data["group_tag"] == group].copy()
        recs, pred_rows = fit_eval_one_group(
            df_group=dg,
            feature_cols=feature_cols,
            label_col=label_col,
            seed=args.seed,
            test_size=args.test_size,
            min_rows=args.min_rows_per_group,
            min_cells=args.min_cells_per_group,
        )
        for r in recs:
            r.update(
                {
                    "group_tag": group,
                    "target_mode": args.target_mode,
                    "label_mode": args.label_mode,
                    "mode_tag": mode_tag,
                    "max_input_cycle": int(args.max_input_cycle),
                    "feature_count": int(len(feature_cols)),
                }
            )
        summary_rows.extend(recs)
        for p in pred_rows:
            p.update(
                {
                    "target_mode": args.target_mode,
                    "label_mode": args.label_mode,
                    "mode_tag": mode_tag,
                    "max_input_cycle": int(args.max_input_cycle),
                }
            )
        pred_rows_all.extend(pred_rows)

    if not summary_rows:
        raise ValueError("No valid group/model results. Check sample sizes per group.")

    res = pd.DataFrame(summary_rows).sort_values(["group_tag", "rmse"]).reset_index(drop=True)

    res_csv = out_dir / f"results__{args.target_mode}__{args.label_mode}__{mode_tag}.csv"
    res.to_csv(res_csv, index=False)

    pred_csv = out_dir / f"predictions__{args.target_mode}__{args.label_mode}__{mode_tag}.csv"
    pd.DataFrame(pred_rows_all).to_csv(pred_csv, index=False)

    run_meta = {
        "table_csv": str(args.table_csv),
        "target_mode": args.target_mode,
        "label_mode": args.label_mode,
        "T": int(args.T),
        "future_k": int(args.future_k),
        "max_input_cycle": int(args.max_input_cycle),
        "seed": int(args.seed),
        "test_size": float(args.test_size),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
    }
    meta_json = out_dir / f"run_meta__{args.target_mode}__{args.label_mode}__{mode_tag}.json"
    meta_json.write_text(json.dumps(run_meta, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"[INFO] Saved results: {res_csv}")
    print(f"[INFO] Saved predictions: {pred_csv}")
    print(f"[INFO] Saved run meta: {meta_json}")
    print(res)


if __name__ == "__main__":
    main()
