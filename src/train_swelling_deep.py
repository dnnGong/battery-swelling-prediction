#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


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


def pick_rows_current_cycle(df: pd.DataFrame, max_input_cycle: int) -> pd.DataFrame:
    sub = df[df["cycle_t"] <= max_input_cycle].copy()
    if sub.empty:
        return sub
    sub["target_abs"] = sub["y_abs_thickness_t"].astype(float)
    sub["target_delta"] = sub["y_delta_thickness_baseline_t"].astype(float)
    sub["target_cycle"] = sub["cycle_t"].astype(int)
    return sub


def train_test_group_split(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df["cell_key"].to_numpy()
    idx = np.arange(len(df))
    tr, te = next(gss.split(idx, groups=groups))
    return tr, te


def pick_feature_cols(
    data: pd.DataFrame,
    feature_prefix: str,
    feature_set: str,
    variance_top_n: int,
    custom_features: Optional[List[str]] = None,
) -> List[str]:
    feat_cols = [
        c for c in data.columns if c.startswith(feature_prefix) and pd.api.types.is_numeric_dtype(data[c])
    ]
    feat_cols = [c for c in feat_cols if data[c].nunique(dropna=True) > 1]
    if not feat_cols:
        return []
    if feature_set == "full":
        return feat_cols
    if feature_set == "variance":
        var = data[feat_cols].var(numeric_only=True).sort_values(ascending=False)
        return var.head(max(3, min(variance_top_n, len(var)))).index.tolist()
    if feature_set == "discharge":
        pats = ["cycle", "capacity", "acir", "dcir", "ocv", "thickness", "group_"]
        out = [c for c in feat_cols if any(p in c.lower() for p in pats)]
        return out if out else feat_cols
    if feature_set == "ecm":
        out = [c for c in feat_cols if ("ecm" in c.lower()) or ("fit_" in c.lower()) or ("group_" in c.lower())]
        return out if out else feat_cols
    if feature_set == "custom":
        wanted = custom_features or []
        out = [c for c in feat_cols if c in wanted]
        if len(out) < 3:
            raise ValueError("Custom feature set must contain at least 3 valid feature columns.")
        return out
    return feat_cols


def torch_seed_everything(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_torch_model(model_name: str, n_features: int, hidden_dim: int, dropout: float):
    import torch
    import torch.nn as nn

    m = model_name.lower()
    if m == "mlp":
        return nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    if m == "cnn":
        class CNNRegressor(nn.Module):
            def __init__(self, n_feat: int, h: int, d: float):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(8),
                )
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(32 * 8, h),
                    nn.ReLU(),
                    nn.Dropout(d),
                    nn.Linear(h, 1),
                )

            def forward(self, x):
                # x: [B, F]
                z = x.unsqueeze(1)  # [B,1,F]
                z = self.conv(z)
                return self.head(z)

        return CNNRegressor(n_features, hidden_dim, dropout)

    if m == "lstm":
        class LSTMRegressor(nn.Module):
            def __init__(self, h: int, d: float):
                super().__init__()
                self.lstm = nn.LSTM(input_size=1, hidden_size=h, num_layers=1, batch_first=True)
                self.head = nn.Sequential(nn.Dropout(d), nn.Linear(h, 1))

            def forward(self, x):
                # treat features as sequence length F with 1D signal per step.
                z = x.unsqueeze(-1)  # [B,F,1]
                out, _ = self.lstm(z)
                last = out[:, -1, :]
                return self.head(last)

        return LSTMRegressor(hidden_dim, dropout)

    raise ValueError(f"Unsupported deep model: {model_name}")


def fit_predict_torch(
    model_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    hidden_dim: int,
    dropout: float,
) -> np.ndarray:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    torch_seed_everything(seed)
    device = torch.device("cpu")

    model = build_torch_model(model_name=model_name, n_features=X_tr.shape[1], hidden_dim=hidden_dim, dropout=dropout)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    xtr = torch.tensor(X_tr, dtype=torch.float32)
    ytr = torch.tensor(y_tr.reshape(-1, 1), dtype=torch.float32)
    ds = TensorDataset(xtr, ytr)
    dl = DataLoader(ds, batch_size=max(8, int(batch_size)), shuffle=True)

    model.train()
    for _ in range(max(1, int(epochs))):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    model.eval()
    xte = torch.tensor(X_te, dtype=torch.float32).to(device)
    with torch.no_grad():
        yp = model(xte).cpu().numpy().reshape(-1)
    return yp


def fit_eval_one_group(
    df_group: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    model_names: List[str],
    seed: int,
    test_size: float,
    min_rows: int,
    min_cells: int,
    epochs: int,
    lr: float,
    batch_size: int,
    hidden_dim: int,
    dropout: float,
) -> Tuple[List[Dict], List[Dict]]:
    records: List[Dict] = []
    pred_rows: List[Dict] = []

    sub = df_group.dropna(subset=[label_col, "cell_key"]).copy()
    if len(sub) < min_rows or sub["cell_key"].nunique() < min_cells:
        return records, pred_rows

    valid_cols = [c for c in feature_cols if sub[c].notna().sum() > 0]
    if len(valid_cols) < 3:
        return records, pred_rows

    tr_idx, te_idx = train_test_group_split(sub, test_size=test_size, seed=seed)
    tr_df = sub.iloc[tr_idx].copy()
    te_df = sub.iloc[te_idx].copy()

    med = tr_df[valid_cols].median(numeric_only=True)
    tr_df[valid_cols] = tr_df[valid_cols].fillna(med).fillna(0.0)
    te_df[valid_cols] = te_df[valid_cols].fillna(med).fillna(0.0)

    X_tr = tr_df[valid_cols].to_numpy(dtype=float)
    X_te = te_df[valid_cols].to_numpy(dtype=float)
    y_tr = tr_df[label_col].to_numpy(dtype=float).reshape(-1)
    y_te = te_df[label_col].to_numpy(dtype=float).reshape(-1)

    # standardize from train stats
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True)
    sd = np.where(sd == 0.0, 1.0, sd)
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd

    for name in model_names:
        pred = fit_predict_torch(
            model_name=name,
            X_tr=X_tr,
            y_tr=y_tr,
            X_te=X_te,
            seed=seed,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        records.append(
            {
                "model": name.upper(),
                "n_train": int(len(X_tr)),
                "n_test": int(len(X_te)),
                "n_cells_train": int(sub.iloc[tr_idx]["cell_key"].nunique()),
                "n_cells_test": int(sub.iloc[te_idx]["cell_key"].nunique()),
                "n_features_used": int(len(valid_cols)),
                "rmse": safe_rmse(y_te, pred),
                "mae": float(np.mean(np.abs(y_te - pred))),
            }
        )
        for row, yt, yp in zip(te_df.itertuples(index=False), y_te, pred):
            pred_rows.append(
                {
                    "model": name.upper(),
                    "cell_key": row.cell_key,
                    "serial": getattr(row, "serial", ""),
                    "group_tag": getattr(row, "group_tag", ""),
                    "cycle_t": int(row.cycle_t),
                    "target_cycle": int(row.target_cycle),
                    "label_col": label_col,
                    "y_true": float(yt),
                    "y_pred": float(yp),
                    "abs_error": float(abs(yt - yp)),
                }
            )

    return records, pred_rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train deep neural networks (MLP/CNN/LSTM) for grouped swelling prediction.",
        epilog=(
            "Example:\n"
            "  python src/train_swelling_deep.py \\\n"
            "    --table_csv ./data/ml/hycl_od/feature_table_hycl_pruned.csv \\\n"
            "    --out_dir ./data/ml/hycl_od/results_deep \\\n"
            "    --target_mode fixed_T --label_mode absolute --T 100 --max_input_cycle 50 \\\n"
            "    --models mlp,cnn,lstm --feature_set variance --variance_top_n 20"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--table_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_mode", choices=["current", "fixed_T", "future_delta_TK"], required=True)
    ap.add_argument("--label_mode", choices=["absolute", "delta"], required=True)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--future_k", type=int, default=20)
    ap.add_argument("--max_input_cycle", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--min_rows_per_group", type=int, default=6)
    ap.add_argument("--min_cells_per_group", type=int, default=4)
    ap.add_argument("--feature_prefix", default="feat_")
    ap.add_argument("--feature_set", choices=["full", "variance", "discharge", "ecm", "custom"], default="full")
    ap.add_argument("--variance_top_n", type=int, default=20)
    ap.add_argument("--custom_features", default="", help="Comma list for --feature_set custom.")
    ap.add_argument("--groups", default="CL,FLC,HYCL", help="Comma list of group_tag to train.")
    ap.add_argument("--models", default="mlp,lstm", help="Comma list from: mlp,cnn,lstm")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--run_tag", default="")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError("PyTorch is required for train_swelling_deep.py. Install with: pip install torch") from e

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.table_csv)
    if df.empty:
        raise ValueError("Input table is empty.")

    if args.target_mode == "current":
        data = pick_rows_current_cycle(df, max_input_cycle=args.max_input_cycle)
        mode_tag = "current_cycle"
    elif args.target_mode == "fixed_T":
        data = pick_anchor_rows_fixed_T(df, T=args.T, max_input_cycle=args.max_input_cycle)
        mode_tag = f"fixedT_{args.T}"
    else:
        data = pick_rows_future_delta_TK(df, max_input_cycle=args.max_input_cycle)
        mode_tag = f"futureK_{args.future_k}"
    if args.run_tag:
        mode_tag = f"{mode_tag}__{args.run_tag}"
    if data.empty:
        raise ValueError("No training rows after target-mode filtering.")

    label_col = "target_abs" if args.label_mode == "absolute" else "target_delta"
    custom_features = [x.strip() for x in str(args.custom_features).split(",") if x.strip()]
    feat_cols = pick_feature_cols(
        data=data,
        feature_prefix=args.feature_prefix,
        feature_set=args.feature_set,
        variance_top_n=args.variance_top_n,
        custom_features=custom_features,
    )
    if not feat_cols:
        raise ValueError("No usable feature columns found.")

    model_names = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    allowed = {"mlp", "cnn", "lstm"}
    model_names = [m for m in model_names if m in allowed]
    if not model_names:
        raise ValueError("No valid deep models selected. Use --models from: mlp,cnn,lstm")
    groups = [g.strip() for g in str(args.groups).split(",") if g.strip()]
    if not groups:
        raise ValueError("No valid groups selected. Use --groups, e.g. HYCL or CL,FLC,HYCL")

    summary_rows: List[Dict] = []
    pred_rows_all: List[Dict] = []
    for group in groups:
        dg = data[data["group_tag"] == group].copy()
        recs, pred_rows = fit_eval_one_group(
            df_group=dg,
            feature_cols=feat_cols,
            label_col=label_col,
            model_names=model_names,
            seed=args.seed,
            test_size=args.test_size,
            min_rows=args.min_rows_per_group,
            min_cells=args.min_cells_per_group,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
        for r in recs:
            r.update(
                {
                    "group_tag": group,
                    "target_mode": args.target_mode,
                    "label_mode": args.label_mode,
                    "mode_tag": mode_tag,
                    "max_input_cycle": int(args.max_input_cycle),
                    "feature_count": int(len(feat_cols)),
                    "feature_set": args.feature_set,
                    "deep_models": ",".join(model_names),
                    "groups": ",".join(groups),
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
                    "feature_set": args.feature_set,
                    "deep_models": ",".join(model_names),
                    "groups": ",".join(groups),
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
        "feature_count": int(len(feat_cols)),
        "feature_columns": feat_cols,
        "feature_set": args.feature_set,
        "custom_features": custom_features,
        "groups": groups,
        "models": model_names,
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "run_tag": args.run_tag,
    }
    meta_json = out_dir / f"run_meta__{args.target_mode}__{args.label_mode}__{mode_tag}.json"
    meta_json.write_text(json.dumps(run_meta, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"[INFO] Saved results: {res_csv}")
    print(f"[INFO] Saved predictions: {pred_csv}")
    print(f"[INFO] Saved run meta: {meta_json}")
    print(res)


if __name__ == "__main__":
    main()
