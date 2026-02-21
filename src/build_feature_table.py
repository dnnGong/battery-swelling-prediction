#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root is importable when script is launched from outside repo dir.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse robust parsers already implemented in this project.
from src.cycle_plot import (
    load_sheet,
    detect_serials,
    extract_from_cycle,
    extract_from_cyclemeasure,
    extract_from_cycledcir,
)


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)


def infer_group_from_path(xlsx_path: Path) -> str:
    for p in xlsx_path.parents:
        n = p.name.strip().lower()
        if n in ("cl", "flc", "hycl"):
            return n.upper()
    return "UNGROUPED"


def numeric_last_le(df: pd.DataFrame, x_col: str, y_col: str, t: float) -> float:
    if x_col not in df.columns or y_col not in df.columns:
        return float("nan")
    sub = df[[x_col, y_col]].dropna()
    if sub.empty:
        return float("nan")
    sub = sub[sub[x_col] <= t]
    if sub.empty:
        return float("nan")
    sub = sub.sort_values(x_col)
    return float(sub.iloc[-1][y_col])


def slope_last_window(df: pd.DataFrame, x_col: str, y_col: str, t: float, window: float = 10.0) -> float:
    if x_col not in df.columns or y_col not in df.columns:
        return float("nan")
    sub = df[[x_col, y_col]].dropna()
    if sub.empty:
        return float("nan")
    sub = sub[(sub[x_col] <= t) & (sub[x_col] >= t - window)]
    if len(sub) < 3:
        return float("nan")
    sub = sub.sort_values(x_col)
    x = sub[x_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    try:
        coef = np.polyfit(x, y, 1)
        return float(coef[0])
    except Exception:
        return float("nan")


def load_best_ecm_results(ecm_dir: Path) -> Dict[Tuple[str, str], Dict]:
    """
    Index best ECM fit per (xlsx_file_name, serial) by smallest rmse.
    Requires fit_result__*.json produced by src/ecm_fit.py.
    """
    idx: Dict[Tuple[str, str], Dict] = {}
    for p in ecm_dir.rglob("fit_result__*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            key = (str(obj.get("file_name", "")), str(obj.get("serial", "")))
            if not key[0] or not key[1]:
                continue
            rmse = float(obj.get("rmse_complex_ohm", np.inf))
            old = idx.get(key)
            if old is None or rmse < float(old.get("rmse_complex_ohm", np.inf)):
                idx[key] = obj
        except Exception:
            continue
    return idx


def load_metrics_for_serial(ecm_dir: Path, file_name: str, serial: str) -> Dict[str, float]:
    """
    Find the best metrics json for (file_name, serial) by lowest rmse_complex_ohm.
    """
    candidates = []
    file_stem = sanitize_filename(Path(file_name).stem)
    for p in ecm_dir.rglob(f"{file_stem}/{sanitize_filename(serial)}/fit_metrics__*.json"):
        candidates.append(p)
    if not candidates:
        for p in ecm_dir.rglob(f"{sanitize_filename(serial)}/fit_metrics__*.json"):
            candidates.append(p)

    best = None
    best_rmse = np.inf
    for p in candidates:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            rmse = float(obj.get("rmse_complex_ohm", np.inf))
            if rmse < best_rmse:
                best = obj
                best_rmse = rmse
        except Exception:
            pass
    return best or {}


def select_soc_slice(dcir: pd.DataFrame, soc_target: float, tol: float = 5.0) -> pd.DataFrame:
    if dcir.empty or "soc" not in dcir.columns:
        return dcir.iloc[:0].copy()
    sub = dcir.dropna(subset=["soc"]).copy()
    sub["soc_err"] = (sub["soc"] - soc_target).abs()
    near = sub[sub["soc_err"] <= tol].copy()
    if near.empty:
        # fallback: nearest SOC bucket
        best_err = float(sub["soc_err"].min())
        near = sub[sub["soc_err"] == best_err].copy()
    return near


def build_rows_for_cell(
    file_name: str,
    xlsx_stem: str,
    group_tag: str,
    serial: str,
    cyc: pd.DataFrame,
    cm: pd.DataFrame,
    dcir: pd.DataFrame,
    ecm_result: Dict,
    ecm_metrics: Dict,
    min_cycle: int,
    max_cycle: int,
    future_k: int,
    soc_target: float,
) -> List[Dict]:
    rows: List[Dict] = []

    if cm.empty or "cycle_actual" not in cm.columns or "thickness2_mm" not in cm.columns:
        return rows

    cm2 = cm.dropna(subset=["cycle_actual", "thickness2_mm"]).sort_values("cycle_actual").copy()
    if cm2.empty:
        return rows

    baseline_thk = float(cm2.iloc[0]["thickness2_mm"])
    cycles_available = sorted({int(x) for x in cm2["cycle_actual"].dropna().astype(int).tolist()})
    cycles_t = [c for c in cycles_available if min_cycle <= c <= max_cycle]
    if not cycles_t:
        return rows

    dcir_s = select_soc_slice(dcir, soc_target=soc_target)

    # ECM params from fit result
    params = ecm_result.get("params", []) if ecm_result else []
    circuit = str(ecm_result.get("circuit", "")) if ecm_result else ""

    for t in cycles_t:
        thk_t = numeric_last_le(cm2, "cycle_actual", "thickness2_mm", float(t))
        if not np.isfinite(thk_t):
            continue

        thk_tk = numeric_last_le(cm2, "cycle_actual", "thickness2_mm", float(t + future_k))
        has_future = np.isfinite(thk_tk) and (t + future_k > t)

        row = {
            "file_name": file_name,
            "xlsx_stem": xlsx_stem,
            "group_tag": group_tag,
            "serial": serial,
            "cell_key": f"{file_name}::{serial}",
            "cycle_t": int(t),
            "future_k": int(future_k),

            # labels at t
            "y_abs_thickness_t": float(thk_t),
            "y_delta_thickness_baseline_t": float(thk_t - baseline_thk),

            # future labels for T->T+K mode
            "has_future_k": int(bool(has_future)),
            "y_future_abs_thickness_tk": float(thk_tk) if has_future else float("nan"),
            "y_future_delta_thickness_tk": float(thk_tk - thk_t) if has_future else float("nan"),

            # anchor info
            "baseline_thickness_mm": float(baseline_thk),

            # cycle-derived features up to t
            "feat_cycle_t": float(t),
            "feat_thickness_t": float(thk_t),
            "feat_thickness_slope_10": slope_last_window(cm2, "cycle_actual", "thickness2_mm", float(t), window=10),
            "feat_ocv_t": numeric_last_le(cm, "cycle_actual", "ocv_v", float(t)),
            "feat_acir_t": numeric_last_le(cm, "cycle_actual", "acir_mohm", float(t)),
            "feat_capacity_t": numeric_last_le(cyc, "cycle", "discharge_capacity_mAh", float(t)),
            "feat_capacity_slope_10": slope_last_window(cyc, "cycle", "discharge_capacity_mAh", float(t), window=10),
            "feat_dcir_soc_t": numeric_last_le(dcir_s, "cycle_target", "dcir_mohm", float(t)),
            "feat_dcir_soc_slope_10": slope_last_window(dcir_s, "cycle_target", "dcir_mohm", float(t), window=10),
        }

        # ECM feature block
        row["feat_ecm_rmse_complex_ohm"] = float(ecm_result.get("rmse_complex_ohm", np.nan)) if ecm_result else float("nan")
        row["feat_ecm_circuit"] = circuit
        for i, v in enumerate(params):
            row[f"feat_ecm_param_{i}"] = float(v)

        # Fit-quality metrics
        for k, v in (ecm_metrics or {}).items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                row[f"feat_fit_{k}"] = float(v)

        rows.append(row)

    return rows


def build_table(
    xlsx_dir: Path,
    ecm_dir: Path,
    min_cycle: int,
    max_cycle: int,
    future_k: int,
    soc_target: float,
) -> pd.DataFrame:
    ecm_idx = load_best_ecm_results(ecm_dir)
    all_rows: List[Dict] = []

    xlsx_files = sorted(xlsx_dir.rglob("*.xlsx"))
    for xlsx in xlsx_files:
        group_tag = infer_group_from_path(xlsx)
        file_name = xlsx.name
        xlsx_stem = sanitize_filename(xlsx.stem)

        try:
            df_cycle = load_sheet(str(xlsx), "03-1_Cycle")
            df_cm = load_sheet(str(xlsx), "03-1_CycleMeasure")
            df_dcir = load_sheet(str(xlsx), "03-1_CycleDCIR")
        except Exception:
            continue

        serials = detect_serials(df_cycle, serial_row_idx=1)
        if not serials:
            serials = detect_serials(df_cm, serial_row_idx=1)
        if not serials:
            serials = detect_serials(df_dcir, serial_row_idx=1)

        for serial in serials:
            try:
                cyc = extract_from_cycle(df_cycle, serial)
                cm = extract_from_cyclemeasure(df_cm, serial)
                dcir = extract_from_cycledcir(df_dcir, serial)
            except Exception:
                continue

            ecm_key = (file_name, serial)
            ecm_result = ecm_idx.get(ecm_key, {})
            ecm_metrics = load_metrics_for_serial(ecm_dir, file_name=file_name, serial=serial)

            rows = build_rows_for_cell(
                file_name=file_name,
                xlsx_stem=xlsx_stem,
                group_tag=group_tag,
                serial=serial,
                cyc=cyc,
                cm=cm,
                dcir=dcir,
                ecm_result=ecm_result,
                ecm_metrics=ecm_metrics,
                min_cycle=min_cycle,
                max_cycle=max_cycle,
                future_k=future_k,
                soc_target=soc_target,
            )
            all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Add group one-hot style numeric features for tree/linear models.
    for g in ["CL", "FLC", "HYCL"]:
        df[f"feat_group_{g}"] = (df["group_tag"] == g).astype(float)

    # Keep deterministic order.
    df = df.sort_values(["group_tag", "file_name", "serial", "cycle_t"]).reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx_dir", required=True, help="Root directory containing xlsx files.")
    ap.add_argument("--ecm_dir", required=True, help="Root directory containing ecm_fit outputs.")
    ap.add_argument("--out_csv", required=True, help="Output feature table CSV path.")
    ap.add_argument("--min_cycle", type=int, default=5)
    ap.add_argument("--max_cycle", type=int, default=200)
    ap.add_argument("--future_k", type=int, default=20, help="K for future target columns (t->t+K).")
    ap.add_argument("--soc_target", type=float, default=50.0, help="SOC target for DCIR feature slice.")
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = build_table(
        xlsx_dir=Path(args.xlsx_dir),
        ecm_dir=Path(args.ecm_dir),
        min_cycle=args.min_cycle,
        max_cycle=args.max_cycle,
        future_k=args.future_k,
        soc_target=args.soc_target,
    )

    if df.empty:
        print("[WARN] No rows were built. Check xlsx/ecm paths and sheet availability.")
        return

    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved feature table: {out_csv}")
    print(f"[INFO] Rows={len(df)}, Cols={len(df.columns)}")
    print(f"[INFO] Groups={df['group_tag'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
