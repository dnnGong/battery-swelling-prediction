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


class TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def setup_log_tee(log_file: str) -> None:
    if not log_file:
        return
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w", encoding="utf-8")
    sys.stdout = TeeStream(sys.stdout, fh)
    sys.stderr = TeeStream(sys.stderr, fh)


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


def load_ecm_results_by_cell(ecm_dir: Path) -> Dict[Tuple[str, str], List[Dict]]:
    """
    Load all ECM fit results per (xlsx_file_name, serial), including optional
    measurement_cycle metadata and the matching fit_metrics payload.
    """
    idx: Dict[Tuple[str, str], List[Dict]] = {}
    for p in ecm_dir.rglob("fit_result__*.json"):
        try:
            fit_result = json.loads(p.read_text(encoding="utf-8"))
            key = (str(fit_result.get("file_name", "")), str(fit_result.get("serial", "")))
            if not key[0] or not key[1]:
                continue

            metrics = {}
            metrics_path = p.with_name(p.name.replace("fit_result__", "fit_metrics__"))
            if metrics_path.exists():
                try:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                except Exception:
                    metrics = {}

            rec = {
                "fit_result": fit_result,
                "fit_metrics": metrics,
                "measurement_cycle": fit_result.get("measurement_cycle", None),
                "sheet": str(fit_result.get("sheet", "")),
                "rmse_complex_ohm": float(fit_result.get("rmse_complex_ohm", np.inf)),
            }
            idx.setdefault(key, []).append(rec)
        except Exception:
            continue

    for key, recs in idx.items():
        def sort_key(rec: Dict) -> Tuple[int, float]:
            cycle = rec.get("measurement_cycle", None)
            sheet = str(rec.get("sheet", "")).lower()
            if cycle is not None and np.isfinite(cycle):
                return (1, float(cycle))
            if "preeis" in sheet:
                return (0, -1.0)
            if "posteis" in sheet:
                return (2, np.inf)
            return (3, np.inf)

        recs.sort(key=sort_key)
    return idx


def select_ecm_entry_for_cycle(ecm_entries: List[Dict], cycle_t: int) -> Tuple[Dict, Dict]:
    """
    Choose the ECM record that best matches the current cycle.

    Preference:
    1. latest measurement_cycle <= cycle_t
    2. earliest measurement_cycle > cycle_t
    3. PreEIS fallback
    4. lowest-rmse fallback among remaining records
    """
    if not ecm_entries:
        return {}, {}

    cycle_entries = []
    pre_entries = []
    other_entries = []
    for rec in ecm_entries:
        fit_result = rec.get("fit_result", {})
        fit_metrics = rec.get("fit_metrics", {})
        cyc = fit_result.get("measurement_cycle", None)
        sheet = str(fit_result.get("sheet", "")).lower()
        if cyc is not None and np.isfinite(cyc):
            cycle_entries.append((int(cyc), fit_result, fit_metrics))
        elif "preeis" in sheet:
            pre_entries.append((fit_result, fit_metrics))
        else:
            other_entries.append((fit_result, fit_metrics))

    if cycle_entries:
        cycle_entries.sort(key=lambda x: x[0])
        le = [x for x in cycle_entries if x[0] <= cycle_t]
        if le:
            _, fit_result, fit_metrics = le[-1]
            return fit_result, fit_metrics
        _, fit_result, fit_metrics = cycle_entries[0]
        return fit_result, fit_metrics

    if pre_entries:
        fit_result, fit_metrics = min(
            pre_entries,
            key=lambda x: float(x[0].get("rmse_complex_ohm", np.inf)),
        )
        return fit_result, fit_metrics

    if other_entries:
        fit_result, fit_metrics = min(
            other_entries,
            key=lambda x: float(x[0].get("rmse_complex_ohm", np.inf)),
        )
        return fit_result, fit_metrics

    return {}, {}


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


def is_ecm_fit_usable(ecm_result: Dict, ecm_metrics: Dict) -> bool:
    """
    Minimal bad-fit filter for ECM-derived features.

    The goal is not to perfectly score every fit, but to block clearly failed
    fits from entering the ML table as giant, destabilizing outliers.
    """
    if not ecm_result:
        return False

    rmse_complex = float(ecm_result.get("rmse_complex_ohm", np.nan))
    if not np.isfinite(rmse_complex):
        return False

    # Hard fail for obviously broken fits. Good fits in this project are
    # usually orders of magnitude smaller than this.
    if rmse_complex > 1.0:
        return False

    nrmse_pct = float(ecm_metrics.get("nrmse_complex_percent_of_mean_absZ", np.nan)) if ecm_metrics else float("nan")
    if np.isfinite(nrmse_pct) and nrmse_pct > 50.0:
        return False

    r2_real = float(ecm_metrics.get("r2_real", np.nan)) if ecm_metrics else float("nan")
    r2_imag = float(ecm_metrics.get("r2_imag", np.nan)) if ecm_metrics else float("nan")
    if np.isfinite(r2_real) and r2_real < 0.0:
        return False
    if np.isfinite(r2_imag) and r2_imag < 0.0:
        return False

    return True


def build_rows_for_cell(
    file_name: str,
    xlsx_stem: str,
    group_tag: str,
    serial: str,
    cyc: pd.DataFrame,
    cm: pd.DataFrame,
    dcir: pd.DataFrame,
    ecm_entries: List[Dict],
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

    for t in cycles_t:
        thk_t = numeric_last_le(cm2, "cycle_actual", "thickness2_mm", float(t))
        if not np.isfinite(thk_t):
            continue

        thk_tk = numeric_last_le(cm2, "cycle_actual", "thickness2_mm", float(t + future_k))
        has_future = np.isfinite(thk_tk) and (t + future_k > t)

        ecm_result, ecm_metrics = select_ecm_entry_for_cycle(ecm_entries, t)
        ecm_fit_ok = is_ecm_fit_usable(ecm_result, ecm_metrics)
        ecm_result_use = ecm_result if ecm_fit_ok else {}
        ecm_metrics_use = ecm_metrics if ecm_fit_ok else {}
        params = ecm_result_use.get("params", []) if ecm_result_use else []
        circuit = str(ecm_result_use.get("circuit", "")) if ecm_result_use else ""

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
            "feat_ecm_fit_ok": float(ecm_fit_ok),
            "feat_ecm_measurement_cycle": float(ecm_result_use.get("measurement_cycle", np.nan)) if ecm_result_use else float("nan"),
        }

        # ECM feature block
        row["feat_ecm_rmse_complex_ohm"] = float(ecm_result_use.get("rmse_complex_ohm", np.nan)) if ecm_result_use else float("nan")
        row["feat_ecm_circuit"] = circuit
        for i, v in enumerate(params):
            row[f"feat_ecm_param_{i}"] = float(v)

        # Add human-readable aliases for the common 2-CPE battery circuit.
        # Current project mostly uses:
        #   R0-p(R1,CPE1)-p(R2,CPE2)           -> 7 params
        #   R0-p(R1,CPE1)-p(R2,CPE2)-W1/Wo1... -> Warburg tail appended
        circuit_l = circuit.lower()
        if circuit_l.startswith("r0-p(r1,cpe1)-p(r2,cpe2)") and len(params) >= 7:
            row["feat_Rs_ohm"] = float(params[0])
            row["feat_Rsei_ohm"] = float(params[1])
            row["feat_Qsei"] = float(params[2])
            row["feat_nsei"] = float(params[3])
            row["feat_Rdl_ohm"] = float(params[4])
            row["feat_Qdl"] = float(params[5])
            row["feat_ndl"] = float(params[6])
            row["feat_R_total_ohm"] = float(params[0] + params[1] + params[4])
            # Semi-infinite or finite-length Warburg tail parameter(s), if present.
            if len(params) >= 8:
                row["feat_sigma"] = float(params[7])
            if len(params) >= 9:
                row["feat_warburg_tau"] = float(params[8])

        # Fit-quality metrics
        for k, v in (ecm_metrics_use or {}).items():
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
    ecm_idx = load_ecm_results_by_cell(ecm_dir)
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
            ecm_entries = ecm_idx.get(ecm_key, [])

            rows = build_rows_for_cell(
                file_name=file_name,
                xlsx_stem=xlsx_stem,
                group_tag=group_tag,
                serial=serial,
                cyc=cyc,
                cm=cm,
                dcir=dcir,
                ecm_entries=ecm_entries,
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
    ap = argparse.ArgumentParser(
        description=(
            "Build a unified ML feature table by aligning raw xlsx cycle data with "
            "ECM fitting outputs."
        ),
        epilog=(
            "Example:\n"
            "  python src/build_feature_table.py --xlsx_dir ./dataset/OneDrive_1_2-20-2026 "
            "--ecm_dir ./data/test_ecm_all4 --out_csv ./data/ml/feature_table.csv "
            "--min_cycle 5 --max_cycle 200 --future_k 20 --soc_target 50\n\n"
            "The output CSV contains:\n"
            "  - feat_* columns for ML inputs\n"
            "  - y_* columns for thickness targets\n"
            "  - metadata columns like group_tag, serial, cycle_t"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--xlsx_dir", required=True, help="Root directory containing raw xlsx files.")
    ap.add_argument("--ecm_dir", required=True, help="Root directory containing ecm_fit outputs.")
    ap.add_argument("--out_csv", required=True, help="Output feature table CSV path.")
    ap.add_argument("--min_cycle", type=int, default=5, help="Minimum cycle to keep when building samples.")
    ap.add_argument("--max_cycle", type=int, default=200, help="Maximum cycle to keep when building samples.")
    ap.add_argument("--future_k", type=int, default=20, help="Future horizon K used for y_future_* target columns.")
    ap.add_argument("--soc_target", type=float, default=50.0, help="SOC target used to select the DCIR feature slice.")
    ap.add_argument("--log_file", default="", help="Optional path to save a copy of stdout/stderr logs.")
    args = ap.parse_args()
    setup_log_tee(args.log_file)

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
