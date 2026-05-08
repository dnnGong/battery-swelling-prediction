#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)


def safe_float(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def cpe_tau_approx(r_ohm: float, q: float, alpha: float) -> float:
    if not (math.isfinite(r_ohm) and math.isfinite(q) and math.isfinite(alpha)):
        return float("nan")
    if r_ohm <= 0 or q <= 0 or alpha <= 0:
        return float("nan")
    try:
        return float((r_ohm * q) ** (1.0 / alpha))
    except Exception:
        return float("nan")


def bound_triplet(v: float, buffer_frac: float) -> Dict[str, float]:
    if not math.isfinite(v):
        return {"init": float("nan"), "lb": float("nan"), "ub": float("nan")}
    lb = max(0.0, v * (1.0 - buffer_frac))
    ub = v * (1.0 + buffer_frac)
    return {"init": float(v), "lb": float(lb), "ub": float(ub)}


def parse_fit_result(path: Path, buffer_frac: float) -> Optional[Dict[str, float]]:
    try:
        fit = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    params = fit.get("params", [])
    circuit = str(fit.get("circuit", ""))
    serial = str(fit.get("serial", "")).strip().upper()
    measurement_cycle = safe_float(fit.get("measurement_cycle"))
    file_name = str(fit.get("file_name", ""))
    sheet = str(fit.get("sheet", ""))
    rmse = safe_float(fit.get("rmse_complex_ohm"))

    row: Dict[str, float] = {
        "source_fit_result": str(path),
        "file_name": file_name,
        "sheet": sheet,
        "serial": serial,
        "serial_norm": serial,
        "measurement_cycle": measurement_cycle,
        "circuit": circuit,
        "rmse_complex_ohm": rmse,
    }

    circuit_l = circuit.lower()
    if circuit_l.startswith("r0-p(r1,cpe1)-p(r2,cpe2)") and len(params) >= 7:
        rs = safe_float(params[0])
        rsei = safe_float(params[1])
        qsei = safe_float(params[2])
        nsei = safe_float(params[3])
        rdl = safe_float(params[4])
        qdl = safe_float(params[5])
        ndl = safe_float(params[6])
        sigma = safe_float(params[7]) if len(params) >= 8 else float("nan")
        warburg_tau = safe_float(params[8]) if len(params) >= 9 else float("nan")

        row.update(
            {
                "prior_Rs_ohm": rs,
                "prior_Rsei_ohm": rsei,
                "prior_Qsei": qsei,
                "prior_nsei": nsei,
                "prior_Rdl_ohm": rdl,
                "prior_Qdl": qdl,
                "prior_ndl": ndl,
                "prior_sigma": sigma,
                "prior_warburg_tau": warburg_tau,
                "prior_R_total_ohm": (
                    float(rs + rsei + rdl)
                    if math.isfinite(rs) and math.isfinite(rsei) and math.isfinite(rdl)
                    else float("nan")
                ),
            }
        )

        tau_sei = cpe_tau_approx(rsei, qsei, nsei)
        tau_dl = cpe_tau_approx(rdl, qdl, ndl)
        row["prior_tau_sei_s"] = tau_sei
        row["prior_tau_dl_s"] = tau_dl

        for src_col, prefix in [
            ("prior_Rs_ohm", "prior_Rs"),
            ("prior_Rsei_ohm", "prior_Rsei"),
            ("prior_Rdl_ohm", "prior_Rdl"),
            ("prior_tau_sei_s", "prior_tau_sei"),
            ("prior_tau_dl_s", "prior_tau_dl"),
            ("prior_R_total_ohm", "prior_R_total"),
        ]:
            triplet = bound_triplet(safe_float(row.get(src_col)), buffer_frac=buffer_frac)
            row[f"{prefix}_init"] = triplet["init"]
            row[f"{prefix}_lb"] = triplet["lb"]
            row[f"{prefix}_ub"] = triplet["ub"]

        if math.isfinite(sigma):
            row["prior_sigma_init"] = sigma
            row["prior_sigma_lb"] = 0.0
            row["prior_sigma_ub"] = sigma * (1.0 + max(buffer_frac, 0.2))
        else:
            row["prior_sigma_init"] = float("nan")
            row["prior_sigma_lb"] = float("nan")
            row["prior_sigma_ub"] = float("nan")
        return row

    row["unsupported_circuit"] = circuit
    return row


def collect_prior_rows(ecm_dir: Path, buffer_frac: float) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for p in sorted(ecm_dir.rglob("fit_result__*.json")):
        row = parse_fit_result(p, buffer_frac=buffer_frac)
        if row is not None:
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a frequency-domain prior table for the planned time-domain ECM fitter."
    )
    ap.add_argument("--ecm_dir", required=True, help="Directory containing fit_result__*.json files from src/ecm_fit.py.")
    ap.add_argument("--out_csv", required=True, help="Output CSV of EIS-derived priors and bounds.")
    ap.add_argument(
        "--buffer_frac",
        type=float,
        default=0.05,
        help="Relative buffer added around EIS-derived values to form default lower/upper bounds.",
    )
    args = ap.parse_args()

    ecm_dir = Path(args.ecm_dir)
    if not ecm_dir.exists():
        raise FileNotFoundError(f"ecm_dir not found: {ecm_dir}")

    rows = collect_prior_rows(ecm_dir, buffer_frac=args.buffer_frac)
    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"[INFO] Saved EIS time-domain prior table: {out_csv}")
    print(f"[INFO] Rows={len(df)}")
    if not df.empty:
        supported = int(df["prior_Rs_ohm"].notna().sum()) if "prior_Rs_ohm" in df else 0
        print(f"[INFO] Supported rows with parsed prior_Rs_ohm: {supported}")


if __name__ == "__main__":
    main()
