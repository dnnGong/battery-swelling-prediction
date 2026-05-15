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


def rc_tau_approx(r_ohm: float, c_f: float) -> float:
    if not (math.isfinite(r_ohm) and math.isfinite(c_f)):
        return float("nan")
    if r_ohm <= 0 or c_f <= 0:
        return float("nan")
    return float(r_ohm * c_f)


def infer_circuit_family(circuit: str, fit: Dict[str, object]) -> str:
    family = str(fit.get("circuit_family", "")).strip().lower()
    if family:
        return family
    c = str(circuit or "").strip().lower()
    if c.startswith("r0-p(r1,c1)-r2-p(r3,c2)-p(r4,c3)"):
        return "td_compatible"
    return "legacy"


def add_legacy_aliases_for_td_compatible(row: Dict[str, float]) -> Dict[str, float]:
    """
    Preserve backward-compatible prior names while exposing mentor-aligned names.
    """
    if "prior_Rsei_ohm" in row:
        row["prior_RSEI_ohm"] = row["prior_Rsei_ohm"]
    if "prior_Csei_F" in row:
        row["prior_CSEI_F"] = row["prior_Csei_F"]
    if "prior_Rct_ohm" in row:
        row["prior_RCT_ohm"] = row["prior_Rct_ohm"]
    if "prior_Rw1_ohm" in row:
        row["prior_R1_ohm"] = row["prior_Rw1_ohm"]
    if "prior_Cw1_F" in row:
        row["prior_C1_F"] = row["prior_Cw1_F"]
    if "prior_Rw2_ohm" in row:
        row["prior_R2_ohm"] = row["prior_Rw2_ohm"]
    if "prior_Cw2_F" in row:
        row["prior_C2_F"] = row["prior_Cw2_F"]
    if "prior_tau_Rsei_s" in row:
        row["prior_tau_RSEI_s"] = row["prior_tau_Rsei_s"]
    if "prior_tau_Rw1_s" in row:
        row["prior_tau_R1_s"] = row["prior_tau_Rw1_s"]
    if "prior_tau_Rw2_s" in row:
        row["prior_tau_R2_s"] = row["prior_tau_Rw2_s"]
    return row


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
        "group_tag": str(fit.get("group_tag", "")),
        "sheet": sheet,
        "serial": serial,
        "serial_norm": serial,
        "measurement_cycle": measurement_cycle,
        "circuit": circuit,
        "circuit_family": infer_circuit_family(circuit, fit),
        "rmse_complex_ohm": rmse,
    }

    circuit_l = circuit.lower()
    if row["circuit_family"] == "legacy" and circuit_l.startswith("r0-p(r1,cpe1)-p(r2,cpe2)") and len(params) >= 7:
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

    if row["circuit_family"] == "td_compatible" and circuit_l.startswith("r0-p(r1,c1)-r2-p(r3,c2)-p(r4,c3)") and len(params) >= 8:
        r0 = safe_float(params[0])
        rsei = safe_float(params[1])
        csei = safe_float(params[2])
        rct = safe_float(params[3])
        rw1 = safe_float(params[4])
        cw1 = safe_float(params[5])
        rw2 = safe_float(params[6])
        cw2 = safe_float(params[7])

        tau_sei = rc_tau_approx(rsei, csei)
        tau_w1 = rc_tau_approx(rw1, cw1)
        tau_w2 = rc_tau_approx(rw2, cw2)
        r_tail = (
            float(rw1 + rw2)
            if math.isfinite(rw1) and math.isfinite(rw2)
            else float("nan")
        )
        row.update(
            {
                "prior_R0_ohm": r0,
                "prior_Rs_ohm": r0,
                "prior_Rsei_ohm": rsei,
                "prior_Csei_F": csei,
                "prior_Rct_ohm": rct,
                "prior_Rw1_ohm": rw1,
                "prior_Cw1_F": cw1,
                "prior_Rw2_ohm": rw2,
                "prior_Cw2_F": cw2,
                "prior_tau_Rsei_s": tau_sei,
                "prior_tau_Rw1_s": tau_w1,
                "prior_tau_Rw2_s": tau_w2,
                "prior_Rdl_ohm": r_tail,
                "prior_tau_dl_s": (
                    float(np.nanmedian([tau_w1, tau_w2]))
                    if np.isfinite(tau_w1) or np.isfinite(tau_w2)
                    else float("nan")
                ),
                "prior_R_total_ohm": (
                    float(r0 + rsei + rct + rw1 + rw2)
                    if all(math.isfinite(x) for x in [r0, rsei, rct, rw1, rw2])
                    else float("nan")
                ),
                "prior_sigma": float("nan"),
                "prior_warburg_tau": float("nan"),
            }
        )
        row = add_legacy_aliases_for_td_compatible(row)

        for src_col, prefix in [
            ("prior_R0_ohm", "prior_R0"),
            ("prior_Rs_ohm", "prior_Rs"),
            ("prior_Rsei_ohm", "prior_Rsei"),
            ("prior_Csei_F", "prior_Csei"),
            ("prior_Rct_ohm", "prior_Rct"),
            ("prior_Rw1_ohm", "prior_Rw1"),
            ("prior_Cw1_F", "prior_Cw1"),
            ("prior_Rw2_ohm", "prior_Rw2"),
            ("prior_Cw2_F", "prior_Cw2"),
            ("prior_Rdl_ohm", "prior_Rdl"),
            ("prior_tau_Rsei_s", "prior_tau_Rsei"),
            ("prior_tau_Rw1_s", "prior_tau_Rw1"),
            ("prior_tau_Rw2_s", "prior_tau_Rw2"),
            ("prior_tau_dl_s", "prior_tau_dl"),
            ("prior_R_total_ohm", "prior_R_total"),
        ]:
            triplet = bound_triplet(safe_float(row.get(src_col)), buffer_frac=buffer_frac)
            row[f"{prefix}_init"] = triplet["init"]
            row[f"{prefix}_lb"] = triplet["lb"]
            row[f"{prefix}_ub"] = triplet["ub"]

        # Backward-compatible aliases for older downstream code and prior drafts.
        for src_prefix, legacy_prefix in [
            ("prior_Rsei", "prior_RSEI"),
            ("prior_Csei", "prior_CSEI"),
            ("prior_Rct", "prior_RCT"),
            ("prior_Rw1", "prior_R1"),
            ("prior_Cw1", "prior_C1"),
            ("prior_Rw2", "prior_R2"),
            ("prior_Cw2", "prior_C2"),
            ("prior_tau_Rsei", "prior_tau_RSEI"),
            ("prior_tau_Rw1", "prior_tau_R1"),
            ("prior_tau_Rw2", "prior_tau_R2"),
        ]:
            for suf in ["_init", "_lb", "_ub"]:
                row[f"{legacy_prefix}{suf}"] = row.get(f"{src_prefix}{suf}", float("nan"))

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
