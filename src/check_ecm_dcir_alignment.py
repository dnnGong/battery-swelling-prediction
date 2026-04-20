#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import sys

import pandas as pd

# Ensure project root is importable when script is launched from outside repo dir.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.build_feature_table import (
    infer_group_from_path,
    is_ecm_fit_usable,
    load_ecm_results_by_cell,
    select_soc_slice,
)
from src.cycle_plot import (
    detect_serials,
    extract_from_cycledcir,
    load_sheet,
)


def _sorted_unique_ints(values) -> List[int]:
    out = []
    for v in values:
        try:
            if pd.notna(v):
                out.append(int(float(v)))
        except Exception:
            continue
    return sorted(set(out))


def _join_cycles(vals: List[int]) -> str:
    return ",".join(str(x) for x in vals)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Check exact cycle alignment between ECM measurement_cycle and DCIR cycle_target.",
    )
    ap.add_argument("--xlsx_dir", required=True, help="Root directory containing raw xlsx files.")
    ap.add_argument("--ecm_dir", required=True, help="Root directory containing ECM fit outputs.")
    ap.add_argument("--out_dir", required=True, help="Directory to save alignment reports.")
    ap.add_argument("--group_tag", default=None, help="Optional group filter: CL/FLC/HYCL.")
    ap.add_argument("--soc_target", type=float, default=50.0, help="SOC target used to select the DCIR slice.")
    ap.add_argument("--sheet", default="03-4_EIS", help="Only keep ECM records from this sheet. Use '' to disable.")
    ap.add_argument("--rmse_max", type=float, default=1.0, help="Only keep ECM fits with rmse_complex_ohm <= this threshold.")
    args = ap.parse_args()

    xlsx_dir = Path(args.xlsx_dir)
    ecm_dir = Path(args.ecm_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ecm_idx = load_ecm_results_by_cell(ecm_dir)
    overview_rows: List[Dict] = []
    pair_rows: List[Dict] = []

    for xlsx in sorted(xlsx_dir.rglob("*.xlsx")):
        group_tag = infer_group_from_path(xlsx)
        if args.group_tag and group_tag != args.group_tag:
            continue

        file_name = xlsx.name
        try:
            df_dcir = load_sheet(str(xlsx), "03-1_CycleDCIR")
        except Exception:
            continue

        serials = detect_serials(df_dcir, serial_row_idx=1)
        if not serials:
            serials = detect_serials(df_dcir, serial_row_idx=0)
        if not serials:
            serials = detect_serials(df_dcir, serial_row_idx=2)

        for serial in serials:
            try:
                dcir = extract_from_cycledcir(df_dcir, serial)
            except Exception:
                continue

            dcir_s = select_soc_slice(dcir, soc_target=args.soc_target)
            dcir_cycles = _sorted_unique_ints(dcir_s.get("cycle_target", []))
            dcir_map = {}
            if not dcir_s.empty and "cycle_target" in dcir_s.columns and "dcir_mohm" in dcir_s.columns:
                tmp = dcir_s[["cycle_target", "dcir_mohm"]].dropna().copy()
                tmp["cycle_target"] = tmp["cycle_target"].astype(float).round().astype(int)
                tmp = tmp.sort_values("cycle_target").drop_duplicates("cycle_target", keep="last")
                dcir_map = {int(r["cycle_target"]): float(r["dcir_mohm"]) for _, r in tmp.iterrows()}

            ecm_entries = ecm_idx.get((file_name, serial), [])
            kept_entries = []
            for rec in ecm_entries:
                fit_result = rec.get("fit_result", {})
                fit_metrics = rec.get("fit_metrics", {})
                if args.sheet and str(fit_result.get("sheet", "")) != args.sheet:
                    continue
                if not is_ecm_fit_usable(fit_result, fit_metrics):
                    continue
                rmse = float(fit_result.get("rmse_complex_ohm", float("inf")))
                if rmse > args.rmse_max:
                    continue
                if fit_result.get("measurement_cycle", None) is None:
                    continue
                kept_entries.append(fit_result)

            ecm_cycles = _sorted_unique_ints(x.get("measurement_cycle") for x in kept_entries)
            ecm_by_cycle = {}
            for fr in kept_entries:
                cyc = int(float(fr["measurement_cycle"]))
                params = fr.get("params") or []
                ecm_by_cycle[cyc] = {
                    "sheet": fr.get("sheet"),
                    "rmse_complex_ohm": float(fr.get("rmse_complex_ohm", float("nan"))),
                    "Rs_ohm": float(params[0]) if len(params) >= 1 else float("nan"),
                    "Rsei_ohm": float(params[1]) if len(params) >= 2 else float("nan"),
                    "nsei": float(params[3]) if len(params) >= 4 else float("nan"),
                    "Rdl_ohm": float(params[4]) if len(params) >= 5 else float("nan"),
                    "ndl": float(params[6]) if len(params) >= 7 else float("nan"),
                    "sigma": float(params[7]) if len(params) >= 8 else float("nan"),
                    "R_total_ohm": (
                        float(params[0] + params[1] + params[4])
                        if len(params) >= 5
                        else float("nan")
                    ),
                }

            exact_cycles = sorted(set(ecm_cycles).intersection(dcir_cycles))
            overview_rows.append(
                {
                    "group_tag": group_tag,
                    "file_name": file_name,
                    "serial": serial,
                    "cell_key": f"{file_name}::{serial}",
                    "soc_target": args.soc_target,
                    "ecm_sheet": args.sheet or "ALL",
                    "ecm_cycle_count": len(ecm_cycles),
                    "dcir_cycle_count": len(dcir_cycles),
                    "exact_match_count": len(exact_cycles),
                    "has_any_exact_match": int(len(exact_cycles) > 0),
                    "ecm_cycles": _join_cycles(ecm_cycles),
                    "dcir_cycles": _join_cycles(dcir_cycles),
                    "exact_cycles": _join_cycles(exact_cycles),
                }
            )

            for cyc in exact_cycles:
                e = ecm_by_cycle.get(cyc, {})
                pair_rows.append(
                    {
                        "group_tag": group_tag,
                        "file_name": file_name,
                        "serial": serial,
                        "cell_key": f"{file_name}::{serial}",
                        "cycle_matched": cyc,
                        "dcir_soc_target": args.soc_target,
                        "dcir_mohm": dcir_map.get(cyc),
                        "ecm_sheet": e.get("sheet"),
                        "ecm_rmse_complex_ohm": e.get("rmse_complex_ohm"),
                        "Rs_ohm": e.get("Rs_ohm"),
                        "Rsei_ohm": e.get("Rsei_ohm"),
                        "nsei": e.get("nsei"),
                        "Rdl_ohm": e.get("Rdl_ohm"),
                        "ndl": e.get("ndl"),
                        "sigma": e.get("sigma"),
                        "R_total_ohm": e.get("R_total_ohm"),
                    }
                )

    overview_df = pd.DataFrame(overview_rows)
    if not overview_df.empty:
        overview_df = overview_df.sort_values(["group_tag", "file_name", "serial"]).reset_index(drop=True)

    pairs_df = pd.DataFrame(pair_rows)
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values(["group_tag", "file_name", "serial", "cycle_matched"]).reset_index(drop=True)

    stem = "ecm_dcir_exact_alignment"
    if args.group_tag:
        stem += f"__{args.group_tag}"
    overview_csv = out_dir / f"{stem}__overview.csv"
    pairs_csv = out_dir / f"{stem}__pairs.csv"
    overview_df.to_csv(overview_csv, index=False)
    pairs_df.to_csv(pairs_csv, index=False)

    print(f"[INFO] cells_checked={len(overview_df)}")
    print(f"[INFO] cells_with_exact_match={int(overview_df['has_any_exact_match'].sum()) if not overview_df.empty else 0}")
    print(f"[INFO] exact_pairs={len(pairs_df)}")
    print(f"[INFO] saved overview: {overview_csv}")
    print(f"[INFO] saved pairs: {pairs_csv}")


if __name__ == "__main__":
    main()
