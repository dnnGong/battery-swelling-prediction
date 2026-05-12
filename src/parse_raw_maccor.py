#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def infer_group_from_path(p: Path) -> str:
    for x in p.parents:
        n = x.name.strip().lower()
        if n in {"cl", "flc", "hycl"}:
            return n.upper()
    return "UNGROUPED"


def parse_duration_to_seconds(s: object) -> float:
    if s is None:
        return float("nan")
    t = str(s).strip()
    if not t:
        return float("nan")
    m = re.match(r"^\s*(?:(\d+)\s*d)?\s*(\d+):(\d+):(\d+(?:\.\d+)?)\s*$", t)
    if not m:
        return float("nan")
    d = float(m.group(1) or 0.0)
    hh = float(m.group(2))
    mm = float(m.group(3))
    ss = float(m.group(4))
    return d * 86400.0 + hh * 3600.0 + mm * 60.0 + ss


def parse_metadata(line1: str, file_name: str) -> Dict[str, str]:
    date_of_test = ""
    m_date = re.search(r"Date of Test:\s*([0-9/]+)", line1)
    if m_date:
        date_of_test = m_date.group(1).strip()

    m_file = re.search(r"Filename:\s*([^\t]+)", line1)
    maccor_filename = m_file.group(1).strip() if m_file else ""

    m_proc = re.search(r"Procedure:\s*([^\t]+)", line1)
    procedure = m_proc.group(1).strip() if m_proc else ""

    serial = ""
    m_serial = re.search(r"RDM-MS\d+_([^_]+)_", file_name)
    if m_serial:
        serial = m_serial.group(1).strip()

    phase = ""
    m_phase = re.search(r"_(PRE|POST)_", file_name, flags=re.IGNORECASE)
    if m_phase:
        phase = m_phase.group(1).upper()

    setpoint_c = ""
    m_temp = re.search(r"_([0-9]+(?:\.[0-9]+)?)_v-", file_name)
    if m_temp:
        setpoint_c = m_temp.group(1)

    return {
        "date_of_test": date_of_test,
        "maccor_filename": maccor_filename,
        "procedure": procedure,
        "serial": serial,
        "phase": phase,
        "setpoint_c": setpoint_c,
    }


def parse_one_raw_file(path: Path) -> pd.DataFrame:
    lines = path.read_text(errors="ignore").splitlines()
    if len(lines) < 3:
        return pd.DataFrame()

    meta = parse_metadata(lines[0], path.name)
    header = [x.strip() for x in lines[1].split("\t")]
    if len(header) < 5:
        return pd.DataFrame()

    rows: List[List[str]] = []
    ncol = len(header)
    for ln in lines[2:]:
        if not ln.strip():
            continue
        vals = ln.split("\t")
        if len(vals) < ncol:
            vals = vals + [""] * (ncol - len(vals))
        elif len(vals) > ncol:
            vals = vals[:ncol]
        rows.append(vals)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=header)
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Rec#": "rec_num",
        "Cycle P": "cycle_p",
        "Cycle C": "cycle_c",
        "Step": "step",
        "Test Time": "test_time",
        "Step Time": "step_time",
        "Capacity (AHr)": "capacity_ahr",
        "Energy (WHr)": "energy_whr",
        "Current (A)": "current_a",
        "Voltage (V)": "voltage_v",
        "ACImp (Ohms)": "acimp_ohm",
        "DCIR (Ohms)": "dcir_ohm",
        "EVTemp (C)": "evtemp_c",
        "EVHum (%)": "evhum_pct",
        "DPT": "dpt",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df[new] = df[old]

    for c in [
        "rec_num",
        "cycle_p",
        "cycle_c",
        "step",
        "capacity_ahr",
        "energy_whr",
        "current_a",
        "voltage_v",
        "acimp_ohm",
        "dcir_ohm",
        "evtemp_c",
        "evhum_pct",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "test_time" in df.columns:
        df["test_time_s"] = df["test_time"].map(parse_duration_to_seconds)
    else:
        df["test_time_s"] = np.nan

    df["source_file"] = path.name
    df["source_path"] = str(path)
    df["group_tag"] = infer_group_from_path(path)
    for k, v in meta.items():
        df[k] = v
    df["serial_norm"] = df["serial"].astype(str).str.strip().str.upper()
    return df


def build_cycle_summary(df_row: pd.DataFrame) -> pd.DataFrame:
    required = {"serial_norm", "group_tag", "source_file", "cycle_c"}
    missing = required - set(df_row.columns)
    if missing:
        raise ValueError(f"Missing columns for cycle summary: {sorted(missing)}")

    work = df_row.copy()
    work = work[pd.notna(work["cycle_c"])].copy()
    if work.empty:
        return pd.DataFrame()

    work["cycle_c"] = work["cycle_c"].astype(int)
    sort_cols = [c for c in ["serial_norm", "source_file", "cycle_c", "rec_num", "test_time_s"] if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols)

    keys = ["group_tag", "source_file", "serial", "serial_norm", "phase", "setpoint_c", "date_of_test", "cycle_c"]
    keys = [k for k in keys if k in work.columns]
    g = work.groupby(keys, dropna=False, observed=True)

    out = g.size().rename("n_rows").to_frame()
    for c, new_name in [
        ("test_time_s", "test_time_s_max"),
        ("voltage_v", "voltage_last"),
        ("current_a", "current_last"),
        ("capacity_ahr", "capacity_last"),
        ("energy_whr", "energy_last"),
        ("evtemp_c", "evtemp_last"),
        ("evhum_pct", "evhum_last"),
    ]:
        if c in work.columns:
            out[new_name] = g[c].last()

    for c, base in [("evtemp_c", "evtemp"), ("evhum_pct", "evhum")]:
        if c in work.columns:
            out[f"{base}_mean"] = g[c].mean()
            out[f"{base}_min"] = g[c].min()
            out[f"{base}_max"] = g[c].max()
            out[f"{base}_std"] = g[c].std()

    out = out.reset_index()
    return out


def merge_temp_features(feature_df: pd.DataFrame, cycle_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty or cycle_df.empty:
        return feature_df.copy()

    if "serial" not in feature_df.columns or "cycle_t" not in feature_df.columns:
        raise ValueError("feature_table must contain 'serial' and 'cycle_t' columns.")

    raw_use_cols = [
        "serial_norm",
        "cycle_c",
        "evtemp_last",
        "evtemp_mean",
        "evtemp_std",
        "evhum_mean",
        "evhum_std",
    ]
    raw_use_cols = [c for c in raw_use_cols if c in cycle_df.columns]
    raw = cycle_df[raw_use_cols].copy()
    raw = raw.dropna(subset=["serial_norm", "cycle_c"]).copy()
    raw["cycle_c"] = pd.to_numeric(raw["cycle_c"], errors="coerce")
    raw = raw[pd.notna(raw["cycle_c"])].copy()
    raw["cycle_c"] = raw["cycle_c"].astype(float)
    raw = raw.sort_values(["serial_norm", "cycle_c"])

    ft_all = feature_df.copy()
    ft_all["_rowid"] = np.arange(len(ft_all))
    ft_all["serial_norm"] = ft_all["serial"].astype(str).str.strip().str.upper()
    ft_all["cycle_t_num"] = pd.to_numeric(ft_all["cycle_t"], errors="coerce")

    valid = ft_all[pd.notna(ft_all["cycle_t_num"])].copy()
    invalid = ft_all[pd.isna(ft_all["cycle_t_num"])].copy()
    valid["cycle_t_num"] = valid["cycle_t_num"].astype(float)

    merged_parts: List[pd.DataFrame] = []
    for serial, sub_ft in valid.groupby("serial_norm", sort=False):
        sub_ft = sub_ft.sort_values("cycle_t_num")
        sub_raw = raw[raw["serial_norm"] == serial].sort_values("cycle_c")
        if sub_raw.empty:
            merged_parts.append(sub_ft.copy())
            continue
        m = pd.merge_asof(
            sub_ft,
            sub_raw,
            left_on="cycle_t_num",
            right_on="cycle_c",
            direction="backward",
            allow_exact_matches=True,
        )
        merged_parts.append(m)

    merged = pd.concat(merged_parts + [invalid], ignore_index=True, sort=False)

    rename = {
        "evtemp_last": "feat_raw_evtemp_t",
        "evtemp_mean": "feat_raw_evtemp_mean_t",
        "evtemp_std": "feat_raw_evtemp_std_t",
        "evhum_mean": "feat_raw_evhum_mean_t",
        "evhum_std": "feat_raw_evhum_std_t",
    }
    for old, new in rename.items():
        if old in merged.columns:
            merged[new] = merged[old]

    keep_cols = [
        c
        for c in merged.columns
        if c not in {"serial_norm", "cycle_t_num", "cycle_c", "evtemp_last", "evtemp_mean", "evtemp_std", "evhum_mean", "evhum_std"}
    ]
    merged = merged[keep_cols].sort_values("_rowid").drop(columns=["_rowid"]).reset_index(drop=True)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Parse raw Maccor text exports under dataset/raw_data and extract structured signals "
            "(including EVTemp/EVHum), with optional merge into feature_table for ML features."
        ),
        epilog=(
            "Example 1: parse raw_data only\n"
            "  python src/parse_raw_maccor.py \\\n"
            "    --raw_dir ./dataset/raw_data \\\n"
            "    --out_row_csv ./data/ml/raw_maccor_rows.csv \\\n"
            "    --out_cycle_csv ./data/ml/raw_maccor_cycle_summary.csv\n\n"
            "Example 2: also merge temperature into feature table\n"
            "  python src/parse_raw_maccor.py \\\n"
            "    --raw_dir ./dataset/raw_data \\\n"
            "    --out_row_csv ./data/ml/raw_maccor_rows.csv \\\n"
            "    --out_cycle_csv ./data/ml/raw_maccor_cycle_summary.csv \\\n"
            "    --feature_table_csv ./data/ml/feature_table.csv \\\n"
            "    --out_feature_table_csv ./data/ml/feature_table_with_raw_temp.csv"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--raw_dir", required=True, help="Root dir of raw Maccor files (text-like, tab-separated).")
    ap.add_argument("--out_row_csv", required=True, help="Output CSV for parsed row-level records.")
    ap.add_argument("--out_cycle_csv", required=True, help="Output CSV for cycle-level summary records.")
    ap.add_argument("--feature_table_csv", default="", help="Optional existing feature_table.csv to be augmented.")
    ap.add_argument("--out_feature_table_csv", default="", help="Optional output for feature table merged with raw temp features.")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")

    files = [p for p in sorted(raw_dir.rglob("*")) if p.is_file() and not p.name.startswith(".")]
    if not files:
        raise ValueError(f"No files found under: {raw_dir}")

    row_frames: List[pd.DataFrame] = []
    bad_files: List[Tuple[str, str]] = []
    for p in files:
        try:
            df = parse_one_raw_file(p)
            if not df.empty:
                row_frames.append(df)
        except Exception as e:
            bad_files.append((str(p), str(e)))

    if not row_frames:
        raise ValueError("No valid raw records parsed.")

    row_df = pd.concat(row_frames, ignore_index=True)
    cycle_df = build_cycle_summary(row_df)

    out_row = Path(args.out_row_csv)
    out_cycle = Path(args.out_cycle_csv)
    out_row.parent.mkdir(parents=True, exist_ok=True)
    out_cycle.parent.mkdir(parents=True, exist_ok=True)
    row_df.to_csv(out_row, index=False)
    cycle_df.to_csv(out_cycle, index=False)

    print(f"[INFO] Parsed files: {len(row_frames)} / {len(files)}")
    print(f"[INFO] Bad files: {len(bad_files)}")
    if bad_files:
        for f, err in bad_files[:20]:
            print(f"[WARN] {f} -> {err}")
        if len(bad_files) > 20:
            print(f"[WARN] ... {len(bad_files)-20} more bad file(s)")

    print(f"[INFO] Row-level records: {len(row_df)}")
    print(f"[INFO] Cycle-level records: {len(cycle_df)}")
    if "group_tag" in row_df.columns:
        print(f"[INFO] Row groups: {row_df['group_tag'].value_counts().to_dict()}")
    if "serial_norm" in cycle_df.columns:
        print(f"[INFO] Cycle serial count: {cycle_df['serial_norm'].nunique()}")

    print(f"[INFO] Saved row CSV: {out_row}")
    print(f"[INFO] Saved cycle CSV: {out_cycle}")

    if args.feature_table_csv or args.out_feature_table_csv:
        if not args.feature_table_csv or not args.out_feature_table_csv:
            raise ValueError("Both --feature_table_csv and --out_feature_table_csv are required for merge.")
        ft = pd.read_csv(args.feature_table_csv)
        merged = merge_temp_features(ft, cycle_df)
        out_ft = Path(args.out_feature_table_csv)
        out_ft.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_ft, index=False)

        if len(merged) and "feat_raw_evtemp_t" in merged.columns:
            coverage = float(pd.notna(merged["feat_raw_evtemp_t"]).mean())
        else:
            coverage = 0.0
        print(f"[INFO] Merged feature rows: {len(merged)}")
        print(f"[INFO] Temperature coverage (feat_raw_evtemp_t): {coverage:.2%}")
        print(f"[INFO] Saved merged feature table: {out_ft}")


if __name__ == "__main__":
    main()
