#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utils
# -----------------------------
def safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def lower_cell(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip().lower()

def sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def detect_serials(df_raw: pd.DataFrame, serial_row_idx: int = 1) -> List[str]:
    """
    Detect serial numbers from the repeated-serial row in wide tables.
    """
    if df_raw is None or df_raw.empty or serial_row_idx >= len(df_raw):
        return []

    serials: List[str] = []
    seen = set()
    row = df_raw.iloc[serial_row_idx]

    for v in row.tolist():
        s = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v).strip()
        if not s:
            continue
        s_low = s.lower()
        if "serial" in s_low and "number" in s_low:
            continue

        # Keep alnum/hyphen chunks that look like SNs, e.g. 2AM143681331
        tokens = re.findall(r"[A-Za-z0-9\-]{6,}", s)
        for t in tokens:
            has_alpha = any(c.isalpha() for c in t)
            has_digit = any(c.isdigit() for c in t)
            if not (has_alpha and has_digit):
                continue
            if t in seen:
                continue
            seen.add(t)
            serials.append(t)

    return serials


def find_serial_block(df_raw: pd.DataFrame, serial: str, serial_row_idx: int = 1) -> Optional[Tuple[int, int]]:
    """
    Row serial_row_idx contains repeated serial numbers across a column block.
    Return (start_col, end_col_exclusive) for the requested serial.
    """
    row = df_raw.iloc[serial_row_idx]

    # ✅ 把每个 cell 都安全转成字符串（NaN/float 也不会炸）
    hit_cols = []
    for j, v in enumerate(row.tolist()):
        s = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
        if serial in s:
            hit_cols.append(j)

    if not hit_cols:
        return None

    start = min(hit_cols)

    # expand to the right until serial changes to another non-empty serial (or end)
    end = start
    ncol = df_raw.shape[1]
    for j in range(start, ncol):
        v = df_raw.iloc[serial_row_idx, j]
        s = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v).strip()
        if s == "" or serial in s:
            end = j + 1
            continue
        break

    return start, end


def find_header_row_in_block(df_raw: pd.DataFrame, start: int, end: int, must_have: List[str], scan_rows: int = 20) -> Optional[int]:
    """
    Find a header row that contains required keywords in the given column block.
    """
    max_r = min(scan_rows, len(df_raw))
    for i in range(max_r):
        cells = df_raw.iloc[i, start:end].astype(str).map(lower_cell).tolist()
        ok = all(any(k in c for c in cells) for k in must_have)
        if ok:
            return i
    return None


def pick_col_by_contains(header_row: pd.Series, start: int, end: int, patterns: List[str], forbid: Optional[List[str]] = None) -> Optional[int]:
    """
    Pick the first col index in [start,end) whose header contains all patterns (case-insensitive substring match).
    forbid: if provided, skip headers containing any forbidden substrings.
    """
    forbid = forbid or []
    for j in range(start, end):
        cell = lower_cell(header_row.iloc[j])
        if cell == "":
            continue
        if any(bad in cell for bad in forbid):
            continue
        if all(p in cell for p in patterns):
            return j
    return None


# -----------------------------
# Loaders
# -----------------------------
def load_sheet(xlsx: str, sheet: str) -> pd.DataFrame:
    return pd.read_excel(xlsx, sheet_name=sheet, header=None, engine="openpyxl")


# -----------------------------
# Extractors
# -----------------------------
def extract_from_cycle(df_cycle: pd.DataFrame, serial: str) -> pd.DataFrame:
    """
    03-1_Cycle: need columns [Cycle, Discharge Capacity (mAh)] in the serial block.
    """
    blk = find_serial_block(df_cycle, serial)
    if blk is None:
        raise ValueError(f"[03-1_Cycle] serial not found: {serial}")
    start, end = blk

    # header row typically at row 4 for this sheet, but we detect robustly
    hdr_i = find_header_row_in_block(df_cycle, start, end, must_have=["cycle", "discharge"], scan_rows=12)
    if hdr_i is None:
        hdr_i = 4

    header = df_cycle.iloc[hdr_i]
    j_cycle = pick_col_by_contains(header, start, end, ["cycle"])
    j_cap = pick_col_by_contains(header, start, end, ["discharge", "capacity"])

    if j_cycle is None or j_cap is None:
        raise ValueError("[03-1_Cycle] Cannot locate cycle/capacity columns in block.")

    data = df_cycle.iloc[hdr_i + 1:].copy()
    x = to_num(data.iloc[:, j_cycle])
    y = to_num(data.iloc[:, j_cap])
    ok = x.notna() & y.notna()

    out = pd.DataFrame({"cycle": x[ok].astype(float), "discharge_capacity_mAh": y[ok].astype(float)})
    return out.sort_values("cycle")


def extract_from_cyclemeasure(df_cm: pd.DataFrame, serial: str) -> pd.DataFrame:
    """
    03-1_CycleMeasure: need columns [Cycle - Actual, Thickness 2 Measurement, OCV(V), ACIR(mOhm)].
    注意：同一 serial block 里有很多 '* Equipment ID'，必须排除，否则会选到文本列 -> 全 NaN -> 空图。
    """
    blk = find_serial_block(df_cm, serial)
    if blk is None:
        raise ValueError(f"[03-1_CycleMeasure] serial not found: {serial}")
    start, end = blk

    hdr_i = find_header_row_in_block(df_cm, start, end, must_have=["cycle"], scan_rows=12)
    if hdr_i is None:
        hdr_i = 4

    header = df_cm.iloc[hdr_i]

    j_cycle_actual = pick_col_by_contains(header, start, end, ["cycle", "actual"])
    if j_cycle_actual is None:
        raise ValueError("[03-1_CycleMeasure] Cannot locate 'Cycle - Actual' column in block.")

    # ✅ 排除 equipment id
    forbid = ["equipment", " id"]

    j_thk2 = pick_col_by_contains(header, start, end, ["thickness", "2", "measurement"], forbid=forbid)

    # ✅ OCV 优先找 (v)，且排除 equipment id
    j_ocv = pick_col_by_contains(header, start, end, ["ocv", "(v)"], forbid=forbid)
    if j_ocv is None:
        j_ocv = pick_col_by_contains(header, start, end, ["ocv"], forbid=forbid)

    j_acir = pick_col_by_contains(header, start, end, ["acir"], forbid=forbid)

    data = df_cm.iloc[hdr_i + 1:].copy()

    out = pd.DataFrame()
    out["cycle_actual"] = to_num(data.iloc[:, j_cycle_actual])

    if j_thk2 is not None:
        out["thickness2_mm"] = to_num(data.iloc[:, j_thk2])
    if j_ocv is not None:
        out["ocv_v"] = to_num(data.iloc[:, j_ocv])
    if j_acir is not None:
        out["acir_mohm"] = to_num(data.iloc[:, j_acir])

    out = out[out["cycle_actual"].notna()].copy()
    out["cycle_actual"] = out["cycle_actual"].astype(float)
    return out.sort_values("cycle_actual")


def extract_from_cycledcir(df_dcir: pd.DataFrame, serial: str) -> pd.DataFrame:
    """
    ✅ 关键修复：03-1_CycleDCIR 在你的 test1.xlsx 中是“宽表 + serial 分块”，不是长表。
    常见结构（与你截图一致）：
      - 左边固定列：SOC (%)、Cycle - Target（以及可能有 Frequency 之类，但这里主要取 SOC + Cycle-Target）
      - row=1(索引1) 出现 Serial Number 横向重复（用于 find_serial_block）
      - row=4(索引4) 是字段名行（包含 'DCIR (mOhm)'、'OCV (V)' 等）
    目标：返回 long-form DataFrame：soc, cycle_target, dcir_mohm, ocv_v
    """
    blk = find_serial_block(df_dcir, serial, serial_row_idx=1)
    if blk is None:
        raise ValueError(f"[03-1_CycleDCIR] serial not found (wide table): {serial}")
    start, end = blk

    # 字段名行：通常是第 4 行
    hdr_i = 4
    header = df_dcir.iloc[hdr_i]

    # 左侧固定列：在你的文件里通常是：
    # col0 = SOC (%) 或 Cycle-Target；col1 = 另一个
    # 这里我们用“关键词搜索”在整个表（不是 block）里定位 SOC 和 Cycle-Target 的列索引，更稳。
    def find_left_col_idx(needles: List[str], scan_row: int = 4) -> Optional[int]:
        row = df_dcir.iloc[scan_row].astype(str).map(lower_cell).tolist()
        for j, cell in enumerate(row):
            if all(n in cell for n in needles):
                return j
        return None

    j_soc = find_left_col_idx(["soc"])            # "SOC (%)"
    j_cycle_t = find_left_col_idx(["cycle", "target"])  # "Cycle - Target"

    if j_soc is None or j_cycle_t is None:
        raise ValueError("[03-1_CycleDCIR] cannot locate left columns SOC / Cycle - Target.")

    forbid = ["equipment", " id"]

    # 在 serial block 内定位 DCIR、OCV
    j_dcir = pick_col_by_contains(header, start, end, ["dcir"], forbid=forbid)
    j_ocv = pick_col_by_contains(header, start, end, ["ocv", "(v)"], forbid=forbid)
    if j_ocv is None:
        j_ocv = pick_col_by_contains(header, start, end, ["ocv"], forbid=forbid)

    if j_dcir is None and j_ocv is None:
        raise ValueError(f"[03-1_CycleDCIR] found serial block, but cannot find DCIR/OCV columns for serial={serial}")

    data = df_dcir.iloc[hdr_i + 1:].copy()

    out = pd.DataFrame()
    out["soc"] = to_num(data.iloc[:, j_soc])
    out["cycle_target"] = to_num(data.iloc[:, j_cycle_t])

    if j_ocv is not None:
        out["ocv_v"] = to_num(data.iloc[:, j_ocv])
    if j_dcir is not None:
        out["dcir_mohm"] = to_num(data.iloc[:, j_dcir])

    out = out[out["soc"].notna() & out["cycle_target"].notna()].copy()
    out["soc"] = out["soc"].astype(float)
    out["cycle_target"] = out["cycle_target"].astype(float)

    return out


# -----------------------------
# Plotters (match PPT intent)
# -----------------------------
def save_line(x, y, title, xlabel, ylabel, out_png):
    plt.figure()
    plt.plot(x, y, marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def save_by_soc(df: pd.DataFrame, ycol: str, title: str, xlabel: str, ylabel: str, out_png: str):
    plt.figure()
    for soc, g in df.groupby("soc"):
        g2 = g.sort_values("cycle_target")
        if ycol not in g2.columns:
            continue
        yy = g2[ycol].dropna()
        xx = g2.loc[yy.index, "cycle_target"]
        if len(xx) < 2:
            continue
        plt.plot(xx, yy, marker="o", linewidth=1, label=f"{int(soc) if float(soc).is_integer() else soc:g}")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(title="SOC (%)", ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def run_for_one_serial(
    serial: str,
    out_dir: str,
    df_cycle: pd.DataFrame,
    df_cm: pd.DataFrame,
    df_dcir: pd.DataFrame,
) -> None:
    # 1) Discharge Capacity vs Cycle
    cyc = extract_from_cycle(df_cycle, serial)
    save_line(
        cyc["cycle"], cyc["discharge_capacity_mAh"],
        title=f"Discharge Capacity vs Cycle\nSerial {serial} | CL",
        xlabel="Cycle",
        ylabel="Discharge Capacity (mAh)",
        out_png=os.path.join(out_dir, f"CL_DischargeCapacity_vs_Cycle__{sanitize_filename(serial)}.png")
    )

    # 2/3/4) Thickness2 / OCV / ACIR vs Cycle (Cycle - Actual)
    cm = extract_from_cyclemeasure(df_cm, serial)

    # Thickness2
    if "thickness2_mm" in cm.columns and cm["thickness2_mm"].notna().sum() >= 2:
        sub = cm.dropna(subset=["thickness2_mm"])
        save_line(
            sub["cycle_actual"], sub["thickness2_mm"],
            title=f"Thickness 2 Measurement vs Cycle\nSerial {serial} | CL",
            xlabel="Cycle - Actual",
            ylabel="Thickness 2 Measurement (mm)",
            out_png=os.path.join(out_dir, f"CL_Thickness2_vs_Cycle__{sanitize_filename(serial)}.png")
        )
    else:
        print(f"[WARN] Thickness2 data not found or too sparse in 03-1_CycleMeasure for serial={serial}.")

    # OCV vs Cycle (actual)
    if "ocv_v" in cm.columns and cm["ocv_v"].notna().sum() >= 2:
        sub = cm.dropna(subset=["ocv_v"])
        save_line(
            sub["cycle_actual"], sub["ocv_v"],
            title=f"OCV vs Cycle\nSerial {serial} | CL",
            xlabel="Cycle - Actual",
            ylabel="OCV (V)",
            out_png=os.path.join(out_dir, f"CL_OCV_vs_Cycle__{sanitize_filename(serial)}.png")
        )
    else:
        print(f"[WARN] OCV(V) data not found or too sparse in 03-1_CycleMeasure for serial={serial}.")

    # ACIR vs Cycle (actual)
    if "acir_mohm" in cm.columns and cm["acir_mohm"].notna().sum() >= 2:
        sub = cm.dropna(subset=["acir_mohm"])
        save_line(
            sub["cycle_actual"], sub["acir_mohm"],
            title=f"ACIR vs Cycle\nSerial {serial} | CL",
            xlabel="Cycle - Actual",
            ylabel="ACIR (mOhm)",
            out_png=os.path.join(out_dir, f"CL_ACIR_vs_Cycle__{sanitize_filename(serial)}.png")
        )
    else:
        print(f"[WARN] ACIR(mOhm) data not found or too sparse in 03-1_CycleMeasure for serial={serial}.")

    # 5/6) DCIR & OCV vs Cycle (by SOC) from 03-1_CycleDCIR（✅宽表解析）
    dcir = extract_from_cycledcir(df_dcir, serial)

    if "dcir_mohm" in dcir.columns and dcir["dcir_mohm"].notna().sum() >= 2:
        save_by_soc(
            dcir, "dcir_mohm",
            title=f"DCIR vs Cycle (by SOC)\nSerial {serial} | CL",
            xlabel="Cycle - Target",
            ylabel="DCIR (mOhm)",
            out_png=os.path.join(out_dir, f"CL_DCIR_vs_Cycle_by_SOC__{sanitize_filename(serial)}.png")
        )
    else:
        print(f"[WARN] DCIR data not found or too sparse in 03-1_CycleDCIR for serial={serial}.")

    if "ocv_v" in dcir.columns and dcir["ocv_v"].notna().sum() >= 2:
        save_by_soc(
            dcir, "ocv_v",
            title=f"OCV vs Cycle (by SOC)\nSerial {serial} | CL",
            xlabel="Cycle - Target",
            ylabel="OCV (V)",
            out_png=os.path.join(out_dir, f"CL_OCV_vs_Cycle_by_SOC__{sanitize_filename(serial)}.png")
        )
    else:
        print(f"[WARN] OCV data not found or too sparse in 03-1_CycleDCIR for serial={serial}.")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Generate cycle-related plots from CL UDC Excel files, including "
            "capacity, thickness, OCV, ACIR, and DCIR trends."
        ),
        epilog=(
            "Example:\n"
            "  python src/cycle_plot.py --xlsx ./dataset/CL-TC1.xlsx --out ./data/test_cycle\n\n"
            "Outputs are written under:\n"
            "  <out>/<serial>/CL_DischargeCapacity_vs_Cycle__<serial>.png\n"
            "  <out>/<serial>/CL_Thickness2_vs_Cycle__<serial>.png\n"
            "  <out>/<serial>/CL_OCV_vs_Cycle__<serial>.png\n"
            "  <out>/<serial>/CL_ACIR_vs_Cycle__<serial>.png\n"
            "  <out>/<serial>/CL_DCIR_vs_Cycle_by_SOC__<serial>.png"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--xlsx", required=True, help="Input CL UDC xlsx file path.")
    ap.add_argument(
        "--serial",
        required=False,
        help="Only process one serial. If omitted, the script auto-detects and processes all serials in the workbook.",
    )
    ap.add_argument("--out", required=True, help="Output root directory. Each serial gets its own subfolder.")
    args = ap.parse_args()

    safe_mkdir(args.out)
    xlsx = args.xlsx

    # Load sheets
    df_cycle = load_sheet(xlsx, "03-1_Cycle")
    df_cm = load_sheet(xlsx, "03-1_CycleMeasure")
    df_dcir = load_sheet(xlsx, "03-1_CycleDCIR")

    if args.serial:
        serials = [args.serial]
    else:
        serials = detect_serials(df_cycle, serial_row_idx=1)
        if not serials:
            serials = detect_serials(df_cm, serial_row_idx=1)
        if not serials:
            serials = detect_serials(df_dcir, serial_row_idx=1)
        if not serials:
            raise ValueError("Cannot auto-detect serial numbers from workbook. Please pass --serial explicitly.")
        print(f"[INFO] Auto-detected {len(serials)} serial(s): {', '.join(serials)}")

    ok_cnt = 0
    fail_cnt = 0
    for serial in serials:
        try:
            serial_out = os.path.join(args.out, sanitize_filename(serial))
            safe_mkdir(serial_out)
            print(f"[INFO] Processing serial={serial} -> {serial_out}")
            run_for_one_serial(serial, serial_out, df_cycle, df_cm, df_dcir)
            ok_cnt += 1
        except Exception as e:
            fail_cnt += 1
            print(f"[WARN] serial={serial} failed: {e}")

    print(f"[INFO] Done. Output -> {args.out} | success={ok_cnt}, failed={fail_cnt}")


if __name__ == "__main__":
    main()
