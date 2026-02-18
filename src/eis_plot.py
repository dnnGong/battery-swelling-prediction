#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------

def safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def norm_cell(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()

def lower_cell(x) -> str:
    return norm_cell(x).lower()

def to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def find_best_header_row(df_raw: pd.DataFrame, keywords: List[str], scan_rows: int = 120) -> Optional[int]:
    """
    在前 scan_rows 行中找最像“表头”的行：包含 keywords 的命中数最多。
    """
    best_i, best_score = None, 0
    n = min(len(df_raw), scan_rows)
    for i in range(n):
        row = df_raw.iloc[i].astype(str).map(lower_cell).tolist()
        score = sum(1 for kw in keywords if any(kw in c for c in row))
        if score > best_score:
            best_score = score
            best_i = i
    return best_i if best_score > 0 else None

def pick_col_idx(header_row: pd.Series, must_contain: List[str]) -> Optional[int]:
    """
    从 header_row 里挑一个列索引：包含 must_contain 中尽可能多的关键词。
    """
    cells = header_row.astype(str).map(lower_cell)
    best_j, best_score = None, 0
    for j, cell in enumerate(cells):
        score = sum(1 for kw in must_contain if kw in cell)
        if score > best_score:
            best_score = score
            best_j = j
    return best_j if best_score > 0 else None

def sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)


# -----------------------------
# EIS parsing + plotting
# -----------------------------

@dataclass
class EISTable:
    f_hz: np.ndarray
    z_real: np.ndarray
    z_imag: np.ndarray
    sheet: str
    meta: Dict[str, str]

def extract_eis_from_sheet(df_raw: pd.DataFrame, sheet_name: str) -> List[EISTable]:
    """
    尝试在一个 sheet 中提取 EIS 表（Frequency + Real + Imag）。
    适配 UDC 常见的“多行表头/合并单元格”情况：
      1) 先定位表头行（包含 frequency/real/imag）
      2) 再确定三列索引
      3) 从表头下一行开始抓取连续的数值行
    """
    tables: List[EISTable] = []

    header_i = find_best_header_row(df_raw, ["frequency", "real", "imag"])
    if header_i is None:
        return tables

    header_row = df_raw.iloc[header_i]
    j_f = pick_col_idx(header_row, ["frequency"])
    j_r = pick_col_idx(header_row, ["real"])
    j_i = pick_col_idx(header_row, ["imag"])

    if j_f is None or j_r is None or j_i is None:
        return tables

    df = df_raw.iloc[header_i + 1:].copy()
    f = to_float_series(df.iloc[:, j_f])
    zr = to_float_series(df.iloc[:, j_r])
    zi = to_float_series(df.iloc[:, j_i])

    ok = f.notna() & zr.notna() & zi.notna()
    if ok.sum() < 5:
        return tables

    f = f[ok].to_numpy(dtype=float)
    zr = zr[ok].to_numpy(dtype=float)
    zi = zi[ok].to_numpy(dtype=float)

    # 频率应大多数为正
    if np.mean(f > 0) < 0.8:
        return tables

    tables.append(
        EISTable(
            f_hz=f,
            z_real=zr,
            z_imag=zi,
            sheet=sheet_name,
            meta={"header_row": str(header_i), "col_f": str(j_f), "col_real": str(j_r), "col_imag": str(j_i)},
        )
    )
    return tables

def plot_nyquist(t: EISTable, out_png: str, invert_imag: bool = False) -> None:
    x = t.z_real
    y = (-t.z_imag) if invert_imag else t.z_imag

    plt.figure()
    plt.plot(x, y, marker="o", linewidth=1)
    plt.xlabel("Real (Ohm or mOhm)")
    plt.ylabel("Imag (Ohm or mOhm)" + (" (negated)" if invert_imag else ""))
    plt.title(f"Nyquist - {t.sheet}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_bode(t: EISTable, out_prefix: str) -> None:
    f = t.f_hz
    zr = t.z_real
    zi = t.z_imag

    mag = np.sqrt(zr * zr + zi * zi)
    phase = np.degrees(np.arctan2(zi, zr))

    plt.figure()
    plt.semilogx(f, mag, marker="o", linewidth=1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|Z| (Ohm or mOhm)")
    plt.title(f"Bode Magnitude - {t.sheet}")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_prefix + "_bode_mag.png", dpi=200)
    plt.close()

    plt.figure()
    plt.semilogx(f, phase, marker="o", linewidth=1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (deg)")
    plt.title(f"Bode Phase - {t.sheet}")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_prefix + "_bode_phase.png", dpi=200)
    plt.close()


# -----------------------------
# Measurement-by-cycle/day detection + plotting (generic)
# -----------------------------

def detect_measurement_sheet(df_raw: pd.DataFrame) -> bool:
    # 只要能在前若干行看到这些关键词，就认为可能是 cycle/day 表
    header_i = find_best_header_row(df_raw, ["cycle", "capacity", "dcir", "acir", "ocv", "soc", "day", "date"])
    return header_i is not None

# def plot_measurement_by_cycle_day(df_raw: pd.DataFrame, sheet_name: str, out_dir: str) -> int:
#     """
#     非定制化的 measurement plot：
#       - 找表头行
#       - 找 X 轴（cycle 优先，其次 day/date）
#       - 找常见指标（capacity/dcir/acir/ocv/soc）
#       - 每个指标画一张 vs X
#     """
#     header_i = find_best_header_row(df_raw, ["cycle", "capacity", "dcir", "acir", "ocv", "soc", "day", "date"])
#     if header_i is None:
#         return 0

#     header = df_raw.iloc[header_i].astype(str).map(norm_cell)
#     df = df_raw.iloc[header_i + 1:].copy()
#     df.columns = header

#     # 选 x 轴
#     x_col = None
#     for c in df.columns:
#         if "cycle" in c.lower():
#             x_col = c
#             break
#     if x_col is None:
#         for c in df.columns:
#             if "day" in c.lower() or "date" in c.lower():
#                 x_col = c
#                 break
#     if x_col is None:
#         return 0

#     x = pd.to_numeric(df[x_col], errors="coerce")
#     if x.notna().sum() < 5:
#         return 0

#     metrics = []
#     for key in ["capacity", "dcir", "acir", "ocv", "soc"]:
#         for c in df.columns:
#             if key in c.lower():
#                 metrics.append(c)

#     made = 0
#     for m in metrics:
#         y = pd.to_numeric(df[m], errors="coerce")
#         ok = x.notna() & y.notna()
#         if ok.sum() < 5:
#             continue

#         plt.figure()
#         plt.plot(x[ok], y[ok], marker="o", linewidth=1)
#         plt.xlabel(x_col)
#         plt.ylabel(m)
#         plt.title(f"{m} vs {x_col} - {sheet_name}")
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()

#         out_png = os.path.join(out_dir, f"{sanitize_filename(sheet_name)}__{sanitize_filename(m)}_vs_{sanitize_filename(x_col)}.png")
#         plt.savefig(out_png, dpi=200)
#         plt.close()
#         made += 1

#     return made




def make_unique_columns(cols):
    """
    把重复列名变成唯一：例如 Capacity, Capacity -> Capacity, Capacity__2
    """
    seen = {}
    out = []
    for c in cols:
        c = str(c).strip()
        if c == "" or c.lower() == "nan":
            c = "Unnamed"
        if c not in seen:
            seen[c] = 1
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
    return out


def plot_measurement_by_cycle_day(df_raw: pd.DataFrame, sheet_name: str, out_dir: str) -> int:
    header_i = find_best_header_row(df_raw, ["cycle", "capacity", "dcir", "acir", "ocv", "soc", "day", "date"])
    if header_i is None:
        return 0

    header = df_raw.iloc[header_i].astype(str).map(norm_cell)
    df = df_raw.iloc[header_i + 1:].copy()

    # ✅ 关键修复：列名唯一化（避免 df["Capacity"] 返回 DataFrame）
    df.columns = make_unique_columns(header.tolist())

    # -------------------------
    # 选 x 轴（cycle 优先，其次 day/date）
    # -------------------------
    x_col = None
    for c in df.columns:
        if "cycle" in c.lower():
            x_col = c
            break
    if x_col is None:
        for c in df.columns:
            if "day" in c.lower() or "date" in c.lower():
                x_col = c
                break
    if x_col is None:
        return 0

    x = pd.to_numeric(df[x_col], errors="coerce")
    if x.notna().sum() < 5:
        return 0

    # -------------------------
    # 找常见指标列（可能有多个 DUT 的同类列）
    # -------------------------
    metric_keys = ["capacity", "dcir", "acir", "ocv", "soc"]
    metrics = []
    for key in metric_keys:
        for c in df.columns:
            if key in c.lower():
                metrics.append(c)

    made = 0
    for m in metrics:
        # ✅ 现在 df[m] 一定是 Series，因为列名已唯一
        y = pd.to_numeric(df[m], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() < 5:
            continue

        plt.figure()
        plt.plot(x[ok], y[ok], marker="o", linewidth=1)
        plt.xlabel(x_col)
        plt.ylabel(m)
        plt.title(f"{m} vs {x_col} - {sheet_name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_png = os.path.join(
            out_dir,
            f"{sanitize_filename(sheet_name)}__{sanitize_filename(m)}_vs_{sanitize_filename(x_col)}.png"
        )
        plt.savefig(out_png, dpi=200)
        plt.close()
        made += 1

    return made






# -----------------------------
# Main
# -----------------------------

def process_file(xlsx_path: str, out_dir: str) -> None:
    safe_mkdir(out_dir)

    print(f"[INFO] Reading: {xlsx_path}")
    xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
    print(f"[INFO] Sheets: {xl.sheet_names}")

    total_plots = 0
    detected_any_eis = False
    detected_any_meas = False

    for sh in xl.sheet_names:
        df_raw = pd.read_excel(xlsx_path, sheet_name=sh, header=None, engine="openpyxl")
        if df_raw.empty:
            continue

        # 1) 优先尝试 EIS
        eis_tables = extract_eis_from_sheet(df_raw, sh)
        if eis_tables:
            detected_any_eis = True
            for idx, t in enumerate(eis_tables):
                prefix = os.path.join(out_dir, f"{sanitize_filename(sh)}__eis{idx}")
                plot_nyquist(t, prefix + "_nyquist.png", invert_imag=False)
                plot_bode(t, prefix)
                total_plots += 3
            continue

        # 2) 再尝试 measurement by cycle/day
        if detect_measurement_sheet(df_raw):
            detected_any_meas = True
            made = plot_measurement_by_cycle_day(df_raw, sh, out_dir)
            total_plots += made

    if total_plots == 0:
        print("[WARN] No plots were generated.")
        if detected_any_eis:
            print("       EIS-like header detected but numeric rows were insufficient.")
        elif detected_any_meas:
            print("       Measurement-like header detected but could not find numeric Cycle/Day series.")
        else:
            print("       This xlsx likely does not contain EIS (frequency/real/imag) or cycle/day measurement tables.")
    else:
        print(f"[INFO] Done. Generated {total_plots} plot(s) -> {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Path to UDC xlsx file")
    ap.add_argument("--out", required=True, help="Output directory for plots")
    args = ap.parse_args()
    process_file(args.xlsx, args.out)

if __name__ == "__main__":
    main()