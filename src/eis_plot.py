#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def lower_cell(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip().lower()


def to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)


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


def detect_primary_serial(df_raw: pd.DataFrame) -> str:
    for idx in (1, 0, 2):
        serials = detect_serials(df_raw, serial_row_idx=idx)
        if serials:
            return serials[0]
    return "unknown_serial"


def find_best_header_row(df_raw: pd.DataFrame, keywords: List[str], scan_rows: int = 150) -> Optional[int]:
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
    Pick the best matching column index by counting how many must_contain keywords appear in the header cell.
    """
    cells = header_row.astype(str).map(lower_cell)
    best_j, best_score = None, 0
    for j, cell in enumerate(cells):
        score = sum(1 for kw in must_contain if kw in cell)
        if score > best_score:
            best_score = score
            best_j = j
    return best_j if best_score > 0 else None


def pick_any_col_idx(header_row: pd.Series, any_contains: List[str]) -> Optional[int]:
    """
    Pick a column index if header contains ANY of the provided keywords (weaker matching).
    Prefer more specific (higher count) if multiple match.
    """
    cells = header_row.astype(str).map(lower_cell)
    best_j, best_score = None, 0
    for j, cell in enumerate(cells):
        score = sum(1 for kw in any_contains if kw in cell)
        if score > best_score:
            best_score = score
            best_j = j
    return best_j if best_score > 0 else None


@dataclass
class EIS:
    f: np.ndarray
    zr: np.ndarray
    zi: np.ndarray
    sheet: str
    group: Optional[np.ndarray] = None      # cycle/day id, optional
    group_name: Optional[str] = None        # column name for grouping, optional


def extract_eis(df_raw: pd.DataFrame, sheet_name: str) -> Optional[EIS]:
    """
    Extract ONE EIS block from a sheet:
    - Finds a header row containing frequency/real/imag keywords
    - Picks frequency, real, imag columns
    - Optionally picks a cycle/day grouping column if present
    """
    header_i = find_best_header_row(df_raw, ["frequency", "real", "imag"])
    if header_i is None:
        return None

    header = df_raw.iloc[header_i]

    # Required cols
    j_f = pick_col_idx(header, ["frequency"])
    j_r = pick_col_idx(header, ["real"])
    j_i = pick_col_idx(header, ["imag"])
    if j_f is None or j_r is None or j_i is None:
        return None

    # Optional grouping col (cycle/day)
    # In your UDC, common headers include:
    # "measurement day or cycle", "cycle", "day", "measurement day", "cycle/day"
    j_g = pick_any_col_idx(
        header,
        ["measurement day or cycle", "cycle/day", "cycle", "day", "measurement day", "measurement cycle"]
    )

    df = df_raw.iloc[header_i + 1:].copy()

    f = to_float_series(df.iloc[:, j_f])
    zr = to_float_series(df.iloc[:, j_r])
    zi = to_float_series(df.iloc[:, j_i])

    g = None
    g_name = None
    if j_g is not None:
        # grouping may be numeric (cycle/day index) or string; we try numeric first
        g_series = pd.to_numeric(df.iloc[:, j_g], errors="coerce")
        # If numeric works for enough rows, use it; else use string version
        if g_series.notna().sum() >= max(5, int(0.2 * len(g_series))):
            g = g_series
        else:
            g = df.iloc[:, j_g].astype(str).map(lambda x: str(x).strip())
        g_name = str(header.iloc[j_g]) if header.iloc[j_g] is not None else "cycle/day"

    ok = f.notna() & zr.notna() & zi.notna()
    if g is not None:
        ok = ok & pd.Series(g).notna()

    if ok.sum() < 5:
        return None

    f = f[ok].to_numpy(dtype=float)
    zr = zr[ok].to_numpy(dtype=float)
    zi = zi[ok].to_numpy(dtype=float)

    if np.mean(f > 0) < 0.8:
        return None

    group_arr = None
    if g is not None:
        group_arr = pd.Series(g)[ok].to_numpy()

    return EIS(f=f, zr=zr, zi=zi, sheet=sheet_name, group=group_arr, group_name=g_name)


def _iter_groups(e: EIS) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return list of (group_label, f, zr, zi) where each group is internally sorted by frequency.
    If no group, return a single pseudo-group.
    """
    if e.group is None:
        # Sort by frequency globally (prevents weird connecting when rows are unsorted)
        idx = np.argsort(e.f)
        return [("all", e.f[idx], e.zr[idx], e.zi[idx])]

    groups: Dict[str, List[int]] = {}
    for k, gv in enumerate(e.group):
        label = str(gv)
        groups.setdefault(label, []).append(k)

    out = []
    # Sort groups by numeric if possible
    def _key(x: str):
        try:
            return float(x)
        except Exception:
            return x

    for label in sorted(groups.keys(), key=_key):
        idxs = np.array(groups[label], dtype=int)
        f = e.f[idxs]
        zr = e.zr[idxs]
        zi = e.zi[idxs]
        # Sort within group by frequency
        order = np.argsort(f)
        out.append((label, f[order], zr[order], zi[order]))

    return out


def plot_nyquist(e: EIS, out_png: str, invert_imag: bool = False) -> None:
    """
    Fix: plot each cycle/day as an independent polyline to avoid connecting across cycles.
    """
    plt.figure(figsize=(9, 6))

    for label, f, zr, zi in _iter_groups(e):
        x = zr
        y = (-zi) if invert_imag else zi

        # Each group draws its own line -> no cross-group straight lines
        plt.plot(x, y, marker="o", linewidth=1, markersize=3, alpha=0.9)

    plt.xlabel("Real (Ohm or mOhm)")
    plt.ylabel(("−Imag" if invert_imag else "Imag") + " (Ohm or mOhm)")
    title = f"Nyquist - {e.sheet}"
    if e.group_name is not None:
        title += f" (grouped by {e.group_name})"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_bode(e: EIS, out_prefix: str) -> None:
    """
    Fix: plot each cycle/day separately + sort by frequency within each cycle.
    """
    # Magnitude
    plt.figure(figsize=(9, 6))
    for label, f, zr, zi in _iter_groups(e):
        mag = np.sqrt(zr * zr + zi * zi)
        plt.semilogx(f, mag, marker="o", linewidth=1, markersize=3, alpha=0.9)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|Z| (Ohm or mOhm)")
    title = f"Bode Magnitude - {e.sheet}"
    if e.group_name is not None:
        title += f" (grouped by {e.group_name})"
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_prefix + "_bode_mag.png", dpi=200)
    plt.close()

    # Phase
    plt.figure(figsize=(9, 6))
    for label, f, zr, zi in _iter_groups(e):
        phase = np.degrees(np.arctan2(zi, zr))
        plt.semilogx(f, phase, marker="o", linewidth=1, markersize=3, alpha=0.9)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (deg)")
    title = f"Bode Phase - {e.sheet}"
    if e.group_name is not None:
        title += f" (grouped by {e.group_name})"
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_prefix + "_bode_phase.png", dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Path to UDC xlsx")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument(
        "--invert-imag",
        action="store_true",
        help="Plot Nyquist with -Imag on y-axis (common convention)."
    )
    args = ap.parse_args()

    safe_mkdir(args.out)

    xl = pd.ExcelFile(args.xlsx, engine="openpyxl")
    made = 0

    for sh in xl.sheet_names:
        df_raw = pd.read_excel(args.xlsx, sheet_name=sh, header=None, engine="openpyxl")
        if df_raw.empty:
            continue

        eis = extract_eis(df_raw, sh)
        if eis is None:
            continue

        serial = detect_primary_serial(df_raw)
        serial_dir = os.path.join(args.out, sanitize_filename(serial))
        safe_mkdir(serial_dir)
        prefix = os.path.join(serial_dir, sanitize_filename(sh))
        plot_nyquist(eis, prefix + "_nyquist.png", invert_imag=args.invert_imag)
        plot_bode(eis, prefix)
        made += 3

    if made == 0:
        print("[WARN] No EIS tables found (frequency/real/imag numeric block not detected).")
    else:
        print(f"[INFO] Generated {made} plot(s) -> {args.out}/<serial>/")


if __name__ == "__main__":
    main()
