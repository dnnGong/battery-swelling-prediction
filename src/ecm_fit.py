#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
imped10_rescue.py

Robust EIS loader for wide-table Excel exports (like your test1.xlsx) when:
- Frequency column is numeric,
- But the "named" Real/Imag columns are empty/non-numeric due to export/header misalignment.

Strategy:
1) Read sheet with header=None (raw).
2) Auto-detect header row.
3) Assign block_id by encountering "Frequency (Hz)" left->right.
4) For each block:
   - Identify frequency column index j_f.
   - Build mask_f where freq is numeric and >0 (these are "real data rows").
   - Scan all columns within this block on mask_f rows and score numeric density.
   - Choose Real/Imag columns by (a) numeric density and (b) sign heuristics.
5) Choose best block by joint-valid rows (freq & real & imag).
6) Fit ECM using impedance.py CustomCircuit and save Nyquist plot.

Usage:
rye run python imped10_rescue.py \
  --xlsx "/Users/gongjin/Downloads/project_battery/test_data/test1.xlsx" \
  --sheet "02_PreEIS" \
  --soc 50 \
  --block 2 \
  --out_dir "./out_fit"
"""

import argparse
import os
import re
import json
import sys
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from impedance.models.circuits import CustomCircuit


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


# -----------------------------
# Basic helpers
# -----------------------------

def _clean_cell(x) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\u00a0", " ")
    return s.strip()

def _lower_cell(x) -> str:
    return _clean_cell(x).lower()

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)


def has_existing_serial_fit(serial_out_dir: str, sheet_used: str) -> bool:
    pattern = f"fit_result__{sanitize_filename(str(sheet_used))}__block*.json"
    return any(Path(serial_out_dir).glob(pattern))


def has_existing_block_fit(serial_out_dir: str, sheet_used: str, block: int) -> bool:
    path = Path(serial_out_dir) / f"fit_result__{sanitize_filename(str(sheet_used))}__block{block}.json"
    return path.exists()


def append_progress_record(out_dir: str, record: Dict[str, Any]) -> None:
    path = Path(out_dir) / "ecm_progress.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")

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

def extract_serial_token(x) -> str:
    s = "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x).strip()
    if not s:
        return ""
    s_low = s.lower()
    if "serial" in s_low and "number" in s_low:
        return ""
    tokens = re.findall(r"[A-Za-z0-9\-]{6,}", s)
    for t in tokens:
        has_alpha = any(c.isalpha() for c in t)
        has_digit = any(c.isdigit() for c in t)
        if has_alpha and has_digit:
            return t
    return ""

def detect_serial_blocks(df_raw: pd.DataFrame, serial_row_idx: int = 1) -> List[Tuple[str, int, int]]:
    """
    Detect contiguous column blocks for each serial from the serial row.
    Returns: [(serial, start_col, end_col_exclusive), ...]
    """
    if df_raw is None or df_raw.empty or serial_row_idx >= len(df_raw):
        return []

    row = df_raw.iloc[serial_row_idx].tolist()
    ncol = len(row)
    out: List[Tuple[str, int, int]] = []

    j = 0
    while j < ncol:
        token = extract_serial_token(row[j])
        if not token:
            j += 1
            continue
        start = j
        end = j + 1
        while end < ncol:
            t = extract_serial_token(row[end])
            if (not t) or (t == token):
                end += 1
            else:
                break
        out.append((token, start, end))
        j = end

    return out

def parse_guess(guess_str: str) -> List[float]:
    parts = [x.strip() for x in guess_str.split(",") if x.strip()]
    return [float(x) for x in parts]

def auto_guess_from_circuit(circuit: str, freq: np.ndarray, zre: np.ndarray) -> List[float]:
    """
    Build a reasonable initial guess from data scale.
    Supports common 2-RC / 2-CPE circuits without Warburg.
    """
    c = circuit.lower()
    r0 = float(np.nanmin(zre))
    dR = float(np.nanmax(zre) - np.nanmin(zre))
    if not np.isfinite(dR) or dR <= 0:
        dR = max(1e-12, abs(r0) * 0.5)
    r1 = 0.6 * dR
    r2 = 0.4 * dR

    f1 = float(np.nanpercentile(freq, 70))
    f2 = float(np.nanpercentile(freq, 30))
    f1 = max(f1, 1e-6)
    f2 = max(f2, 1e-6)

    # If circuit has CPE branches, use [R, Q, alpha] per branch.
    if "cpe" in c:
        a1 = 0.85
        a2 = 0.85
        q1 = 1.0 / (max(r1, 1e-12) * (2.0 * np.pi * f1) ** a1)
        q2 = 1.0 / (max(r2, 1e-12) * (2.0 * np.pi * f2) ** a2)
        base = [r0, r1, q1, a1, r2, q2, a2]
        # Semi-infinite Warburg: 1 parameter.
        if "-w" in c and "-wo" not in c and "-ws" not in c:
            base += [0.01]
        # Finite-length Warburg (open/short): usually 2 parameters.
        if "-wo" in c or "-ws" in c:
            base += [0.01, 1.0]
        return base

    if c.startswith("r0-p(r1,c1)-r2-p(r3,c2)-p(r4,c3)"):
        f_hi = float(np.nanpercentile(freq, 80))
        f_mid = float(np.nanpercentile(freq, 45))
        f_lo = float(np.nanpercentile(freq, 12))
        f_hi = max(f_hi, 1e-6)
        f_mid = max(f_mid, 1e-6)
        f_lo = max(f_lo, 1e-6)
        r_ct = 0.15 * dR
        r_sei = 0.20 * dR
        r_w1 = 0.35 * dR
        r_w2 = 0.30 * dR
        c_sei = 1.0 / (2.0 * np.pi * max(r_sei, 1e-12) * f_hi)
        c_w1 = 1.0 / (2.0 * np.pi * max(r_w1, 1e-12) * f_mid)
        c_w2 = 1.0 / (2.0 * np.pi * max(r_w2, 1e-12) * f_lo)
        return [r0, r_sei, c_sei, r_ct, r_w1, c_w1, r_w2, c_w2]

    # Default to 2-RC.
    c1 = 1.0 / (2.0 * np.pi * max(r1, 1e-12) * f1)
    c2 = 1.0 / (2.0 * np.pi * max(r2, 1e-12) * f2)
    base = [r0, r1, c1, r2, c2]
    if "-w" in c and "-wo" not in c and "-ws" not in c:
        base += [0.01]
    if "-wo" in c or "-ws" in c:
        base += [0.01, 1.0]
    return base


def apply_warburg_to_circuit(base_circuit: str, warburg: str) -> str:
    """
    Append Warburg element at the tail of the circuit string.
    warburg: none | W | Wo | Ws
    """
    w = (warburg or "none").strip().lower()
    if w == "none":
        return base_circuit
    if w == "w":
        return f"{base_circuit}-W1"
    if w == "wo":
        return f"{base_circuit}-Wo1"
    if w == "ws":
        return f"{base_circuit}-Ws1"
    raise ValueError(f"Unsupported warburg mode: {warburg}")


def resolve_circuit_spec(
    circuit_family: str,
    circuit: str,
    warburg: str,
) -> Tuple[str, str]:
    family = (circuit_family or "legacy").strip().lower()
    if family == "td_compatible":
        if (warburg or "none").strip().lower() != "none":
            print("[WARN] Ignoring --warburg for circuit_family=td_compatible; the RC-chain already approximates the diffusion tail.")
        return "R0-p(R1,C1)-R2-p(R3,C2)-p(R4,C3)", "td_compatible"

    base = circuit or "R0-p(R1,C1)-p(R2,C2)"
    return apply_warburg_to_circuit(base, warburg), "legacy"


def collect_xlsx_files(
    xlsx: Optional[str],
    xlsx_dir: Optional[str],
    recursive: bool = False,
) -> List[Path]:
    """
    Resolve input xlsx files from either one file or a directory.
    """
    has_file = bool(xlsx)
    has_dir = bool(xlsx_dir)
    if has_file == has_dir:
        raise ValueError("Please provide exactly one of --xlsx or --xlsx_dir.")

    if has_file:
        p = Path(xlsx)
        if not p.exists() or not p.is_file():
            raise ValueError(f"Invalid --xlsx path: {xlsx}")
        if p.suffix.lower() != ".xlsx":
            raise ValueError(f"--xlsx must be a .xlsx file: {xlsx}")
        return [p]

    d = Path(xlsx_dir)
    if not d.exists() or not d.is_dir():
        raise ValueError(f"Invalid --xlsx_dir path: {xlsx_dir}")
    pattern = "**/*.xlsx" if recursive else "*.xlsx"
    files = sorted(d.glob(pattern))
    if not files:
        raise ValueError(f"No .xlsx files found in directory: {xlsx_dir}")
    return files


def infer_dataset_group(xlsx_path: Path, xlsx_dir: Optional[str]) -> str:
    """
    Infer dataset group label (e.g., CL/FLC/HYCL) from path.
    """
    known = {"cl": "CL", "flc": "FLC", "hycl": "HYCL"}

    # Prefer nearest parent name matching known groups.
    for p in xlsx_path.parents:
        name = p.name.strip().lower()
        if name in known:
            return known[name]

    # If no known group found, try the first level under xlsx_dir.
    if xlsx_dir:
        root = Path(xlsx_dir)
        try:
            rel = xlsx_path.relative_to(root)
            if len(rel.parts) >= 2:
                top = rel.parts[0].strip().lower()
                if top in known:
                    return known[top]
                return sanitize_filename(rel.parts[0])
        except Exception:
            pass

    return "UNGROUPED"


def resolve_sheet_name(xlsx_path: Path, requested_sheet: str) -> Optional[str]:
    """
    Resolve sheet name.
    - If requested_sheet != 'auto': return as-is.
    - If requested_sheet == 'auto': choose the best EIS-like sheet in this file.
    """
    if str(requested_sheet).lower() != "auto":
        return requested_sheet

    xl = pd.ExcelFile(str(xlsx_path), engine="openpyxl")
    names = xl.sheet_names
    if not names:
        return None

    # Priority by explicit known names first.
    priority_exact = [
        "02_PreEIS",
        "03-4_EIS",
        "02_EIS",
        "PreEIS",
        "EIS",
    ]
    name_set_lower = {n.lower(): n for n in names}
    for p in priority_exact:
        if p.lower() in name_set_lower:
            return name_set_lower[p.lower()]

    # Fallback: any sheet containing 'eis'
    eis_like = [n for n in names if "eis" in n.lower()]
    if eis_like:
        return eis_like[0]

    # Last fallback: any sheet containing 'impedance'
    imp_like = [n for n in names if "impedance" in n.lower()]
    if imp_like:
        return imp_like[0]

    return None

def _scaled_guess(base: List[float], scale_r: float, scale_cq: float) -> List[float]:
    g = list(base)
    for i in range(len(g)):
        # Keep CPE alpha (typically index 3 and 6 in 2-CPE) unchanged and clipped.
        if i in (3, 6) and 0.0 < g[i] < 2.0:
            g[i] = min(0.99, max(0.5, g[i]))
            continue
        # Heuristic: tiny numbers are usually C/Q terms.
        if abs(g[i]) < 1e-1:
            g[i] = g[i] * scale_cq
        else:
            g[i] = g[i] * scale_r
    return g

def fit_with_restarts(
    circuit: str,
    base_guess: List[float],
    freq: np.ndarray,
    Z: np.ndarray,
    n_starts: int,
    weight_by_modulus: bool,
) -> Tuple[Any, List[float], float, List[float]]:
    """
    Multi-start fit and return the best model by complex RMSE.
    """
    candidates = [base_guess]
    scales = [(0.5, 2.0), (2.0, 0.5), (0.8, 1.5), (1.5, 0.8), (1.0, 1.0)]
    for sr, sc in scales[:max(0, n_starts - 1)]:
        candidates.append(_scaled_guess(base_guess, sr, sc))

    best_model = None
    best_params = None
    best_rmse = np.inf
    best_guess = None
    last_err = None

    for g in candidates:
        try:
            m = CustomCircuit(circuit, initial_guess=g)
            m.fit(freq, Z, weight_by_modulus=weight_by_modulus)
            Zp = m.predict(freq)
            rmse = float(np.sqrt(np.mean(np.abs(Z - Zp) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = m
                best_params = list(m.parameters_)
                best_guess = g
        except Exception as e:
            last_err = e
            continue

    if best_model is None:
        raise ValueError(f"All multi-start fits failed. Last error: {last_err}")
    return best_model, best_params, best_rmse, best_guess

def save_nyquist(Z, Z_fit, out_png: str, title: str) -> None:
    plt.figure()
    plt.plot(np.real(Z), -np.imag(Z), "o", label="Measured")
    plt.plot(np.real(Z_fit), -np.imag(Z_fit), "-", label="Fitted")
    plt.xlabel("Z' (Ohm)")
    plt.ylabel("-Z'' (Ohm)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_serial_mosaic(
    image_items: List[Tuple[str, str]],
    out_png: str,
    cols: int = 4,
    title: Optional[str] = None,
) -> None:
    """
    Merge serial-level fit images into one comparison mosaic.
    image_items: list of (serial, image_path)
    """
    if not image_items:
        return
    cols = max(1, int(cols))
    n = len(image_items)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.6 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    k = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if k >= n:
                ax.axis("off")
                continue
            serial, img_path = image_items[k]
            try:
                img = plt.imread(img_path)
                ax.imshow(img)
                ax.set_title(f"Serial: {serial}", fontsize=10)
                ax.axis("off")
            except Exception as e:
                ax.text(0.5, 0.5, f"Failed to load\n{serial}\n{e}", ha="center", va="center", fontsize=8)
                ax.axis("off")
            k += 1

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

def compute_fit_metrics(Z: np.ndarray, Z_fit: np.ndarray) -> Dict[str, float]:
    """
    Compute residual-based metrics on complex impedance.
    """
    r_re = np.real(Z) - np.real(Z_fit)
    r_im = np.imag(Z) - np.imag(Z_fit)
    r_abs = np.abs(Z - Z_fit)
    z_abs = np.abs(Z)

    rmse_complex = float(np.sqrt(np.mean(r_abs ** 2)))
    mae_complex = float(np.mean(r_abs))
    rmse_real = float(np.sqrt(np.mean(r_re ** 2)))
    rmse_imag = float(np.sqrt(np.mean(r_im ** 2)))

    denom = float(np.mean(z_abs))
    nrmse_pct = float(100.0 * rmse_complex / denom) if denom > 0 else float("nan")

    # R^2 on real/imag components separately
    def _r2(y: np.ndarray, yp: np.ndarray) -> float:
        ss_res = float(np.sum((y - yp) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot <= 0:
            return float("nan")
        return 1.0 - ss_res / ss_tot

    r2_real = _r2(np.real(Z), np.real(Z_fit))
    r2_imag = _r2(np.imag(Z), np.imag(Z_fit))

    return {
        "n_points": int(len(Z)),
        "rmse_complex_ohm": rmse_complex,
        "mae_complex_ohm": mae_complex,
        "nrmse_complex_percent_of_mean_absZ": nrmse_pct,
        "rmse_real_ohm": rmse_real,
        "rmse_imag_ohm": rmse_imag,
        "r2_real": float(r2_real),
        "r2_imag": float(r2_imag),
    }

def find_best_header_row(df_raw: pd.DataFrame, keywords: List[str], scan_rows: int = 120) -> Optional[int]:
    best_i, best_score = None, 0
    n = min(len(df_raw), scan_rows)
    for i in range(n):
        row = df_raw.iloc[i].astype(str).map(_lower_cell).tolist()
        score = sum(1 for kw in keywords if any(kw in c for c in row))
        if score > best_score:
            best_score = score
            best_i = i
    return best_i if best_score > 0 else None


# -----------------------------
# Block/header logic
# -----------------------------

def is_frequency_header(cell: str) -> bool:
    c = cell.lower()
    return ("frequency" in c) and ("hz" in c)

def assign_block_ids(header_cells: List[str]) -> List[Optional[int]]:
    """
    block increments on each Frequency-like header encountered.
    All columns after that belong to that block until the next Frequency.
    """
    block_ids: List[Optional[int]] = [None] * len(header_cells)
    current_block = -1
    for j, cell in enumerate(header_cells):
        if is_frequency_header(cell):
            current_block += 1
        if current_block >= 0:
            block_ids[j] = current_block
    return block_ids

def idxs_in_block(block_ids: List[Optional[int]], block: int) -> List[int]:
    return [j for j, b in enumerate(block_ids) if b == block]


def candidate_blocks_for_serial(
    df_raw: pd.DataFrame,
    header_row: int,
    serial_col_range: Optional[Tuple[int, int]] = None,
) -> List[int]:
    header_cells = df_raw.iloc[header_row].astype(str).map(_clean_cell).tolist()
    block_ids = assign_block_ids(header_cells)
    max_block = max([b for b in block_ids if b is not None], default=-1)
    if max_block < 0:
        return []

    out: List[int] = []
    for b in range(max_block + 1):
        cols_b = idxs_in_block(block_ids, b)
        j_f = find_freq_col_in_block(header_cells, cols_b)
        if j_f is None:
            continue
        if serial_col_range is not None:
            c0, c1 = serial_col_range
            if not (c0 <= j_f < c1):
                continue
        out.append(int(b))
    return out


def _first_nonempty(values: List[str]) -> str:
    for v in values:
        s = _clean_cell(v)
        if s:
            return s
    return ""


def _coerce_cycle_value(v) -> Optional[int]:
    if pd.isna(v):
        return None
    s = _clean_cell(v)
    if not s:
        return None
    m = re.search(r"(-?\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def extract_block_measurement_meta(
    df_raw: pd.DataFrame,
    header_row: int,
    block: int,
) -> Dict[str, Optional[object]]:
    """
    Extract measurement basis / cycle metadata for one EIS block.

    In 03-4_EIS sheets we often see headers like:
      - Series Basis
      - Measurement Day or Cycle / Day/Cycle
      - EIS Equipment ID
      - Frequency (Hz)
    """
    if df_raw.empty or header_row >= len(df_raw):
        return {"measurement_basis": None, "measurement_cycle": None}

    header_cells = df_raw.iloc[header_row].astype(str).map(_clean_cell).tolist()
    block_ids = assign_block_ids(header_cells)
    cols_b = idxs_in_block(block_ids, block)
    if not cols_b:
        return {"measurement_basis": None, "measurement_cycle": None}

    basis_col = None
    cycle_col = None
    for j in cols_b:
        h = header_cells[j].lower()
        if ("series basis" in h) or (h == "basis"):
            basis_col = j
        if ("measurement day or cycle" in h) or ("day/cycle" in h) or ("measurement cycle" in h):
            cycle_col = j

    measurement_basis = None
    measurement_cycle = None
    if basis_col is not None and header_row + 1 < len(df_raw):
        vals = df_raw.iloc[header_row + 1 :, basis_col].tolist()
        basis = _first_nonempty(vals)
        measurement_basis = basis or None

    if cycle_col is not None and header_row + 1 < len(df_raw):
        vals = df_raw.iloc[header_row + 1 :, cycle_col].tolist()
        for v in vals:
            measurement_cycle = _coerce_cycle_value(v)
            if measurement_cycle is not None:
                break

    return {
        "measurement_basis": measurement_basis,
        "measurement_cycle": measurement_cycle,
    }

def find_freq_col_in_block(header_cells: List[str], cols: List[int]) -> Optional[int]:
    for j in cols:
        if is_frequency_header(header_cells[j]):
            return j
    return None


# -----------------------------
# Core rescue: choose real/imag by numeric density on freq rows
# -----------------------------

def score_col_on_mask(data: pd.DataFrame, j: int, mask_f: pd.Series) -> Dict[str, Any]:
    """
    Compute numeric density and sign stats for column j restricted to mask_f rows.
    """
    s = to_float_series(data.iloc[:, j])
    s_masked = s[mask_f]

    nn = int(s_masked.notna().sum())
    if nn == 0:
        return {
            "j": j,
            "nn": 0,
            "neg_frac": 0.0,
            "median": None,
            "sample_raw": data.iloc[:8, j].tolist(),
        }

    vals = s_masked.dropna().to_numpy(dtype=float)
    neg_frac = float(np.mean(vals < 0)) if len(vals) else 0.0
    med = float(np.median(vals)) if len(vals) else None

    return {
        "j": j,
        "nn": nn,
        "neg_frac": neg_frac,
        "median": med,
        "sample_raw": data.iloc[:8, j].tolist(),
    }

def choose_real_imag_cols_by_rescue(
    df_raw: pd.DataFrame,
    header_row: int,
    header_cells: List[str],
    cols_in_block: List[int],
    j_f: int,
    min_points: int,
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Use frequency numeric rows as anchor, then select best Real/Imag columns
    by numeric density and sign heuristics.

    - Real: prefer high nn and median >= 0
    - Imag: prefer high nn and higher neg_frac (often negative)
    """
    data = df_raw.iloc[header_row + 1:].copy()

    freq_s = to_float_series(data.iloc[:, j_f])
    mask_f = freq_s.notna() & (freq_s > 0)

    mask_f_cnt = int(mask_f.sum())

    # Score all columns in this block except freq itself
    candidates = [j for j in cols_in_block if j != j_f]
    scored = [score_col_on_mask(data, j, mask_f) for j in candidates]
    # Sort by nn desc
    scored_sorted = sorted(scored, key=lambda d: d["nn"], reverse=True)

    # Debug top candidates
    top_any = scored_sorted[:10]

    # Pick Real: among best nn, prefer median>=0
    real_pick = None
    for d in scored_sorted:
        if d["nn"] >= min_points and d["median"] is not None and d["median"] >= 0:
            real_pick = d
            break
    if real_pick is None and scored_sorted:
        real_pick = scored_sorted[0]

    # Pick Imag: prefer neg_frac high, and not the same as real
    imag_pick = None
    scored_imag = sorted(scored_sorted, key=lambda d: (d["nn"], d["neg_frac"]), reverse=True)
    for d in scored_imag:
        if d["j"] == real_pick["j"]:
            continue
        if d["nn"] >= min_points:
            # if we have any negative tendency, great; else still accept
            imag_pick = d
            break
    if imag_pick is None:
        # last resort: pick next best different col
        for d in scored_sorted:
            if d["j"] != real_pick["j"]:
                imag_pick = d
                break

    if real_pick is None or imag_pick is None:
        raise ValueError("Rescue failed: cannot select real/imag columns.")

    dbg = {
        "mask_f_cnt": mask_f_cnt,
        "top_any_candidates": [
            {
                "j": d["j"],
                "header": header_cells[d["j"]],
                "nn": d["nn"],
                "neg_frac": d["neg_frac"],
                "median": d["median"],
                "sample_raw_top8": d["sample_raw"],
            }
            for d in top_any
        ],
        "chosen_real": {
            "j": real_pick["j"],
            "header": header_cells[real_pick["j"]],
            "nn": real_pick["nn"],
            "neg_frac": real_pick["neg_frac"],
            "median": real_pick["median"],
            "sample_raw_top8": real_pick["sample_raw"],
        },
        "chosen_imag": {
            "j": imag_pick["j"],
            "header": header_cells[imag_pick["j"]],
            "nn": imag_pick["nn"],
            "neg_frac": imag_pick["neg_frac"],
            "median": imag_pick["median"],
            "sample_raw_top8": imag_pick["sample_raw"],
        },
    }

    return int(real_pick["j"]), int(imag_pick["j"]), dbg


def extract_triplet(
    df_raw: pd.DataFrame,
    header_row: int,
    j_f: int,
    j_r: int,
    j_i: int,
    imag_is_negative: bool,
    assume_mohm: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Extract arrays from raw using column indices.
    """
    data = df_raw.iloc[header_row + 1:].copy()

    freq_s = to_float_series(data.iloc[:, j_f])
    zre_s = to_float_series(data.iloc[:, j_r])
    zim_s = to_float_series(data.iloc[:, j_i])

    mask = freq_s.notna() & zre_s.notna() & zim_s.notna() & (freq_s > 0)

    dbg = {
        "non_nan_total": {
            "freq": int(freq_s.notna().sum()),
            "real": int(zre_s.notna().sum()),
            "imag": int(zim_s.notna().sum()),
        },
        "joint_valid": int(mask.sum()),
        "raw_samples_top12": {
            "freq": data.iloc[:12, j_f].tolist(),
            "real": data.iloc[:12, j_r].tolist(),
            "imag": data.iloc[:12, j_i].tolist(),
        },
    }

    freq = freq_s[mask].to_numpy(dtype=float)
    zre = zre_s[mask].to_numpy(dtype=float)
    zim = zim_s[mask].to_numpy(dtype=float)

    # unit conversion
    if assume_mohm:
        zre = zre * 1e-3
        zim = zim * 1e-3

    if imag_is_negative:
        zim = -zim

    return freq, zre, zim, dbg


# -----------------------------
# Load with fallback across blocks
# -----------------------------

def load_eis_with_rescue_and_fallback(
    xlsx_path: str,
    sheet_name: str,
    requested_block: int,
    header_row: Optional[int],
    imag_is_negative: bool,
    min_points: int,
    assume_mohm: bool,
    serial_col_range: Optional[Tuple[int, int]] = None,
    search_all_blocks: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, Tuple[int,int,int], int]:
    df_raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    if df_raw.empty:
        raise ValueError(f"Empty sheet: {sheet_name}")

    if header_row is None:
        header_row = find_best_header_row(df_raw, ["frequency", "real", "imag", "soc"], scan_rows=200)
        if header_row is None:
            header_row = find_best_header_row(df_raw, ["frequency", "real", "imag"], scan_rows=200)
        if header_row is None:
            raise ValueError("Cannot auto-detect header row. Please pass --header explicitly.")

    header_cells = df_raw.iloc[header_row].astype(str).map(_clean_cell).tolist()
    block_ids = assign_block_ids(header_cells)
    max_block = max([b for b in block_ids if b is not None], default=-1)
    if max_block < 0:
        raise ValueError("No blocks detected (no Frequency/Hz header found).")

    print(f"[DEBUG] header_row={header_row} | detected max_block={max_block} | requested_block={requested_block}")
    print(f"[DEBUG] header preview (0..60): {header_cells[:60]}")

    best = None
    best_joint = -1
    best_block = None
    best_cols = None

    def try_block(b: int):
        nonlocal best, best_joint, best_block, best_cols

        cols_b = idxs_in_block(block_ids, b)
        j_f = find_freq_col_in_block(header_cells, cols_b)
        if j_f is None:
            print(f"[DEBUG] block={b}: no freq col found in this block.")
            return
        if serial_col_range is not None:
            c0, c1 = serial_col_range
            if not (c0 <= j_f < c1):
                return

        # Rescue choose real/imag within this block
        j_r, j_i, rescue_dbg = choose_real_imag_cols_by_rescue(
            df_raw=df_raw,
            header_row=header_row,
            header_cells=header_cells,
            cols_in_block=cols_b,
            j_f=j_f,
            min_points=min_points,
        )

        print(f"[DEBUG] block={b}: freq_col={j_f}({header_cells[j_f]})")
        print(f"[DEBUG] block={b}: chosen real={j_r}({header_cells[j_r]}) imag={j_i}({header_cells[j_i]})")
        print(f"[DEBUG] block={b}: rescue mask_f_cnt={rescue_dbg['mask_f_cnt']}")
        print(f"[DEBUG] block={b}: top_any_candidates[:3] = {rescue_dbg['top_any_candidates'][:3]}")

        freq, zre, zim, dbg = extract_triplet(
            df_raw=df_raw,
            header_row=header_row,
            j_f=j_f,
            j_r=j_r,
            j_i=j_i,
            imag_is_negative=imag_is_negative,
            assume_mohm=assume_mohm,
        )

        joint = int(dbg["joint_valid"])
        print(f"[DEBUG] block={b}: joint_valid={joint} | N={len(freq)}")
        print(f"[DEBUG] block={b}: raw_samples_top12(real)={dbg['raw_samples_top12']['real']}")

        if joint > best_joint:
            best_joint = joint
            best = (freq, zre, zim)
            best_block = b
            best_cols = (j_f, j_r, j_i)

    # Try requested first, then all
    try_block(requested_block)
    if search_all_blocks:
        for b in range(max_block + 1):
            if b != requested_block:
                try_block(b)

    if best is None or best_joint < min_points:
        raise ValueError(
            f"No usable EIS data found in any block. best_joint_valid={best_joint} (<{min_points}).\n"
            "Important: your log shows Real/Imag named columns are empty ('Decimal','mOhm', then NaN).\n"
            "This script already rescues by scanning numeric columns aligned to Frequency rows.\n"
            "If still 0, then this sheet likely does NOT contain numeric Real/Imag data at all (only Frequency),\n"
            "or the true EIS numeric table is elsewhere (another sheet / another header region)."
        )

    if best_block != requested_block:
        print(f"[WARN] Falling back to best block={best_block} (joint_valid={best_joint}) instead of requested={requested_block}")

    freq, zre, zim = best
    return freq, zre, zim, int(best_block), (int(best_cols[0]), int(best_cols[1]), int(best_cols[2])), int(header_row)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Fit EIS data to equivalent circuit models, with optional Warburg tail, "
            "batch xlsx traversal, serial auto-detection, and fit-quality export."
        ),
        epilog=(
            "Examples:\n"
            "  Single file, no Warburg:\n"
            "    python src/ecm_fit.py --xlsx ./dataset/test1.xlsx --sheet 02_PreEIS --block 2 "
            "--circuit \"R0-p(R1,CPE1)-p(R2,CPE2)\" --guess \"\" --out_dir ./data/test_ecm\n\n"
            "  Directory batch, auto sheet, with Warburg:\n"
            "    python src/ecm_fit.py --xlsx_dir ./dataset/OneDrive_1_2-20-2026 --recursive "
            "--sheet auto --circuit \"R0-p(R1,CPE1)-p(R2,CPE2)\" --warburg W --guess \"\" "
            "--merge_serial_plots --out_dir ./data/test_ecm_all\n\n"
            "Outputs include:\n"
            "  nyquist_fit__*.png, fit_metrics__*.json, fit_residuals__*.csv, fit_result__*.json"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--xlsx", required=False, help="Single xlsx file path. Mutually exclusive with --xlsx_dir.")
    ap.add_argument("--xlsx_dir", required=False, help="Directory containing xlsx files. Mutually exclusive with --xlsx.")
    ap.add_argument("--recursive", action="store_true", help="Recursively scan --xlsx_dir for *.xlsx files.")
    ap.add_argument("--sheet", default="02_PreEIS", help="Sheet name to use, or 'auto' to detect from common EIS sheet names.")
    ap.add_argument(
        "--fit_mode",
        choices=["best_block", "all_valid_blocks"],
        default="best_block",
        help=(
            "Block fitting mode:\n"
            "  best_block       : keep existing behavior, one best/fallback block per serial\n"
            "  all_valid_blocks : fit all candidate EIS blocks within each serial range separately"
        ),
    )
    ap.add_argument("--soc", type=int, default=50, help="Kept for CLI compatibility; rescue logic does not rely on SOC header.")
    ap.add_argument("--block", type=int, default=2, help="Preferred serial block index. Script may fall back to a better block.")
    ap.add_argument("--header", type=int, default=None, help="Optional header row override.")
    ap.add_argument("--imag_is_negative", action="store_true", help="Interpret imag column as already negative.")
    ap.add_argument("--auto_sign", dest="auto_sign", action="store_true", help="Auto-detect imag sign for Nyquist consistency.")
    ap.add_argument("--no_auto_sign", dest="auto_sign", action="store_false", help="Disable auto sign detection.")
    ap.set_defaults(auto_sign=True)
    ap.add_argument("--assume_mohm", action="store_true", help="Assume Real/Imag numeric values are in mOhm and convert to Ohm.")

    # no-Warburg default: 2-CPE often fits battery arcs better than ideal 2-RC
    ap.add_argument(
        "--circuit_family",
        default="legacy",
        choices=["legacy", "td_compatible"],
        help=(
            "Circuit family selector. legacy keeps the original frequency-domain model path; "
            "td_compatible switches to a mentor-style RC-chain circuit that is easier to reuse as a time-domain prior source."
        ),
    )
    ap.add_argument("--circuit", default="R0-p(R1,CPE1)-p(R2,CPE2)", help="Base ECM topology before optional Warburg append.")
    ap.add_argument("--warburg", default="none", choices=["none", "W", "Wo", "Ws"], help="Append a Warburg element to circuit tail.")
    ap.add_argument("--guess", default="", help="Comma-separated initial guess. Empty -> auto guess from data.")

    ap.add_argument("--out_dir", default="./out_fit", help="Output root directory.")
    ap.add_argument("--serial", required=False, help="Only process this serial. If omitted, process all detected serials.")
    ap.add_argument("--min_points", type=int, default=5, help="Minimum valid EIS points required for fitting.")
    ap.add_argument("--drop_first_n", type=int, default=0, help="Drop N highest-frequency points before fitting.")
    ap.add_argument("--fmin", type=float, default=None, help="Minimum frequency (Hz) to keep.")
    ap.add_argument("--fmax", type=float, default=None, help="Maximum frequency (Hz) to keep.")
    ap.add_argument("--n_starts", type=int, default=5, help="Number of multi-start attempts.")
    ap.add_argument("--weight_by_modulus", action="store_true", help="Use modulus weighting in nonlinear fit.")
    ap.add_argument("--merge_serial_plots", action="store_true", help="Merge all serial fit plots per xlsx into one comparison image.")
    ap.add_argument("--merge_cols", type=int, default=4, help="Column count in merged serial comparison image.")
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip serials that already have a fit_result for the current sheet in out_dir.",
    )
    ap.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="Force re-fit even if outputs already exist for a serial/sheet.",
    )
    ap.add_argument("--log_file", default="", help="Optional path to save a copy of stdout/stderr logs.")
    args = ap.parse_args()
    setup_log_tee(args.log_file)

    ensure_dir(args.out_dir)
    xlsx_files = collect_xlsx_files(args.xlsx, args.xlsx_dir, recursive=args.recursive)
    print(f"[INFO] Found {len(xlsx_files)} xlsx file(s) to process.")

    planned_tasks: List[Dict[str, Any]] = []
    for xlsx_path in xlsx_files:
        try:
            sheet_used = resolve_sheet_name(Path(xlsx_path), args.sheet)
            if sheet_used is None:
                continue
            df_sheet = pd.read_excel(xlsx_path, sheet_name=sheet_used, header=None, engine="openpyxl")
            serial_blocks = detect_serial_blocks(df_sheet, serial_row_idx=1)
            if not serial_blocks:
                serial_blocks = detect_serial_blocks(df_sheet, serial_row_idx=0)
            if not serial_blocks:
                serial_blocks = detect_serial_blocks(df_sheet, serial_row_idx=2)
            if not serial_blocks:
                serial_blocks = [(detect_primary_serial(df_sheet), 0, df_sheet.shape[1])]
            if args.serial:
                serial_blocks = [x for x in serial_blocks if x[0] == args.serial]
            header_row_plan = args.header
            if header_row_plan is None:
                header_row_plan = find_best_header_row(df_sheet, ["frequency", "real", "imag", "soc"], scan_rows=200)
                if header_row_plan is None:
                    header_row_plan = find_best_header_row(df_sheet, ["frequency", "real", "imag"], scan_rows=200)
            if header_row_plan is None:
                continue

            for serial, c0, c1 in serial_blocks:
                if args.fit_mode == "all_valid_blocks":
                    blocks = candidate_blocks_for_serial(df_sheet, header_row_plan, serial_col_range=(c0, c1))
                    for b in blocks:
                        planned_tasks.append(
                            {
                                "xlsx_path": str(xlsx_path),
                                "sheet_used": str(sheet_used),
                                "serial": str(serial),
                                "col_range": (int(c0), int(c1)),
                                "requested_block": int(b),
                            }
                        )
                else:
                    planned_tasks.append(
                        {
                            "xlsx_path": str(xlsx_path),
                            "sheet_used": str(sheet_used),
                            "serial": str(serial),
                            "col_range": (int(c0), int(c1)),
                            "requested_block": int(args.block),
                        }
                    )
        except Exception:
            continue

    total_tasks = len(planned_tasks)
    print(f"[INFO] Planned ECM fit tasks: {total_tasks} (fit_mode={args.fit_mode})")

    ok_cnt = 0
    fail_cnt = 0
    skip_cnt = 0
    done_cnt = 0

    for xlsx_path in xlsx_files:
        try:
            print(f"\n[INFO] Processing file: {xlsx_path}")
            sheet_used = resolve_sheet_name(Path(xlsx_path), args.sheet)
            if sheet_used is None:
                raise ValueError("No EIS-like sheet found for --sheet auto.")
            print(f"[INFO] Using sheet: {sheet_used}")
            df_sheet = pd.read_excel(xlsx_path, sheet_name=sheet_used, header=None, engine="openpyxl")
            serial_blocks = detect_serial_blocks(df_sheet, serial_row_idx=1)
            if not serial_blocks:
                serial_blocks = detect_serial_blocks(df_sheet, serial_row_idx=0)
            if not serial_blocks:
                serial_blocks = detect_serial_blocks(df_sheet, serial_row_idx=2)
            if not serial_blocks:
                serial_blocks = [(detect_primary_serial(df_sheet), 0, df_sheet.shape[1])]

            if args.serial:
                serial_blocks = [x for x in serial_blocks if x[0] == args.serial]
                if not serial_blocks:
                    raise ValueError(f"Requested serial not found in sheet: {args.serial}")

            print(f"[INFO] Detected {len(serial_blocks)} serial block(s) in sheet={sheet_used}")
        except Exception as e:
            fail_cnt += 1
            print(f"[WARN] file={xlsx_path} failed during setup: {e}")
            continue

        group_tag = infer_dataset_group(Path(xlsx_path), args.xlsx_dir)
        file_out_dir = Path(args.out_dir) / group_tag / sanitize_filename(xlsx_path.stem)
        ensure_dir(str(file_out_dir))
        merged_items: List[Tuple[str, str]] = []

        for serial, c0, c1 in serial_blocks:
            try:
                serial_out_dir = os.path.join(str(file_out_dir), sanitize_filename(serial))
                ensure_dir(serial_out_dir)

                if args.overwrite_existing and args.skip_existing:
                    raise ValueError("Use only one of --skip_existing or --overwrite_existing.")

                block_requests: List[int]
                if args.fit_mode == "all_valid_blocks":
                    header_row_tmp = args.header
                    if header_row_tmp is None:
                        header_row_tmp = find_best_header_row(df_sheet, ["frequency", "real", "imag", "soc"], scan_rows=200)
                        if header_row_tmp is None:
                            header_row_tmp = find_best_header_row(df_sheet, ["frequency", "real", "imag"], scan_rows=200)
                    if header_row_tmp is None:
                        raise ValueError("Cannot auto-detect header row for all_valid_blocks mode.")
                    block_requests = candidate_blocks_for_serial(df_sheet, header_row_tmp, serial_col_range=(c0, c1))
                else:
                    block_requests = [int(args.block)]

                for requested_block in block_requests:
                    if args.fit_mode == "all_valid_blocks":
                        if args.skip_existing and has_existing_block_fit(serial_out_dir, sheet_used, requested_block):
                            done_cnt += 1
                            skip_cnt += 1
                            print(f"[INFO] [{done_cnt}/{total_tasks}] Skipping serial={serial} block={requested_block} because existing fit_result was found")
                            append_progress_record(
                                args.out_dir,
                                {
                                    "file_name": str(xlsx_path.name),
                                    "group_tag": str(group_tag),
                                    "serial": str(serial),
                                    "sheet": str(sheet_used),
                                    "requested_block": int(requested_block),
                                    "status": "skipped_existing",
                                },
                            )
                            continue
                    else:
                        if args.skip_existing and has_existing_serial_fit(serial_out_dir, sheet_used):
                            done_cnt += 1
                            skip_cnt += 1
                            print(f"[INFO] [{done_cnt}/{total_tasks}] Skipping serial={serial} because existing fit_result was found for sheet={sheet_used}")
                            append_progress_record(
                                args.out_dir,
                                {
                                    "file_name": str(xlsx_path.name),
                                    "group_tag": str(group_tag),
                                    "serial": str(serial),
                                    "sheet": str(sheet_used),
                                    "requested_block": int(requested_block),
                                    "status": "skipped_existing",
                                },
                            )
                            continue

                    print(f"\n[INFO] [{done_cnt + 1}/{total_tasks}] Processing serial={serial} cols=[{c0},{c1}) requested_block={requested_block}")

                    freq, zre, zim, chosen_block, cols, hdr = load_eis_with_rescue_and_fallback(
                        xlsx_path=str(xlsx_path),
                        sheet_name=sheet_used,
                        requested_block=requested_block,
                        header_row=args.header,
                        imag_is_negative=args.imag_is_negative,
                        min_points=args.min_points,
                        assume_mohm=args.assume_mohm,
                        serial_col_range=(c0, c1),
                        search_all_blocks=(args.fit_mode == "best_block"),
                    )

                    j_f, j_r, j_i = cols
                    meas_meta = extract_block_measurement_meta(
                        df_raw=df_sheet,
                        header_row=hdr,
                        block=chosen_block,
                    )
                    print(f"[INFO] Header row used = {hdr}")
                    print(f"[INFO] chosen_block = {chosen_block} (requested={requested_block})")
                    print(f"[INFO] Using RAW column indices: freq={j_f} | real={j_r} | imag={j_i}")
                    print(f"[INFO] Valid rows N = {len(freq)}")
                    print("[INFO] freq[:5] =", freq[:5])
                    print("[INFO] zre[:5]  =", zre[:5])
                    print("[INFO] zim[:5]  =", zim[:5])
                    print("[INFO] measurement basis/cycle =", meas_meta)

                    # 1) sort by frequency (important)
                    order = np.argsort(freq)
                    freq = freq[order]
                    zre = zre[order]
                    zim = zim[order]

                # 2) optional frequency window filtering
                    if args.fmin is not None:
                        m = freq >= args.fmin
                        freq, zre, zim = freq[m], zre[m], zim[m]
                    if args.fmax is not None:
                        m = freq <= args.fmax
                        freq, zre, zim = freq[m], zre[m], zim[m]
                    if len(freq) < args.min_points:
                        raise ValueError(f"Too few points after fmin/fmax filtering: {len(freq)}")

                # 3) drop highest-frequency outliers if requested
                    if args.drop_first_n > 0:
                        od = np.argsort(-freq)  # high -> low
                        fd, rd, id_ = freq[od], zre[od], zim[od]
                        fd, rd, id_ = fd[args.drop_first_n:], rd[args.drop_first_n:], id_[args.drop_first_n:]
                        oa = np.argsort(fd)
                        freq, zre, zim = fd[oa], rd[oa], id_[oa]
                        if len(freq) < args.min_points:
                            raise ValueError(f"Too few points after drop_first_n={args.drop_first_n}: {len(freq)}")

                # 4) choose imag sign for Nyquist consistency
                    if args.auto_sign:
                        z1 = zre + 1j * zim
                        z2 = zre + 1j * (-zim)
                        neg1 = float(np.mean(np.imag(z1) < 0))
                        neg2 = float(np.mean(np.imag(z2) < 0))
                        if neg1 >= neg2:
                            Z = z1
                            sign_mode = "imag=raw"
                        else:
                            Z = z2
                            sign_mode = "imag=-raw"
                    else:
                        Z = zre + 1j * (-zim)
                        sign_mode = "imag=-raw (forced)"

                # 5) build initial guess
                    circuit, circuit_family_used = resolve_circuit_spec(
                        args.circuit_family,
                        args.circuit,
                        args.warburg,
                    )
                    if args.guess.strip():
                        guess = parse_guess(args.guess)
                    else:
                        guess = auto_guess_from_circuit(circuit, freq, zre)

                    print("[DEBUG] Sign mode:", sign_mode)
                    print("[DEBUG] Init guess:", guess)
                    print("[DEBUG] Circuit:", circuit)

                    model, params, rmse, used_guess = fit_with_restarts(
                        circuit=circuit,
                        base_guess=guess,
                        freq=freq,
                        Z=Z,
                        n_starts=max(1, args.n_starts),
                        weight_by_modulus=args.weight_by_modulus,
                    )

                    print("=== Fit result ===")
                    print("Serial:", serial)
                    print("Circuit:", circuit)
                    print("Params:", params)
                    print("RMSE(|Z| complex):", rmse)
                    print("Used init guess:", used_guess)

                    Z_fit = model.predict(freq)
                    out_png = os.path.join(
                        serial_out_dir,
                        f"nyquist_fit__{sanitize_filename(str(sheet_used))}__block{chosen_block}.png"
                    )
                    title = f"Nyquist + Fit | file={xlsx_path.name} | serial={serial} | sheet={sheet_used} | block={chosen_block}"
                    save_nyquist(Z, Z_fit, out_png, title)
                    print(f"[INFO] Saved plot -> {out_png}")
                    merged_items.append((f"{serial}#b{chosen_block}", out_png))

                    metrics = compute_fit_metrics(Z, Z_fit)
                    print("[INFO] Fit metrics:", metrics)

                    metrics_path = os.path.join(
                        serial_out_dir,
                        f"fit_metrics__{sanitize_filename(str(sheet_used))}__block{chosen_block}.json"
                    )
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        json.dump(metrics, f, indent=2, ensure_ascii=True)
                    print(f"[INFO] Saved metrics -> {metrics_path}")

                    fit_result = {
                        "file_name": str(xlsx_path.name),
                        "group_tag": str(group_tag),
                        "serial": str(serial),
                        "sheet": str(sheet_used),
                        "measurement_basis": meas_meta.get("measurement_basis"),
                        "measurement_cycle": meas_meta.get("measurement_cycle"),
                        "circuit_family": str(circuit_family_used),
                        "requested_block": int(requested_block),
                        "chosen_block": int(chosen_block),
                        "raw_col_indices": {
                            "freq": int(j_f),
                            "real": int(j_r),
                            "imag": int(j_i),
                        },
                        "circuit": str(circuit),
                        "params": [float(x) for x in params],
                        "rmse_complex_ohm": float(rmse),
                        "used_init_guess": [float(x) for x in used_guess],
                    }
                    fit_result_path = os.path.join(
                        serial_out_dir,
                        f"fit_result__{sanitize_filename(str(sheet_used))}__block{chosen_block}.json"
                    )
                    with open(fit_result_path, "w", encoding="utf-8") as f:
                        json.dump(fit_result, f, indent=2, ensure_ascii=True)
                    print(f"[INFO] Saved fit result -> {fit_result_path}")

                    resid_df = pd.DataFrame({
                        "freq_hz": freq,
                        "z_real_meas_ohm": np.real(Z),
                        "z_imag_meas_ohm": np.imag(Z),
                        "z_real_fit_ohm": np.real(Z_fit),
                        "z_imag_fit_ohm": np.imag(Z_fit),
                        "resid_real_ohm": np.real(Z) - np.real(Z_fit),
                        "resid_imag_ohm": np.imag(Z) - np.imag(Z_fit),
                        "resid_abs_ohm": np.abs(Z - Z_fit),
                    })
                    resid_path = os.path.join(
                        serial_out_dir,
                        f"fit_residuals__{sanitize_filename(str(sheet_used))}__block{chosen_block}.csv"
                    )
                    resid_df.to_csv(resid_path, index=False)
                    print(f"[INFO] Saved residuals -> {resid_path}")
                    done_cnt += 1
                    append_progress_record(
                        args.out_dir,
                        {
                            "file_name": str(xlsx_path.name),
                            "group_tag": str(group_tag),
                            "serial": str(serial),
                            "sheet": str(sheet_used),
                            "requested_block": int(requested_block),
                            "chosen_block": int(chosen_block),
                            "measurement_cycle": meas_meta.get("measurement_cycle"),
                            "circuit_family": str(circuit_family_used),
                            "status": "ok",
                            "rmse_complex_ohm": float(rmse),
                            "done_tasks": int(done_cnt),
                            "total_tasks": int(total_tasks),
                        },
                    )
                    print(f"[INFO] Progress: {done_cnt}/{total_tasks} tasks completed")
                    ok_cnt += 1
            except Exception as e:
                fail_cnt += 1
                done_cnt += 1
                print(f"[WARN] file={xlsx_path.name} serial={serial} failed: {e}")
                append_progress_record(
                    args.out_dir,
                    {
                        "file_name": str(xlsx_path.name),
                        "group_tag": str(group_tag),
                        "serial": str(serial),
                        "sheet": str(sheet_used),
                        "done_tasks": int(done_cnt),
                        "total_tasks": int(total_tasks),
                        "status": "failed",
                        "error": str(e),
                    },
                )

        if args.merge_serial_plots and merged_items:
            merge_png = file_out_dir / f"nyquist_fit_mosaic__{sanitize_filename(str(sheet_used))}.png"
            save_serial_mosaic(
                image_items=merged_items,
                out_png=str(merge_png),
                cols=args.merge_cols,
                title=f"Nyquist Fit Comparison | file={xlsx_path.name} | sheet={sheet_used}",
            )
            print(f"[INFO] Saved merged serial plot -> {merge_png}")

    print(
        f"\n[INFO] Done. Output -> {args.out_dir} | "
        f"success={ok_cnt}, failed={fail_cnt}, skipped_existing={skip_cnt}, total={total_tasks}"
    )


if __name__ == "__main__":
    main()
