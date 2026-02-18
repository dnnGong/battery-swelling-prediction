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
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from impedance.models.circuits import CustomCircuit


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

def parse_guess(guess_str: str) -> List[float]:
    parts = [x.strip() for x in guess_str.split(",") if x.strip()]
    return [float(x) for x in parts]

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--sheet", default="02_PreEIS")
    ap.add_argument("--soc", type=int, default=50, help="kept for CLI compatibility; rescue logic does not rely on SOC header")
    ap.add_argument("--block", type=int, default=2)
    ap.add_argument("--header", type=int, default=None)
    ap.add_argument("--imag_is_negative", action="store_true")
    ap.add_argument("--assume_mohm", action="store_true", help="Assume Real/Imag numeric are in mOhm and convert to Ohm (x1e-3).")

    # - 一阶RC
    # ap.add_argument("--circuit", default="R0-p(R1,C1)")
    # ap.add_argument("--guess", default="0.02,0.05,1e-3")
    # - 二阶RC
    ap.add_argument("--circuit", default="R0-p(R1,C1)-p(R2,C2)")
    ap.add_argument("--guess", default="0.02,0.05,1e-3,0.03,1e-2")

    ap.add_argument("--out_dir", default="./out_fit")
    ap.add_argument("--min_points", type=int, default=5)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    freq, zre, zim, chosen_block, cols, hdr = load_eis_with_rescue_and_fallback(
        xlsx_path=args.xlsx,
        sheet_name=args.sheet,
        requested_block=args.block,
        header_row=args.header,
        imag_is_negative=args.imag_is_negative,
        min_points=args.min_points,
        assume_mohm=args.assume_mohm,
    )

    j_f, j_r, j_i = cols
    print(f"\n[INFO] Header row used = {hdr}")
    print(f"[INFO] chosen_block = {chosen_block} (requested={args.block})")
    print(f"[INFO] Using RAW column indices: freq={j_f} | real={j_r} | imag={j_i}")
    print(f"[INFO] Valid rows N = {len(freq)}")
    print("[INFO] freq[:5] =", freq[:5])
    print("[INFO] zre[:5]  =", zre[:5])
    print("[INFO] zim[:5]  =", zim[:5])

    # -------------- 一阶RC --------------
    # # Fit ECM
    # zim = -zim
    # Z = zre + 1j * zim
    # guess = parse_guess(args.guess)
    # model = CustomCircuit(args.circuit, initial_guess=guess)

    # model.fit(freq, Z)
    # params = model.parameters_
    # -------------- 一阶RC --------------




    # -------------- 二阶RC --------------
    # -----------------------------
    # Fit ECM (2-RC) with better guesses
    # -----------------------------

    # 1) sort by frequency (important)
    order = np.argsort(freq)
    freq = freq[order]
    zre = zre[order]
    zim = zim[order]

    # 2) Nyquist convention:
    #    if your data zim is Im(Z), Nyquist uses -Im(Z).
    #    Here we build Z with Im(Z) = (-zim) so that plotting -Im(Z) is consistent.
    zim_for_Z = -zim
    Z = zre + 1j * zim_for_Z

    # 3) auto initial guess for 2-RC
    #    R0 ~ high-frequency intercept ~ min(Re)
    R0_0 = float(np.nanmin(zre))

    # total span in Re
    dR = float(np.nanmax(zre) - np.nanmin(zre))
    if not np.isfinite(dR) or dR <= 0:
        dR = max(1e-12, abs(R0_0) * 0.5)

    # split the polarization resistance into two arcs
    R1_0 = 0.6 * dR
    R2_0 = 0.4 * dR

    # pick characteristic frequencies (rough): use quartiles
    # (you can tune these; they're just to get C in the right ballpark)
    f1 = float(np.nanpercentile(freq, 70))  # higher freq -> smaller tau
    f2 = float(np.nanpercentile(freq, 30))  # lower freq  -> larger tau
    f1 = max(f1, 1e-6)
    f2 = max(f2, 1e-6)

    # C ≈ 1/(2π R f_peak)
    C1_0 = 1.0 / (2.0 * np.pi * max(R1_0, 1e-12) * f1)
    C2_0 = 1.0 / (2.0 * np.pi * max(R2_0, 1e-12) * f2)

    # If user provides --guess explicitly, respect it; else use auto
    if args.guess.strip():
        guess = parse_guess(args.guess)
    else:
        guess = [R0_0, R1_0, C1_0, R2_0, C2_0]

    # 4) enforce 2-RC circuit unless user overrides
    circuit = args.circuit or "R0-p(R1,C1)-p(R2,C2)"

    print("[DEBUG] AutoGuess (2RC):", guess)
    print("[DEBUG] Circuit:", circuit)

    model = CustomCircuit(circuit, initial_guess=guess)
    model.fit(freq, Z)
    params = model.parameters_
    # -------------- 二阶RC --------------





    print("\n=== Fit result ===")
    print("Circuit:", args.circuit)
    print("Params:", params)

    Z_fit = model.predict(freq)

    out_png = os.path.join(
        args.out_dir,
        f"nyquist_fit__{str(args.sheet)}__block{chosen_block}.png"
    )
    title = f"Nyquist + Fit | sheet={args.sheet} | block={chosen_block}"
    save_nyquist(Z, Z_fit, out_png, title)

    print(f"\n[INFO] Saved plot -> {out_png}")


if __name__ == "__main__":
    main()