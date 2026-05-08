#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover
    curve_fit = None

from parse_raw_maccor import parse_one_raw_file


def biexp_relax_model(t: np.ndarray, v_inf: float, a1: float, tau1: float, a2: float, tau2: float) -> np.ndarray:
    return v_inf + a1 * np.exp(-t / np.maximum(tau1, 1e-9)) + a2 * np.exp(-t / np.maximum(tau2, 1e-9))


def biexp_warburg_relax_model(
    t: np.ndarray,
    v_inf: float,
    a1: float,
    tau1: float,
    a2: float,
    tau2: float,
    b_w: float,
) -> np.ndarray:
    return (
        v_inf
        + a1 * np.exp(-t / np.maximum(tau1, 1e-9))
        + a2 * np.exp(-t / np.maximum(tau2, 1e-9))
        + b_w * np.sqrt(np.maximum(t, 0.0))
    )


def slope_vs_sqrt_t(t_s: np.ndarray, v: np.ndarray, t_min_s: float = 5.0, t_max_s: float = 120.0) -> float:
    mask = np.isfinite(t_s) & np.isfinite(v) & (t_s >= t_min_s) & (t_s <= t_max_s)
    if mask.sum() < 5:
        return float("nan")
    x = np.sqrt(t_s[mask])
    y = v[mask]
    try:
        coef = np.polyfit(x, y, 1)
        return float(coef[0])
    except Exception:
        return float("nan")


def fit_biexponential_relaxation(t_s: np.ndarray, v: np.ndarray) -> Dict[str, float]:
    out = {
        "feat_dev_v_inf_v": float("nan"),
        "feat_dev_a1_v": float("nan"),
        "feat_dev_tau1_s": float("nan"),
        "feat_dev_a2_v": float("nan"),
        "feat_dev_tau2_s": float("nan"),
        "feat_dev_relax_fit_rmse_v": float("nan"),
    }
    if curve_fit is None:
        return out
    mask = np.isfinite(t_s) & np.isfinite(v)
    if mask.sum() < 8:
        return out
    x = t_s[mask]
    y = v[mask]
    v_inf0 = float(np.nanmedian(y[-min(5, len(y)):]))
    amp = float(y[0] - v_inf0)
    p0 = [v_inf0, 0.6 * amp, 20.0, 0.4 * amp, 200.0]
    bounds = (
        [min(y) - 1.0, -10.0, 1.0, -10.0, 5.0],
        [max(y) + 1.0, 10.0, 5_000.0, 10.0, 50_000.0],
    )
    try:
        popt, _ = curve_fit(biexp_relax_model, x, y, p0=p0, bounds=bounds, maxfev=20_000)
        yhat = biexp_relax_model(x, *popt)
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        out.update(
            {
                "feat_dev_v_inf_v": float(popt[0]),
                "feat_dev_a1_v": float(popt[1]),
                "feat_dev_tau1_s": float(popt[2]),
                "feat_dev_a2_v": float(popt[3]),
                "feat_dev_tau2_s": float(popt[4]),
                "feat_dev_relax_fit_rmse_v": rmse,
            }
        )
    except Exception:
        pass
    return out


def fit_biexponential_warburg_relaxation(t_s: np.ndarray, v: np.ndarray) -> Dict[str, float]:
    out = {
        "feat_dev_joint_v_inf_v": float("nan"),
        "feat_dev_joint_a1_v": float("nan"),
        "feat_dev_joint_tau1_s": float("nan"),
        "feat_dev_joint_a2_v": float("nan"),
        "feat_dev_joint_tau2_s": float("nan"),
        "feat_dev_joint_B_warburg_proxy": float("nan"),
        "feat_dev_joint_fit_rmse_v": float("nan"),
    }
    if curve_fit is None:
        return out
    mask = np.isfinite(t_s) & np.isfinite(v)
    if mask.sum() < 8:
        return out
    x = t_s[mask]
    y = v[mask]
    v_inf0 = float(np.nanmedian(y[-min(5, len(y)):]))
    amp = float(y[0] - v_inf0)
    slope0 = slope_vs_sqrt_t(x, y)
    if not np.isfinite(slope0):
        slope0 = 0.0
    p0 = [v_inf0, 0.6 * amp, 20.0, 0.4 * amp, 200.0, slope0]
    bounds = (
        [min(y) - 1.0, -10.0, 1.0, -10.0, 5.0, -10.0],
        [max(y) + 1.0, 10.0, 5_000.0, 10.0, 50_000.0, 10.0],
    )
    try:
        popt, _ = curve_fit(
            biexp_warburg_relax_model,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=40_000,
        )
        yhat = biexp_warburg_relax_model(x, *popt)
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        out.update(
            {
                "feat_dev_joint_v_inf_v": float(popt[0]),
                "feat_dev_joint_a1_v": float(popt[1]),
                "feat_dev_joint_tau1_s": float(popt[2]),
                "feat_dev_joint_a2_v": float(popt[3]),
                "feat_dev_joint_tau2_s": float(popt[4]),
                "feat_dev_joint_B_warburg_proxy": float(popt[5]),
                "feat_dev_joint_fit_rmse_v": rmse,
            }
        )
    except Exception:
        pass
    return out


def contiguous_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif (not val) and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def detect_relaxation_segments(
    df: pd.DataFrame,
    rest_current_a: float = 0.05,
    min_step_current_a: float = 0.5,
    min_relax_s: float = 60.0,
    pre_window_pts: int = 10,
) -> List[Dict[str, float]]:
    if df.empty:
        return []
    work = df.copy()
    if "test_time_s" not in work.columns or "current_a" not in work.columns or "voltage_v" not in work.columns:
        return []
    work = work.sort_values("test_time_s").reset_index(drop=True)
    t = pd.to_numeric(work["test_time_s"], errors="coerce").to_numpy(dtype=float)
    i = pd.to_numeric(work["current_a"], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(work["voltage_v"], errors="coerce").to_numpy(dtype=float)
    cyc = pd.to_numeric(work.get("cycle_c"), errors="coerce").to_numpy(dtype=float) if "cycle_c" in work.columns else np.full(len(work), np.nan)

    rest_mask = np.isfinite(i) & (np.abs(i) <= rest_current_a)
    runs = contiguous_true_runs(rest_mask)
    segments: List[Dict[str, float]] = []
    for start, end in runs:
        if end - start < 5:
            continue
        if not np.isfinite(t[start]) or not np.isfinite(t[end - 1]):
            continue
        dur = float(t[end - 1] - t[start])
        if dur < min_relax_s:
            continue
        prev_lo = max(0, start - pre_window_pts)
        prev_i = i[prev_lo:start]
        prev_v = v[prev_lo:start]
        prev_cyc = cyc[prev_lo:start]
        prev_i = prev_i[np.isfinite(prev_i)]
        prev_v = prev_v[np.isfinite(prev_v)]
        prev_cyc = prev_cyc[np.isfinite(prev_cyc)]
        if len(prev_i) == 0:
            continue
        i_prev = float(np.nanmedian(prev_i))
        if abs(i_prev) < min_step_current_a:
            continue
        v_before = float(np.nanmedian(prev_v[-min(3, len(prev_v)):])) if len(prev_v) else float("nan")
        v_start = float(v[start]) if np.isfinite(v[start]) else float("nan")
        cycle_c = float(np.nanmedian(prev_cyc)) if len(prev_cyc) else float(cyc[start]) if np.isfinite(cyc[start]) else float("nan")
        segments.append(
            {
                "start_idx": start,
                "end_idx": end,
                "duration_s": dur,
                "i_prev_a": i_prev,
                "v_before_v": v_before,
                "v_start_v": v_start,
                "cycle_c": cycle_c,
            }
        )
    return segments


def summarize_relaxation_segment(df: pd.DataFrame, seg: Dict[str, float]) -> Dict[str, float]:
    start = int(seg["start_idx"])
    end = int(seg["end_idx"])
    sub = df.iloc[start:end].copy()
    t = pd.to_numeric(sub["test_time_s"], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(sub["voltage_v"], errors="coerce").to_numpy(dtype=float)
    if len(sub) == 0 or not np.isfinite(t).any() or not np.isfinite(v).any():
        return {}
    t0 = float(t[0])
    t_rel = t - t0
    out: Dict[str, float] = {
        "cycle_c": float(seg["cycle_c"]),
        "feat_dev_relax_duration_s": float(seg["duration_s"]),
        "feat_dev_pre_current_a": float(seg["i_prev_a"]),
        "feat_dev_pre_voltage_v": float(seg["v_before_v"]),
        "feat_dev_relax_start_voltage_v": float(seg["v_start_v"]),
        "feat_dev_r0_proxy_ohm": (
            float((seg["v_before_v"] - seg["v_start_v"]) / abs(seg["i_prev_a"]))
            if np.isfinite(seg["v_before_v"]) and np.isfinite(seg["v_start_v"]) and abs(seg["i_prev_a"]) > 1e-9
            else float("nan")
        ),
        "feat_dev_sigma_proxy_v_per_sqrt_s": slope_vs_sqrt_t(t_rel, v),
    }

    for sec in [10.0, 30.0, 60.0, 120.0]:
        mask = np.isfinite(t_rel) & np.isfinite(v) & (t_rel <= sec)
        if mask.any():
            out[f"feat_dev_relax_dv_{int(sec)}s_v"] = float(v[0] - v[mask][-1])
        else:
            out[f"feat_dev_relax_dv_{int(sec)}s_v"] = float("nan")

    fit = fit_biexponential_relaxation(t_rel, v)
    out.update(fit)
    joint_fit = fit_biexponential_warburg_relaxation(t_rel, v)
    out.update(joint_fit)

    i_abs = abs(seg["i_prev_a"])
    if i_abs > 1e-9:
        if np.isfinite(out["feat_dev_a1_v"]):
            out["feat_dev_R1_proxy_ohm"] = float(abs(out["feat_dev_a1_v"]) / i_abs)
        if np.isfinite(out["feat_dev_a2_v"]):
            out["feat_dev_R2_proxy_ohm"] = float(abs(out["feat_dev_a2_v"]) / i_abs)
        rvals = [out.get("feat_dev_r0_proxy_ohm", np.nan), out.get("feat_dev_R1_proxy_ohm", np.nan), out.get("feat_dev_R2_proxy_ohm", np.nan)]
        rvals = [float(x) for x in rvals if np.isfinite(x)]
        out["feat_dev_R_total_proxy_ohm"] = float(sum(rvals)) if rvals else float("nan")
        if np.isfinite(out["feat_dev_joint_a1_v"]):
            out["feat_dev_joint_R1_proxy_ohm"] = float(abs(out["feat_dev_joint_a1_v"]) / i_abs)
        else:
            out["feat_dev_joint_R1_proxy_ohm"] = float("nan")
        if np.isfinite(out["feat_dev_joint_a2_v"]):
            out["feat_dev_joint_R2_proxy_ohm"] = float(abs(out["feat_dev_joint_a2_v"]) / i_abs)
        else:
            out["feat_dev_joint_R2_proxy_ohm"] = float("nan")
        joint_rvals = [
            out.get("feat_dev_r0_proxy_ohm", np.nan),
            out.get("feat_dev_joint_R1_proxy_ohm", np.nan),
            out.get("feat_dev_joint_R2_proxy_ohm", np.nan),
        ]
        joint_rvals = [float(x) for x in joint_rvals if np.isfinite(x)]
        out["feat_dev_joint_R_total_proxy_ohm"] = float(sum(joint_rvals)) if joint_rvals else float("nan")
    else:
        out["feat_dev_R1_proxy_ohm"] = float("nan")
        out["feat_dev_R2_proxy_ohm"] = float("nan")
        out["feat_dev_R_total_proxy_ohm"] = float("nan")
        out["feat_dev_joint_R1_proxy_ohm"] = float("nan")
        out["feat_dev_joint_R2_proxy_ohm"] = float("nan")
        out["feat_dev_joint_R_total_proxy_ohm"] = float("nan")
    return out


def extract_device_ecm_features_for_file(path: Path, choose: str = "longest") -> pd.DataFrame:
    df = parse_one_raw_file(path)
    if df.empty:
        return pd.DataFrame()
    out_rows: List[Dict[str, float]] = []
    for cycle_c, sub in df.groupby("cycle_c", dropna=True, sort=True):
        segs = detect_relaxation_segments(sub)
        if not segs:
            continue
        seg = max(segs, key=lambda x: x["duration_s"]) if choose == "longest" else segs[0]
        feat = summarize_relaxation_segment(sub.sort_values("test_time_s").reset_index(drop=True), seg)
        if not feat:
            continue
        feat["serial_norm"] = str(df["serial_norm"].iloc[0]).strip().upper()
        feat["serial"] = str(df["serial"].iloc[0]).strip().upper()
        feat["source_file"] = path.name
        feat["group_tag"] = str(df["group_tag"].iloc[0])
        out_rows.append(feat)
    return pd.DataFrame(out_rows)


def merge_device_ecm_features(feature_df: pd.DataFrame, device_df: pd.DataFrame, align_mode: str = "last_le") -> pd.DataFrame:
    if feature_df.empty or device_df.empty:
        return feature_df.copy()

    work = feature_df.copy()
    work["_rowid"] = np.arange(len(work))
    work["serial_norm"] = work["serial"].astype(str).str.strip().str.upper()
    work["cycle_t_num"] = pd.to_numeric(work["cycle_t"], errors="coerce")
    valid = work[pd.notna(work["cycle_t_num"])].copy()
    invalid = work[pd.isna(work["cycle_t_num"])].copy()
    valid["cycle_t_num"] = valid["cycle_t_num"].astype(float)

    dev = device_df.copy()
    dev["serial_norm"] = dev["serial_norm"].astype(str).str.strip().str.upper()
    dev["cycle_c"] = pd.to_numeric(dev["cycle_c"], errors="coerce")
    dev = dev[pd.notna(dev["cycle_c"])].copy().sort_values(["serial_norm", "cycle_c"])

    merged_parts: List[pd.DataFrame] = []
    for serial, sub_ft in valid.groupby("serial_norm", sort=False):
        sub_ft = sub_ft.sort_values("cycle_t_num")
        sub_dev = dev[dev["serial_norm"] == serial].sort_values("cycle_c")
        if sub_dev.empty:
            merged_parts.append(sub_ft.copy())
            continue
        direction = "backward" if align_mode == "last_le" else "nearest"
        m = pd.merge_asof(
            sub_ft,
            sub_dev,
            left_on="cycle_t_num",
            right_on="cycle_c",
            direction=direction,
            allow_exact_matches=True,
            suffixes=("", "_device"),
        )
        if align_mode == "exact":
            mismatch = ~np.isclose(m["cycle_t_num"].astype(float), pd.to_numeric(m["cycle_c"], errors="coerce").astype(float), equal_nan=False)
            device_cols = [c for c in m.columns if c.startswith("feat_dev_")] + ["cycle_c"]
            for c in device_cols:
                if c in m.columns:
                    m.loc[mismatch, c] = np.nan
        merged_parts.append(m)

    merged = pd.concat(merged_parts + [invalid], ignore_index=True, sort=False)
    merged = merged.sort_values("_rowid").drop(columns=["_rowid"]).reset_index(drop=True)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract device-side ECM proxy features from raw Maccor time-domain data, with optional merge into an existing feature table."
    )
    ap.add_argument("--raw_dir", required=True, help="Directory containing raw Maccor files.")
    ap.add_argument("--out_csv", required=True, help="Output CSV of per-cycle device ECM proxy features.")
    ap.add_argument("--feature_table_csv", default="", help="Optional feature table to augment.")
    ap.add_argument("--out_feature_table_csv", default="", help="Optional output merged feature table.")
    ap.add_argument("--align_mode", default="last_le", choices=["last_le", "exact"], help="How to align device-cycle features to cycle_t when merging.")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    files = [p for p in sorted(raw_dir.rglob("*")) if p.is_file() and not p.name.startswith(".")]
    if not files:
        raise ValueError(f"No files found under: {raw_dir}")

    frames: List[pd.DataFrame] = []
    bad: List[Tuple[str, str]] = []
    for p in files:
        try:
            one = extract_device_ecm_features_for_file(p)
            if not one.empty:
                frames.append(one)
        except Exception as exc:
            bad.append((p.name, str(exc)))

    out_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved device ECM proxy CSV: {out_path}")
    print(f"[INFO] Rows={len(out_df)}")
    print(f"[INFO] Bad files={len(bad)}")
    if bad:
        for f, e in bad[:10]:
            print(f"[WARN] {f} -> {e}")

    if args.feature_table_csv or args.out_feature_table_csv:
        if not args.feature_table_csv or not args.out_feature_table_csv:
            raise ValueError("Both --feature_table_csv and --out_feature_table_csv are required for merge.")
        ft = pd.read_csv(args.feature_table_csv)
        merged = merge_device_ecm_features(ft, out_df, align_mode=args.align_mode)
        out_ft = Path(args.out_feature_table_csv)
        out_ft.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_ft, index=False)
        print(f"[INFO] Saved merged feature table: {out_ft}")


if __name__ == "__main__":
    main()
