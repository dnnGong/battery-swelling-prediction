#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import curve_fit, least_squares
except Exception:  # pragma: no cover
    curve_fit = None
    least_squares = None

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


def maybe_downsample_pair(t_s: np.ndarray, v: np.ndarray, max_points: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(t_s) & np.isfinite(v)
    x = np.asarray(t_s[mask], dtype=float)
    y = np.asarray(v[mask], dtype=float)
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
    return x[idx], y[idx]


def rc_chain_relax_model(
    t: np.ndarray,
    v_inf: float,
    a_sei: float,
    tau_sei: float,
    a_w1: float,
    tau_w1: float,
    a_w2: float,
    tau_w2: float,
) -> np.ndarray:
    return (
        v_inf
        + a_sei * np.exp(-t / np.maximum(tau_sei, 1e-9))
        + a_w1 * np.exp(-t / np.maximum(tau_w1, 1e-9))
        + a_w2 * np.exp(-t / np.maximum(tau_w2, 1e-9))
    )


def _finite_or(x: float, fallback: float) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(fallback)


def _bounded_triplet(init: float, lb: float, ub: float, fallback_lb: float, fallback_ub: float) -> Tuple[float, float, float]:
    init_v = _finite_or(init, np.nan)
    lb_v = _finite_or(lb, fallback_lb)
    ub_v = _finite_or(ub, fallback_ub)
    if not np.isfinite(init_v):
        init_v = 0.5 * (lb_v + ub_v)
    init_v = min(max(init_v, lb_v), ub_v)
    return float(init_v), float(lb_v), float(ub_v)


def _match_prior_row_for_cycle(
    serial_norm: str,
    cycle_c: float,
    prior_df: Optional[pd.DataFrame],
    align_mode: str = "last_le",
) -> Optional[pd.Series]:
    if prior_df is None or prior_df.empty:
        return None
    pri = prior_df.copy()
    pri["serial_norm"] = pri["serial_norm"].astype(str).str.strip().str.upper()
    sub = pri[pri["serial_norm"] == str(serial_norm).strip().upper()].copy()
    if sub.empty:
        return None
    sub["measurement_cycle_num"] = pd.to_numeric(sub.get("measurement_cycle"), errors="coerce")
    cyc_rows = sub[pd.notna(sub["measurement_cycle_num"])].sort_values("measurement_cycle_num")
    if not cyc_rows.empty:
        if align_mode == "exact":
            exact = cyc_rows[np.isclose(cyc_rows["measurement_cycle_num"].astype(float), float(cycle_c))]
            if not exact.empty:
                return exact.iloc[-1]
        elif align_mode == "last_le":
            le = cyc_rows[cyc_rows["measurement_cycle_num"].astype(float) <= float(cycle_c)]
            if not le.empty:
                return le.iloc[-1]
            return cyc_rows.iloc[0]
        else:
            return cyc_rows.iloc[-1]
    return sub.iloc[0]


def fit_constrained_rc_chain_relaxation(
    t_s: np.ndarray,
    v: np.ndarray,
    i_prev_a: float,
    prior_row: Optional[pd.Series] = None,
) -> Dict[str, float]:
    out = {
        "feat_dev_td_v_inf_v": float("nan"),
        "feat_dev_td_a_sei_v": float("nan"),
        "feat_dev_td_tau_sei_s": float("nan"),
        "feat_dev_td_a_w1_v": float("nan"),
        "feat_dev_td_tau_w1_s": float("nan"),
        "feat_dev_td_a_w2_v": float("nan"),
        "feat_dev_td_tau_w2_s": float("nan"),
        "feat_dev_td_fit_rmse_v": float("nan"),
        "feat_dev_td_fit_status": "not_run",
        "feat_dev_td_prior_used": 0.0,
        "feat_dev_td_prior_cycle_used": float("nan"),
    }
    if least_squares is None:
        out["feat_dev_td_fit_status"] = "scipy_unavailable"
        return out
    x, y = maybe_downsample_pair(t_s, v, max_points=400)
    if len(x) < 8:
        out["feat_dev_td_fit_status"] = "insufficient_points"
        return out
    v_inf0 = float(np.nanmedian(y[-min(5, len(y)):]))
    amp = float(y[0] - v_inf0)
    i_abs = abs(float(i_prev_a))
    amp_abs = abs(amp)

    # Generic fallback bounds.
    rsei_guess = amp_abs * 0.35 / max(i_abs, 1e-6)
    rw1_guess = amp_abs * 0.35 / max(i_abs, 1e-6)
    rw2_guess = amp_abs * 0.30 / max(i_abs, 1e-6)
    tau_sei_guess = 20.0
    tau_w1_guess = 120.0
    tau_w2_guess = 600.0

    prior_cycle_used = float("nan")
    if prior_row is not None:
        prior_cycle_used = _finite_or(prior_row.get("measurement_cycle", np.nan), np.nan)
        rsei_guess, rsei_lb, rsei_ub = _bounded_triplet(
            prior_row.get("prior_Rsei_init", np.nan),
            prior_row.get("prior_Rsei_lb", np.nan),
            prior_row.get("prior_Rsei_ub", np.nan),
            max(1e-6, 0.25 * rsei_guess),
            max(1e-5, 4.0 * rsei_guess),
        )
        tau_sei_guess, tau_sei_lb, tau_sei_ub = _bounded_triplet(
            prior_row.get("prior_tau_sei_init", np.nan),
            prior_row.get("prior_tau_sei_lb", np.nan),
            prior_row.get("prior_tau_sei_ub", np.nan),
            1e-3,
            2_000.0,
        )
        rdl_init = _finite_or(prior_row.get("prior_Rdl_init", np.nan), np.nan)
        rdl_lb = _finite_or(prior_row.get("prior_Rdl_lb", np.nan), np.nan)
        rdl_ub = _finite_or(prior_row.get("prior_Rdl_ub", np.nan), np.nan)
        if np.isfinite(rdl_init):
            rw1_guess = 0.6 * rdl_init
            rw2_guess = 0.4 * rdl_init
        if np.isfinite(rdl_lb) and np.isfinite(rdl_ub):
            rw1_lb, rw1_ub = max(1e-8, 0.25 * rdl_lb), max(1e-7, 1.2 * rdl_ub)
            rw2_lb, rw2_ub = max(1e-8, 0.10 * rdl_lb), max(1e-7, 1.2 * rdl_ub)
        else:
            rw1_lb, rw1_ub = max(1e-8, 0.25 * rw1_guess), max(1e-7, 4.0 * rw1_guess)
            rw2_lb, rw2_ub = max(1e-8, 0.25 * rw2_guess), max(1e-7, 4.0 * rw2_guess)
        tau_dl_init = _finite_or(prior_row.get("prior_tau_dl_init", np.nan), np.nan)
        if np.isfinite(tau_dl_init):
            tau_w1_guess = max(5.0, 0.3 * tau_dl_init)
            tau_w2_guess = max(20.0, 1.2 * tau_dl_init)
        tau_w1_lb, tau_w1_ub = 5.0, 5_000.0
        tau_w2_lb, tau_w2_ub = 20.0, 50_000.0
        out["feat_dev_td_prior_used"] = 1.0
        out["feat_dev_td_prior_cycle_used"] = prior_cycle_used
    else:
        rsei_lb, rsei_ub = max(1e-8, 0.25 * rsei_guess), max(1e-7, 4.0 * rsei_guess)
        rw1_lb, rw1_ub = max(1e-8, 0.25 * rw1_guess), max(1e-7, 4.0 * rw1_guess)
        rw2_lb, rw2_ub = max(1e-8, 0.25 * rw2_guess), max(1e-7, 4.0 * rw2_guess)
        tau_sei_lb, tau_sei_ub = 1e-3, 2_000.0
        tau_w1_lb, tau_w1_ub = 5.0, 5_000.0
        tau_w2_lb, tau_w2_ub = 20.0, 50_000.0

    p0 = np.array(
        [
            v_inf0,
            -i_abs * rsei_guess,
            tau_sei_guess,
            -i_abs * rw1_guess,
            tau_w1_guess,
            -i_abs * rw2_guess,
            tau_w2_guess,
        ],
        dtype=float,
    )
    lb = np.array(
        [
            float(np.nanmin(y) - 1.0),
            -i_abs * rsei_ub,
            tau_sei_lb,
            -i_abs * rw1_ub,
            tau_w1_lb,
            -i_abs * rw2_ub,
            tau_w2_lb,
        ],
        dtype=float,
    )
    ub = np.array(
        [
            float(np.nanmax(y) + 1.0),
            i_abs * rsei_ub,
            tau_sei_ub,
            i_abs * rw1_ub,
            tau_w1_ub,
            i_abs * rw2_ub,
            tau_w2_ub,
        ],
        dtype=float,
    )
    p0 = np.minimum(np.maximum(p0, lb), ub)

    def residual(theta: np.ndarray) -> np.ndarray:
        yhat = rc_chain_relax_model(x, *theta)
        return yhat - y

    try:
        res = least_squares(residual, p0, bounds=(lb, ub), max_nfev=20_000)
        yhat = rc_chain_relax_model(x, *res.x)
        rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
        out.update(
            {
                "feat_dev_td_v_inf_v": float(res.x[0]),
                "feat_dev_td_a_sei_v": float(res.x[1]),
                "feat_dev_td_tau_sei_s": float(res.x[2]),
                "feat_dev_td_a_w1_v": float(res.x[3]),
                "feat_dev_td_tau_w1_s": float(res.x[4]),
                "feat_dev_td_a_w2_v": float(res.x[5]),
                "feat_dev_td_tau_w2_s": float(res.x[6]),
                "feat_dev_td_fit_rmse_v": rmse,
                "feat_dev_td_fit_status": str(res.status),
            }
        )
    except Exception:
        out["feat_dev_td_fit_status"] = "fit_failed"
    return out


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
    x, y = maybe_downsample_pair(t_s, v, max_points=400)
    if len(x) < 8:
        return out
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
    x, y = maybe_downsample_pair(t_s, v, max_points=400)
    if len(x) < 8:
        return out
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


def summarize_relaxation_segment(
    df: pd.DataFrame,
    seg: Dict[str, float],
    prior_row: Optional[pd.Series] = None,
    fit_mode: str = "full",
) -> Dict[str, float]:
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

    if fit_mode == "full":
        fit = fit_biexponential_relaxation(t_rel, v)
        out.update(fit)
        joint_fit = fit_biexponential_warburg_relaxation(t_rel, v)
        out.update(joint_fit)
    else:
        out.update(
            {
                "feat_dev_v_inf_v": float("nan"),
                "feat_dev_a1_v": float("nan"),
                "feat_dev_tau1_s": float("nan"),
                "feat_dev_a2_v": float("nan"),
                "feat_dev_tau2_s": float("nan"),
                "feat_dev_relax_fit_rmse_v": float("nan"),
                "feat_dev_joint_v_inf_v": float("nan"),
                "feat_dev_joint_a1_v": float("nan"),
                "feat_dev_joint_tau1_s": float("nan"),
                "feat_dev_joint_a2_v": float("nan"),
                "feat_dev_joint_tau2_s": float("nan"),
                "feat_dev_joint_B_warburg_proxy": float("nan"),
                "feat_dev_joint_fit_rmse_v": float("nan"),
            }
        )
    td_fit = fit_constrained_rc_chain_relaxation(t_rel, v, seg["i_prev_a"], prior_row=prior_row)
    out.update(td_fit)

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
        out["feat_dev_td_Rsei_ohm"] = float(abs(out["feat_dev_td_a_sei_v"]) / i_abs) if np.isfinite(out.get("feat_dev_td_a_sei_v", np.nan)) else float("nan")
        out["feat_dev_td_Rw1_ohm"] = float(abs(out["feat_dev_td_a_w1_v"]) / i_abs) if np.isfinite(out.get("feat_dev_td_a_w1_v", np.nan)) else float("nan")
        out["feat_dev_td_Rw2_ohm"] = float(abs(out["feat_dev_td_a_w2_v"]) / i_abs) if np.isfinite(out.get("feat_dev_td_a_w2_v", np.nan)) else float("nan")
        td_rvals = [
            out.get("feat_dev_r0_proxy_ohm", np.nan),
            out.get("feat_dev_td_Rsei_ohm", np.nan),
            out.get("feat_dev_td_Rw1_ohm", np.nan),
            out.get("feat_dev_td_Rw2_ohm", np.nan),
        ]
        td_rvals = [float(x) for x in td_rvals if np.isfinite(x)]
        out["feat_dev_td_R_diff_total_ohm"] = float(
            sum(float(x) for x in [out.get("feat_dev_td_Rw1_ohm", np.nan), out.get("feat_dev_td_Rw2_ohm", np.nan)] if np.isfinite(x))
        )
        out["feat_dev_td_R_total_proxy_ohm"] = float(sum(td_rvals)) if td_rvals else float("nan")
    else:
        out["feat_dev_R1_proxy_ohm"] = float("nan")
        out["feat_dev_R2_proxy_ohm"] = float("nan")
        out["feat_dev_R_total_proxy_ohm"] = float("nan")
        out["feat_dev_joint_R1_proxy_ohm"] = float("nan")
        out["feat_dev_joint_R2_proxy_ohm"] = float("nan")
        out["feat_dev_joint_R_total_proxy_ohm"] = float("nan")
        out["feat_dev_td_Rsei_ohm"] = float("nan")
        out["feat_dev_td_Rw1_ohm"] = float("nan")
        out["feat_dev_td_Rw2_ohm"] = float("nan")
        out["feat_dev_td_R_diff_total_ohm"] = float("nan")
        out["feat_dev_td_R_total_proxy_ohm"] = float("nan")
    return out


def extract_device_ecm_features_for_file(
    path: Path,
    choose: str = "longest",
    prior_df: Optional[pd.DataFrame] = None,
    prior_align_mode: str = "last_le",
    fit_mode: str = "full",
    cycle_mode: str = "all",
) -> pd.DataFrame:
    df = parse_one_raw_file(path)
    if df.empty:
        return pd.DataFrame()
    serial_norm = str(df["serial_norm"].iloc[0]).strip().upper()
    out_rows: List[Dict[str, float]] = []
    grouped = list(df.groupby("cycle_c", dropna=True, sort=True))
    if cycle_mode == "last" and grouped:
        grouped = [grouped[-1]]
    elif cycle_mode == "first" and grouped:
        grouped = [grouped[0]]
    for cycle_c, sub in grouped:
        segs = detect_relaxation_segments(sub)
        if not segs:
            continue
        seg = max(segs, key=lambda x: x["duration_s"]) if choose == "longest" else segs[0]
        prior_row = _match_prior_row_for_cycle(serial_norm, float(cycle_c), prior_df, align_mode=prior_align_mode)
        feat = summarize_relaxation_segment(
            sub.sort_values("test_time_s").reset_index(drop=True),
            seg,
            prior_row=prior_row,
            fit_mode=fit_mode,
        )
        if not feat:
            continue
        feat["serial_norm"] = serial_norm
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


def merge_eis_time_domain_priors(device_df: pd.DataFrame, prior_df: pd.DataFrame, align_mode: str = "last_le") -> pd.DataFrame:
    if device_df.empty or prior_df.empty:
        return device_df.copy()

    work = device_df.copy()
    work["_rowid"] = np.arange(len(work))
    work["serial_norm"] = work["serial_norm"].astype(str).str.strip().str.upper()
    work["cycle_c_num"] = pd.to_numeric(work["cycle_c"], errors="coerce")
    valid = work[pd.notna(work["cycle_c_num"])].copy()
    invalid = work[pd.isna(work["cycle_c_num"])].copy()
    valid["cycle_c_num"] = valid["cycle_c_num"].astype(float)

    pri_all = prior_df.copy()
    pri_all["serial_norm"] = pri_all["serial_norm"].astype(str).str.strip().str.upper()
    pri_all["measurement_cycle"] = pd.to_numeric(pri_all["measurement_cycle"], errors="coerce")
    pri_cycle = pri_all[pd.notna(pri_all["measurement_cycle"])].copy().sort_values(["serial_norm", "measurement_cycle"])
    pri_fallback = pri_all.sort_values(["serial_norm"]).copy()

    merged_parts: List[pd.DataFrame] = []
    for serial, sub_dev in valid.groupby("serial_norm", sort=False):
        sub_dev = sub_dev.sort_values("cycle_c_num")
        sub_pri = pri_cycle[pri_cycle["serial_norm"] == serial].sort_values("measurement_cycle")
        sub_pri_fallback = pri_fallback[pri_fallback["serial_norm"] == serial].copy()
        if sub_pri.empty and sub_pri_fallback.empty:
            merged_parts.append(sub_dev.copy())
            continue

        if not sub_pri.empty:
            direction = "backward" if align_mode == "last_le" else "nearest"
            m = pd.merge_asof(
                sub_dev,
                sub_pri,
                left_on="cycle_c_num",
                right_on="measurement_cycle",
                direction=direction,
                allow_exact_matches=True,
                suffixes=("", "_prior"),
            )
            if align_mode == "exact":
                mismatch = ~np.isclose(
                    m["cycle_c_num"].astype(float),
                    pd.to_numeric(m["measurement_cycle"], errors="coerce").astype(float),
                    equal_nan=False,
                )
                prior_cols = [c for c in m.columns if c.startswith("prior_")] + ["measurement_cycle", "sheet", "circuit", "rmse_complex_ohm"]
                for c in prior_cols:
                    if c in m.columns:
                        m.loc[mismatch, c] = np.nan
        else:
            # Fallback for serials that only have PreEIS-style priors without explicit measurement_cycle.
            base = sub_pri_fallback.iloc[[0]].copy()
            base = base.drop(columns=["serial_norm"], errors="ignore").reset_index(drop=True)
            m = sub_dev.reset_index(drop=True).copy()
            for c in base.columns:
                if c not in m.columns:
                    m[c] = base.at[0, c]
        merged_parts.append(m)

    merged = pd.concat(merged_parts + [invalid], ignore_index=True, sort=False)
    merged = merged.sort_values("_rowid").drop(columns=["_rowid"]).reset_index(drop=True)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract device-side ECM proxy features from raw Maccor time-domain data, with optional EIS-prior alignment and optional merge into an existing feature table."
    )
    ap.add_argument("--raw_dir", required=True, help="Directory containing raw Maccor files.")
    ap.add_argument("--out_csv", required=True, help="Output CSV of per-cycle device ECM proxy features.")
    ap.add_argument("--eis_prior_csv", default="", help="Optional CSV from src/build_eis_time_domain_priors.py to align onto device rows.")
    ap.add_argument("--prior_align_mode", default="last_le", choices=["last_le", "exact"], help="How to align EIS prior rows to raw-data cycle_c when --eis_prior_csv is provided.")
    ap.add_argument("--fit_mode", default="full", choices=["full", "td_only"], help="Whether to run all exploratory fits or only the mentor-style constrained time-domain fitter.")
    ap.add_argument("--cycle_mode", default="all", choices=["all", "first", "last"], help="Whether to extract features for all raw cycles or only one representative cycle per file.")
    ap.add_argument("--feature_table_csv", default="", help="Optional feature table to augment.")
    ap.add_argument("--out_feature_table_csv", default="", help="Optional output merged feature table.")
    ap.add_argument("--align_mode", default="last_le", choices=["last_le", "exact"], help="How to align device-cycle features to cycle_t when merging.")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    files = [p for p in sorted(raw_dir.rglob("*")) if p.is_file() and not p.name.startswith(".")]
    if not files:
        raise ValueError(f"No files found under: {raw_dir}")

    prior_df: Optional[pd.DataFrame] = None
    if args.eis_prior_csv:
        prior_csv = Path(args.eis_prior_csv)
        if not prior_csv.exists():
            raise FileNotFoundError(f"eis_prior_csv not found: {prior_csv}")
        prior_df = pd.read_csv(prior_csv)

    frames: List[pd.DataFrame] = []
    bad: List[Tuple[str, str]] = []
    for p in files:
        try:
            one = extract_device_ecm_features_for_file(
                p,
                prior_df=prior_df,
                prior_align_mode=args.prior_align_mode,
                fit_mode=args.fit_mode,
                cycle_mode=args.cycle_mode,
            )
            if not one.empty:
                frames.append(one)
        except Exception as exc:
            bad.append((p.name, str(exc)))

    out_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if prior_df is not None:
        out_df = merge_eis_time_domain_priors(out_df, prior_df, align_mode=args.prior_align_mode)
        print(f"[INFO] Merged EIS time-domain priors from: {args.eis_prior_csv}")

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
