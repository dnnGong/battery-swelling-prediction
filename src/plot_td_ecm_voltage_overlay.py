#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extract_device_ecm_features import (
    _select_prior_row,
    choose_relaxation_segment,
    detect_relaxation_segments,
    fit_constrained_rc_chain_relaxation,
    maybe_downsample_pair,
    parse_one_raw_file,
    rc_chain_relax_model,
    score_relaxation_segment,
)


def _is_success_status(status: object) -> bool:
    s = str(status).strip().lower()
    return s.isdigit()


def _candidate_from_file(
    path: Path,
    prior_df: Optional[pd.DataFrame],
    prior_align_mode: str,
    prior_mode: str,
    cycle_mode: str,
    segment_choice: str,
) -> Optional[Dict[str, object]]:
    df = parse_one_raw_file(path)
    if df.empty:
        return None

    serial_norm = str(df["serial_norm"].iloc[0]).strip().upper()
    grouped = list(df.groupby("cycle_c", dropna=True, sort=True))
    if not grouped:
        return None

    chosen: Optional[Tuple[float, pd.DataFrame, Dict[str, float], Tuple[int, float]]] = None
    if cycle_mode == "last":
        grouped = [grouped[-1]]
    elif cycle_mode == "first":
        grouped = [grouped[0]]
    elif cycle_mode == "best":
        for cycle_c, sub in grouped:
            sub_sorted = sub.sort_values("test_time_s").reset_index(drop=True)
            segs = detect_relaxation_segments(sub_sorted)
            if not segs:
                continue
            seg = choose_relaxation_segment(sub_sorted, segs, strategy=segment_choice)
            score = score_relaxation_segment(sub_sorted, seg)
            item = (float(cycle_c), sub_sorted, seg, score)
            if chosen is None or item[3] > chosen[3]:
                chosen = item
        grouped = [(chosen[0], chosen[1])] if chosen is not None else []

    best_local: Optional[Dict[str, object]] = None
    for cycle_c, sub in grouped:
        sub_sorted = sub.sort_values("test_time_s").reset_index(drop=True)
        segs = detect_relaxation_segments(sub_sorted)
        if not segs:
            continue
        seg = choose_relaxation_segment(sub_sorted, segs, strategy=segment_choice)
        prior_row = _select_prior_row(
            serial_norm,
            float(cycle_c),
            prior_df,
            align_mode=prior_align_mode,
            prior_mode=prior_mode,
            group_tag=str(df["group_tag"].iloc[0]),
        )
        start = int(seg["start_idx"])
        end = int(seg["end_idx"])
        rel = sub_sorted.iloc[start:end].copy()
        if rel.empty:
            continue
        t = pd.to_numeric(rel["test_time_s"], errors="coerce").to_numpy(dtype=float)
        v = pd.to_numeric(rel["voltage_v"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(t).any() or not np.isfinite(v).any():
            continue
        t_rel = t - float(t[0])
        fit = fit_constrained_rc_chain_relaxation(t_rel, v, float(seg["i_prev_a"]), prior_row=prior_row)
        x, y = maybe_downsample_pair(t_rel, v, max_points=400)

        yhat = np.full_like(y, np.nan, dtype=float)
        if _is_success_status(fit.get("feat_dev_td_fit_status")):
            yhat = rc_chain_relax_model(
                x,
                float(fit["feat_dev_td_v_inf_v"]),
                float(fit["feat_dev_td_a_Rsei_v"]),
                float(fit["feat_dev_td_tau_Rsei_s"]),
                float(fit["feat_dev_td_a_Rw1_v"]),
                float(fit["feat_dev_td_tau_Rw1_s"]),
                float(fit["feat_dev_td_a_Rw2_v"]),
                float(fit["feat_dev_td_tau_Rw2_s"]),
            )

        valid_component_count = int(
            sum(
                int(np.isfinite(fit.get(col, np.nan)))
                for col in ["feat_dev_td_a_Rsei_v", "feat_dev_td_a_Rw1_v", "feat_dev_td_a_Rw2_v"]
            )
        )
        rmse = float(fit.get("feat_dev_td_fit_rmse_v", np.nan))
        effective_points = int(len(x))
        success = _is_success_status(fit.get("feat_dev_td_fit_status"))

        candidate: Dict[str, object] = {
            "source_file": path.name,
            "serial_norm": serial_norm,
            "group_tag": str(df["group_tag"].iloc[0]),
            "cycle_c": float(cycle_c),
            "segment_start_idx": start,
            "segment_end_idx": end,
            "relax_duration_s": float(seg["duration_s"]),
            "pre_current_a": float(seg["i_prev_a"]),
            "effective_points": effective_points,
            "fit_status": str(fit.get("feat_dev_td_fit_status")),
            "fit_rmse_v": rmse,
            "valid_component_count": valid_component_count,
            "prior_cycle_used": float(fit.get("feat_dev_td_prior_cycle_used", np.nan)),
            "td_v_inf_v": float(fit.get("feat_dev_td_v_inf_v", np.nan)),
            "td_Rsei_ohm": float(abs(fit["feat_dev_td_a_Rsei_v"]) / abs(seg["i_prev_a"])) if np.isfinite(fit.get("feat_dev_td_a_Rsei_v", np.nan)) and abs(float(seg["i_prev_a"])) > 1e-9 else float("nan"),
            "td_Rw1_ohm": float(abs(fit["feat_dev_td_a_Rw1_v"]) / abs(seg["i_prev_a"])) if np.isfinite(fit.get("feat_dev_td_a_Rw1_v", np.nan)) and abs(float(seg["i_prev_a"])) > 1e-9 else float("nan"),
            "td_Rw2_ohm": float(abs(fit["feat_dev_td_a_Rw2_v"]) / abs(seg["i_prev_a"])) if np.isfinite(fit.get("feat_dev_td_a_Rw2_v", np.nan)) and abs(float(seg["i_prev_a"])) > 1e-9 else float("nan"),
            "t_rel_s": x.tolist(),
            "voltage_actual_v": y.tolist(),
            "voltage_pred_v": yhat.tolist(),
            "v_before_v": float(seg["v_before_v"]),
            "v_start_v": float(seg["v_start_v"]),
        }

        rank = (
            int(success),
            valid_component_count,
            effective_points,
            float(seg["duration_s"]),
            -rmse if np.isfinite(rmse) else -1e9,
        )
        candidate["_rank"] = rank
        if best_local is None or rank > best_local["_rank"]:
            best_local = candidate
    return best_local


def _plot_overlay(candidate: Dict[str, object], out_png: Path) -> None:
    t = np.asarray(candidate["t_rel_s"], dtype=float)
    y = np.asarray(candidate["voltage_actual_v"], dtype=float)
    yhat = np.asarray(candidate["voltage_pred_v"], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(t, y, marker="o", ms=4, lw=1.8, label="Measured voltage")
    if np.isfinite(yhat).any():
        ax.plot(t, yhat, marker="s", ms=3, lw=1.6, label="TD-ECM fitted response")
    ax.scatter([0.0], [candidate["v_start_v"]], color="tab:red", s=40, label="Relax start")
    ax.set_xlabel("Relaxation time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(
        f"TD-ECM voltage overlay\n{candidate['source_file']} | cycle={candidate['cycle_c']} | status={candidate['fit_status']}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    txt = [
        f"points={candidate['effective_points']}",
        f"duration={candidate['relax_duration_s']:.1f}s",
        f"I_prev={candidate['pre_current_a']:.3f}A",
    ]
    if np.isfinite(candidate.get("fit_rmse_v", np.nan)):
        txt.append(f"RMSE={candidate['fit_rmse_v']:.5f}V")
    if np.isfinite(candidate.get("td_Rsei_ohm", np.nan)):
        txt.append(f"Rsei={candidate['td_Rsei_ohm']:.5f}Ω")
    if np.isfinite(candidate.get("td_Rw1_ohm", np.nan)):
        txt.append(f"Rw1={candidate['td_Rw1_ohm']:.5f}Ω")
    if np.isfinite(candidate.get("td_Rw2_ohm", np.nan)):
        txt.append(f"Rw2={candidate['td_Rw2_ohm']:.5f}Ω")
    ax.text(0.99, 0.01, "\n".join(txt), transform=ax.transAxes, ha="right", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot time-domain ECM fitted voltage response overlaid with measured relaxation voltage."
    )
    ap.add_argument("--raw_dir", default="", help="Directory containing raw files. Used to auto-select an example if --raw_path is not provided.")
    ap.add_argument("--raw_path", default="", help="Specific raw file to visualize.")
    ap.add_argument("--eis_prior_csv", default="", help="Optional EIS prior CSV used during constrained TD fitting.")
    ap.add_argument("--prior_mode", default="global", choices=["serial_cycle", "serial_only", "global", "none"])
    ap.add_argument("--prior_align_mode", default="last_le", choices=["last_le", "exact"])
    ap.add_argument("--cycle_mode", default="best", choices=["all", "first", "last", "best"])
    ap.add_argument("--segment_choice", default="best_td", choices=["longest", "first", "most_points", "best_td"])
    ap.add_argument("--source_file_match", default="", help="Optional substring to force-select a specific raw file when using --raw_dir.")
    ap.add_argument("--out_dir", required=True, help="Directory to save overlay plot and summary files.")
    args = ap.parse_args()

    prior_df = pd.read_csv(args.eis_prior_csv) if args.eis_prior_csv else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: List[Dict[str, object]] = []
    if args.raw_path:
        cand = _candidate_from_file(
            Path(args.raw_path),
            prior_df=prior_df,
            prior_align_mode=args.prior_align_mode,
            prior_mode=args.prior_mode,
            cycle_mode=args.cycle_mode,
            segment_choice=args.segment_choice,
        )
        if cand is not None:
            candidates.append(cand)
    else:
        raw_dir = Path(args.raw_dir)
        files = [p for p in sorted(raw_dir.rglob("*")) if p.is_file() and not p.name.startswith(".")]
        if args.source_file_match:
            files = [p for p in files if args.source_file_match in p.name]
        for p in files:
            cand = _candidate_from_file(
                p,
                prior_df=prior_df,
                prior_align_mode=args.prior_align_mode,
                prior_mode=args.prior_mode,
                cycle_mode=args.cycle_mode,
                segment_choice=args.segment_choice,
            )
            if cand is not None:
                candidates.append(cand)

    if not candidates:
        raise ValueError("No usable relaxation candidate found for overlay plotting.")

    rank_sorted = sorted(candidates, key=lambda x: x["_rank"], reverse=True)
    best = rank_sorted[0]

    summary = pd.DataFrame(
        [
            {k: v for k, v in c.items() if k not in {"t_rel_s", "voltage_actual_v", "voltage_pred_v", "_rank"}}
            for c in rank_sorted
        ]
    )
    summary_csv = out_dir / "td_ecm_overlay_candidates.csv"
    summary.to_csv(summary_csv, index=False)

    trace_csv = out_dir / "td_ecm_overlay_best_trace.csv"
    pd.DataFrame(
        {
            "t_rel_s": best["t_rel_s"],
            "voltage_actual_v": best["voltage_actual_v"],
            "voltage_pred_v": best["voltage_pred_v"],
        }
    ).to_csv(trace_csv, index=False)

    meta_json = out_dir / "td_ecm_overlay_best_metadata.json"
    with meta_json.open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in best.items() if k not in {"_rank"}}, f, ensure_ascii=False, indent=2)

    out_png = out_dir / "td_ecm_overlay_best_candidate.png"
    _plot_overlay(best, out_png)

    print(f"[INFO] Saved candidate summary: {summary_csv}")
    print(f"[INFO] Saved best trace CSV: {trace_csv}")
    print(f"[INFO] Saved best metadata JSON: {meta_json}")
    print(f"[INFO] Saved overlay PNG: {out_png}")
    print(
        "[INFO] Best candidate:",
        best["source_file"],
        "cycle=",
        best["cycle_c"],
        "status=",
        best["fit_status"],
        "points=",
        best["effective_points"],
    )


if __name__ == "__main__":
    main()
