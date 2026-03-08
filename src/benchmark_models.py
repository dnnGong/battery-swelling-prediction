#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Batch benchmark runner for classic/deep/transformer training scripts. "
            "Runs configured combinations and aggregates results."
        ),
        epilog=(
            "Example:\n"
            "  python src/benchmark_models.py \\\n"
            "    --table_csv ./data/ml/hycl_new/feature_table_hycl.csv \\\n"
            "    --out_dir ./data/ml/hycl_new/benchmark \\\n"
            "    --target_mode fixed_T --label_mode absolute --T 100 --max_input_cycle 50 \\\n"
            "    --runners classic,deep,transformer \\\n"
            "    --model_sets basic,extended \\\n"
            "    --feature_sets full,variance,discharge"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--table_csv", required=True, help="Input feature table CSV path.")
    ap.add_argument("--out_dir", required=True, help="Benchmark output root directory.")
    ap.add_argument("--target_mode", choices=["fixed_T", "future_delta_TK"], required=True)
    ap.add_argument("--label_mode", choices=["absolute", "delta"], required=True)
    ap.add_argument("--T", type=int, default=100, help="Target cycle for fixed_T.")
    ap.add_argument("--future_k", type=int, default=20, help="Future horizon for future_delta_TK.")
    ap.add_argument("--max_input_cycle", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--min_rows_per_group", type=int, default=6)
    ap.add_argument("--min_cells_per_group", type=int, default=4)
    ap.add_argument(
        "--runners",
        default="classic",
        help="Comma list from {classic,deep,transformer}.",
    )
    ap.add_argument("--groups", default="CL,FLC,HYCL", help="Comma list for deep/transformer scripts.")
    ap.add_argument("--model_sets", default="basic,extended", help="Comma list from {basic,extended,all}.")
    ap.add_argument("--feature_sets", default="full,variance,discharge", help="Comma list from {full,variance,discharge,ecm}.")
    ap.add_argument("--variance_top_n", type=int, default=16)
    ap.add_argument("--extra_models", default="", help="Optional comma model names passed to --models.")
    ap.add_argument("--deep_models", default="mlp,cnn,lstm", help="Comma list for train_swelling_deep.py.")
    ap.add_argument("--deep_epochs", type=int, default=120)
    ap.add_argument("--deep_lr", type=float, default=1e-3)
    ap.add_argument("--deep_batch_size", type=int, default=32)
    ap.add_argument("--deep_hidden_dim", type=int, default=64)
    ap.add_argument("--deep_dropout", type=float, default=0.1)
    ap.add_argument("--transformer_epochs", type=int, default=160)
    ap.add_argument("--transformer_lr", type=float, default=5e-4)
    ap.add_argument("--transformer_batch_size", type=int, default=32)
    ap.add_argument("--transformer_hidden_dim", type=int, default=64)
    ap.add_argument("--transformer_dropout", type=float, default=0.1)
    ap.add_argument("--transformer_n_heads", type=int, default=4)
    ap.add_argument("--transformer_n_layers", type=int, default=2)
    ap.add_argument("--transformer_ff_dim", type=int, default=128)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    runners = [x.strip().lower() for x in args.runners.split(",") if x.strip()]
    model_sets = [x.strip() for x in args.model_sets.split(",") if x.strip()]
    feature_sets = [x.strip() for x in args.feature_sets.split(",") if x.strip()]

    def run_one(run_tag: str, runner: str, script_rel: str, cmd: List[str]) -> Dict[str, object]:
        run_dir = out_root / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Running benchmark: runner={runner}, run_tag={run_tag}")
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        log_file = run_dir / "run.log"
        log_file.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        status = "ok" if proc.returncode == 0 else "failed"
        print(f"[INFO] status={status} run_tag={run_tag} log={log_file}")
        return {
            "runner": runner,
            "script": script_rel,
            "run_tag": run_tag,
            "status": status,
            "return_code": int(proc.returncode),
            "run_dir": str(run_dir),
            "log_file": str(log_file),
        }

    runs: List[dict] = []
    if "classic" in runners:
        for ms in model_sets:
            for fs in feature_sets:
                run_tag = f"classic__{ms}__{fs}"
                run_dir = out_root / run_tag
                cmd = [
                    "python",
                    "src/train_swelling_models.py",
                    "--table_csv",
                    args.table_csv,
                    "--out_dir",
                    str(run_dir),
                    "--target_mode",
                    args.target_mode,
                    "--label_mode",
                    args.label_mode,
                    "--T",
                    str(args.T),
                    "--future_k",
                    str(args.future_k),
                    "--max_input_cycle",
                    str(args.max_input_cycle),
                    "--seed",
                    str(args.seed),
                    "--test_size",
                    str(args.test_size),
                    "--min_rows_per_group",
                    str(args.min_rows_per_group),
                    "--min_cells_per_group",
                    str(args.min_cells_per_group),
                    "--model_set",
                    ms,
                    "--feature_set",
                    fs,
                    "--variance_top_n",
                    str(args.variance_top_n),
                    "--run_tag",
                    run_tag,
                ]
                if args.extra_models.strip():
                    cmd.extend(["--models", args.extra_models.strip()])
                rec = run_one(run_tag=run_tag, runner="classic", script_rel="src/train_swelling_models.py", cmd=cmd)
                rec["model_set"] = ms
                rec["feature_set"] = fs
                rec["deep_models"] = ""
                runs.append(rec)

    if "deep" in runners:
        for fs in feature_sets:
            run_tag = f"deep__{fs}"
            run_dir = out_root / run_tag
            cmd = [
                "python",
                "src/train_swelling_deep.py",
                "--table_csv",
                args.table_csv,
                "--out_dir",
                str(run_dir),
                "--target_mode",
                args.target_mode,
                "--label_mode",
                args.label_mode,
                "--T",
                str(args.T),
                "--future_k",
                str(args.future_k),
                "--max_input_cycle",
                str(args.max_input_cycle),
                "--seed",
                str(args.seed),
                "--test_size",
                str(args.test_size),
                "--min_rows_per_group",
                str(args.min_rows_per_group),
                "--min_cells_per_group",
                str(args.min_cells_per_group),
                "--groups",
                args.groups,
                "--models",
                args.deep_models,
                "--feature_set",
                fs,
                "--variance_top_n",
                str(args.variance_top_n),
                "--epochs",
                str(args.deep_epochs),
                "--lr",
                str(args.deep_lr),
                "--batch_size",
                str(args.deep_batch_size),
                "--hidden_dim",
                str(args.deep_hidden_dim),
                "--dropout",
                str(args.deep_dropout),
                "--run_tag",
                run_tag,
            ]
            rec = run_one(run_tag=run_tag, runner="deep", script_rel="src/train_swelling_deep.py", cmd=cmd)
            rec["model_set"] = "deep"
            rec["feature_set"] = fs
            rec["deep_models"] = args.deep_models
            runs.append(rec)

    if "transformer" in runners:
        for fs in feature_sets:
            run_tag = f"transformer__{fs}"
            run_dir = out_root / run_tag
            cmd = [
                "python",
                "src/train_swelling_transformer.py",
                "--table_csv",
                args.table_csv,
                "--out_dir",
                str(run_dir),
                "--target_mode",
                args.target_mode,
                "--label_mode",
                args.label_mode,
                "--T",
                str(args.T),
                "--future_k",
                str(args.future_k),
                "--max_input_cycle",
                str(args.max_input_cycle),
                "--seed",
                str(args.seed),
                "--test_size",
                str(args.test_size),
                "--min_rows_per_group",
                str(args.min_rows_per_group),
                "--min_cells_per_group",
                str(args.min_cells_per_group),
                "--groups",
                args.groups,
                "--feature_set",
                fs,
                "--variance_top_n",
                str(args.variance_top_n),
                "--epochs",
                str(args.transformer_epochs),
                "--lr",
                str(args.transformer_lr),
                "--batch_size",
                str(args.transformer_batch_size),
                "--hidden_dim",
                str(args.transformer_hidden_dim),
                "--dropout",
                str(args.transformer_dropout),
                "--n_heads",
                str(args.transformer_n_heads),
                "--n_layers",
                str(args.transformer_n_layers),
                "--ff_dim",
                str(args.transformer_ff_dim),
                "--run_tag",
                run_tag,
            ]
            rec = run_one(
                run_tag=run_tag,
                runner="transformer",
                script_rel="src/train_swelling_transformer.py",
                cmd=cmd,
            )
            rec["model_set"] = "transformer"
            rec["feature_set"] = fs
            rec["deep_models"] = "transformer"
            runs.append(rec)

    runs_df = pd.DataFrame(runs)
    runs_csv = out_root / "benchmark_runs.csv"
    runs_df.to_csv(runs_csv, index=False)
    print(f"[INFO] Saved run ledger: {runs_csv}")

    # Aggregate all successful results__*.csv
    all_rows = []
    for row in runs:
        if row["status"] != "ok":
            continue
        run_dir = Path(row["run_dir"])
        res_files = sorted(run_dir.glob("results__*.csv"))
        for rf in res_files:
            try:
                df = pd.read_csv(rf)
                df["benchmark_run_tag"] = row["run_tag"]
                df["benchmark_runner"] = row.get("runner", "")
                df["benchmark_script"] = row.get("script", "")
                df["benchmark_model_set"] = row["model_set"]
                df["benchmark_feature_set"] = row["feature_set"]
                df["benchmark_results_file"] = str(rf)
                all_rows.append(df)
            except Exception:
                continue

    if all_rows:
        bench = pd.concat(all_rows, ignore_index=True)
        out_csv = out_root / "benchmark_results_aggregate.csv"
        bench.to_csv(out_csv, index=False)
        print(f"[INFO] Saved benchmark aggregate: {out_csv}")
    else:
        print("[WARN] No successful results CSV found to aggregate.")


if __name__ == "__main__":
    main()
