#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def normalize_value(key: str, value: Any) -> str:
    if isinstance(value, list):
        if key in {"models", "custom_features"}:
            return ",".join(str(x) for x in value)
        return ",".join(str(x) for x in value)
    return str(value)


def build_command(config: Dict[str, Any], extra_args: List[str]) -> List[str]:
    script_path = Path(__file__).with_name("train_swelling_models.py")
    cmd: List[str] = [sys.executable, str(script_path)]

    for key, value in config.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        cmd.extend([flag, normalize_value(key, value)])

    cmd.extend(extra_args)
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run train_swelling_models.py from a JSON config file."
    )
    ap.add_argument("--config", required=True, help="Path to a JSON config file.")
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )
    args, extra = ap.parse_known_args()

    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a JSON object at the top level.")

    cmd = build_command(cfg, extra)
    print("[INFO] Config:", config_path)
    print("[INFO] Command:")
    print(" ".join(subprocess.list2cmdline([part]) for part in cmd))

    if args.dry_run:
        return

    completed = subprocess.run(cmd, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
