#!/usr/bin/env python3
"""Run or parse Vivado post-route reports for FQTree RTL projects."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rtl_dirs", nargs="+", type=Path, help="RTL project directories.")
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Only parse existing reports; do not launch Vivado.",
    )
    parser.add_argument(
        "--vivado-env",
        default="~/tools/xilinx_vitis.sh",
        help="Shell script that places Vivado on PATH.",
    )
    parser.add_argument(
        "--vivado-bin",
        default="vivado",
        help="Vivado executable name or path after sourcing --vivado-env.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("vivado_summary.jsonl"),
        help="JSONL summary path.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next RTL project if Vivado or report parsing fails.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)
def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def run_vivado(rtl_dir: Path, *, vivado_env: str, vivado_bin: str) -> tuple[int, Path, float]:
    log_path = rtl_dir / "vivado.log"
    command = (
        f"source {vivado_env} && "
        f"cd {rtl_dir} && "
        f"{vivado_bin} -mode batch -source build_vivado_prj.tcl"
    )
    start = time.time()
    with log_path.open("w") as log_file:
        proc = subprocess.run(
            ["bash", "-lc", command],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return proc.returncode, log_path, time.time() - start


def load_report(rtl_dir: Path) -> dict[str, Any] | None:
    from alkaid._cli.report import load_project

    return load_project(rtl_dir)


def process_one(args: argparse.Namespace, rtl_dir: Path) -> dict[str, Any]:
    rtl_dir = rtl_dir.resolve()
    record: dict[str, Any] = {"path": str(rtl_dir), "label": rtl_dir.name}

    if not args.parse_only:
        log(f"running Vivado for {rtl_dir}")
        exit_code, log_path, seconds = run_vivado(
            rtl_dir,
            vivado_env=args.vivado_env,
            vivado_bin=args.vivado_bin,
        )
        record.update(
            {
                "vivado_exit_code": exit_code,
                "vivado_log": str(log_path),
                "vivado_seconds": seconds,
            }
        )
        if exit_code != 0:
            raise RuntimeError(f"Vivado failed for {rtl_dir}; see {log_path}")

    report = load_report(rtl_dir)
    record["report"] = report
    if report:
        record.update(
            {
                "hw_acc": report.get("comb_metric"),
                "latency_cycles": report.get("latency"),
                "latency_ns": report.get("latency(ns)"),
                "LUT": report.get("LUT"),
                "DSP": report.get("DSP"),
                "FF": report.get("FF"),
                "Fmax_MHz": report.get("Fmax(MHz)"),
                "WNS_ns": report.get("WNS(ns)"),
                "part_name": report.get("part_name"),
            }
        )
    return record


def main() -> None:
    args = parse_args()
    for rtl_dir in args.rtl_dirs:
        try:
            record = process_one(args, rtl_dir)
            write_jsonl(args.summary, record)
            log(
                "parsed {label}: acc={acc}, LUT={lut}, FF={ff}, DSP={dsp}, Fmax={fmax}".format(
                    label=record.get("label"),
                    acc=record.get("hw_acc"),
                    lut=record.get("LUT"),
                    ff=record.get("FF"),
                    dsp=record.get("DSP"),
                    fmax=record.get("Fmax_MHz"),
                )
            )
        except Exception as exc:
            error = {"path": str(rtl_dir), "error": repr(exc)}
            write_jsonl(args.summary, error)
            log(f"failed {rtl_dir}: {exc!r}")
            if not args.continue_on_error:
                raise


if __name__ == "__main__":
    main()
