#!/usr/bin/env python3
"""Compile FQTree RTL projects with Verilator and run bit-exact checks."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rtl_dirs", nargs="+", type=Path, help="RTL project directories.")
    parser.add_argument(
        "--data-cache",
        type=Path,
        default=Path("/tmp/jsc.npz"),
        help="NPZ file containing X_test and y_test arrays.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("verilator_summary.jsonl"),
        help="JSONL summary path.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for a quick check.")
    parser.add_argument("--nproc", type=int, default=1, help="Parallel jobs passed to Verilator make.")
    parser.add_argument(
        "--compile-attempts",
        type=int,
        default=6,
        help="Retry count for Verilator compilation.",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Load an existing Verilator shared library instead of compiling.",
    )
    parser.add_argument(
        "--keep-obj",
        action="store_true",
        help="Keep Verilator obj_dir after compilation.",
    )
    parser.add_argument(
        "--openmp",
        action="store_true",
        help="Build the generated binder with OpenMP support.",
    )
    parser.add_argument(
        "--cxx",
        default="x86_64-conda-linux-gnu-g++",
        help="C++ compiler used when linking the Verilator shared library.",
    )
    parser.add_argument(
        "--verilator-flags",
        default="",
        help="Extra flags passed through RTLModel._compile().",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next RTL project if one check fails.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)
def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def load_test_arrays(data_cache: Path, max_samples: int | None) -> tuple[Any, Any]:
    import numpy as np

    arrays = np.load(data_cache)
    X_test = arrays["X_test"]
    y_test = arrays["y_test"]
    if max_samples is not None:
        X_test = X_test[:max_samples]
        y_test = y_test[:max_samples]
    return X_test, y_test


def compile_rtl(args: argparse.Namespace, rtl: Any, rtl_dir: Path) -> tuple[int, float]:
    if args.skip_compile:
        start = time.time()
        rtl._load_lib()
        return 0, time.time() - start

    last_error: Exception | None = None
    start = time.time()
    for attempt in range(1, args.compile_attempts + 1):
        try:
            log(f"compile attempt {attempt} for {rtl_dir}")
            rtl._compile(
                openmp=args.openmp,
                nproc=args.nproc,
                clean=not args.keep_obj,
                _env={"VERILATOR_FLAGS": args.verilator_flags, "CXX": args.cxx},
            )
            return attempt, time.time() - start
        except Exception as exc:  # pragma: no cover - toolchain path
            last_error = exc
            log(f"compile attempt {attempt} failed for {rtl_dir}: {exc!r}")
    raise RuntimeError(f"Verilator compile failed for {rtl_dir}: {last_error!r}")


def process_one(args: argparse.Namespace, rtl_dir: Path, X_test: Any, y_test: Any) -> dict[str, Any]:
    import numpy as np
    from alkaid.codegen import RTLModel
    from alkaid.types import CombLogic

    rtl_dir = rtl_dir.resolve()
    metadata = json.loads((rtl_dir / "metadata.json").read_text())
    comb = CombLogic.load(rtl_dir / "model/comb.json.gz")
    rtl = RTLModel(comb, rtl_dir, "model")

    attempts, compile_seconds = compile_rtl(args, rtl, rtl_dir)

    start = time.time()
    expected = comb.predict(X_test, n_threads=1)
    comb_seconds = time.time() - start

    start = time.time()
    actual = rtl.predict(X_test, n_threads=1)
    rtl_seconds = time.time() - start

    bit_exact = bool(np.array_equal(actual, expected))
    max_abs_diff = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    hw_acc = float(np.mean(np.argmax(actual, axis=1) == y_test))

    record = {
        "label": metadata.get("label", rtl_dir.name),
        "path": str(rtl_dir),
        "compile_attempts": attempts,
        "compile_seconds": compile_seconds,
        "comb_predict_seconds": comb_seconds,
        "rtl_predict_seconds": rtl_seconds,
        "bit_exact": bit_exact,
        "max_abs_diff": max_abs_diff,
        "hw_acc_from_rtl": hw_acc,
        "n_samples": int(X_test.shape[0]),
        "output_shape": list(actual.shape),
        "metadata": metadata,
    }
    if not bit_exact:
        mismatch = np.argwhere(actual != expected)
        record["first_mismatch"] = mismatch[0].tolist() if mismatch.size else None
        raise AssertionError(f"RTL output differs from combinational model for {rtl_dir}")
    return record


def main() -> None:
    args = parse_args()
    X_test, y_test = load_test_arrays(args.data_cache, args.max_samples)
    log(f"loaded test data: X_test={X_test.shape}, y_test={y_test.shape}")

    for rtl_dir in args.rtl_dirs:
        try:
            record = process_one(args, rtl_dir, X_test, y_test)
            write_jsonl(args.summary, record)
            log(
                "checked {label}: bit_exact={exact}, acc={acc:.6f}, compile={compile_s:.1f}s".format(
                    label=record["label"],
                    exact=record["bit_exact"],
                    acc=record["hw_acc_from_rtl"],
                    compile_s=record["compile_seconds"],
                )
            )
        except Exception as exc:
            log(f"failed {rtl_dir}: {exc!r}")
            traceback.print_exc()
            write_jsonl(
                args.summary,
                {"path": str(rtl_dir), "error": repr(exc), "traceback": traceback.format_exc()},
            )
            if not args.continue_on_error:
                raise


if __name__ == "__main__":
    main()
