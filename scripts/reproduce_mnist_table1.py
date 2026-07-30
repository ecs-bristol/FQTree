#!/usr/bin/env python3
"""Reproduce the MNIST QAT/FQTree rows from Table I.

This script trains the paper-facing FQTree configurations for the MNIST
benchmark, traces them through Alkaid, and writes RTL projects. Vivado and
Verilator are handled by the separate reporting and simulation scripts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


MNIST_CASES: dict[str, dict[str, Any]] = {
    "accuracy": {
        "n_estimators": 64,
        "max_depth": 6,
        "scale": 2.5,
        "bias": -1,
        "n_stages": 2,
        "clock_period": 1.6,
    },
    "balanced": {
        "n_estimators": 48,
        "max_depth": 4,
        "scale": 1.5,
        "bias": -1,
        "n_stages": 2,
        "clock_period": 1.6,
    },
    "low_cost": {
        "n_estimators": 32,
        "max_depth": 4,
        "scale": 1.5,
        "bias": -1,
        "n_stages": 2,
        "clock_period": 1.6,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=["all", *MNIST_CASES.keys()],
        default="all",
        help="MNIST Table I configuration to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summaries and RTL output. Defaults to runs/mnist_table1_qat_<timestamp>.",
    )
    parser.add_argument(
        "--data-cache",
        type=Path,
        default=Path("/tmp/mnist.npz"),
        help="Cached MNIST train/test arrays.",
    )
    parser.add_argument(
        "--no-fetch-openml",
        action="store_true",
        help="Require --data-cache to exist instead of fetching the OpenML dataset.",
    )
    parser.add_argument(
        "--part-name",
        default="xcvu9p-flgb2104-2-i",
        help="FPGA part name to write into generated Vivado projects.",
    )
    parser.add_argument(
        "--xls-opt",
        dest="xls_opt",
        action="store_true",
        default=True,
        help="Use the optimized RTL generation path.",
    )
    parser.add_argument(
        "--no-xls-opt",
        dest="xls_opt",
        action="store_false",
        help="Use the fallback Verilog generation path.",
    )
    parser.add_argument(
        "--hardware-mode",
        choices=["mux", "masking"],
        default="mux",
        help="Alkaid tree tracing mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected configurations and exit without importing heavy dependencies.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def selected_cases(case: str) -> list[tuple[str, dict[str, Any]]]:
    if case == "all":
        return list(MNIST_CASES.items())
    return [(case, MNIST_CASES[case])]


def default_output_dir() -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"mnist_table1_qat_{ts}"


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def prepare_mnist_data(data_cache: Path, *, fetch_openml: bool) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    if not data_cache.exists():
        if not fetch_openml:
            raise FileNotFoundError(f"MNIST cache not found: {data_cache}")

        from sklearn.datasets import fetch_openml as fetch_openml_dataset

        log("fetching OpenML dataset mnist_784")
        data = fetch_openml_dataset("mnist_784", version=1, as_frame=False)
        X, y = data.data, data.target.astype(np.int64)
        X_train, y_train = X[:60000], y[:60000]
        X_test, y_test = X[60000:], y[60000:]
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        X_train_q = np.floor(X_train / 2**7)
        X_test_q = np.floor(X_test / 2**7)
        data_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            data_cache,
            X_train=X_train_q,
            y_train=y_train,
            X_test=X_test_q,
            y_test=y_test,
        )
        log(f"cached MNIST arrays to {data_cache}")

    arrays = np.load(data_cache)
    return arrays["X_train"], arrays["y_train"], arrays["X_test"], arrays["y_test"]


def require_xls_python() -> None:
    try:
        from xls.raw import jit_fn_predict  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-specific path
        raise RuntimeError(
            "xls_opt=True requires the xls-python package in the active environment. "
            "Install it or pass --no-xls-opt."
        ) from exc


def run_case(
    label: str,
    config: dict[str, Any],
    *,
    output_dir: Path,
    data: tuple[Any, Any, Any, Any],
    part_name: str,
    xls_opt: bool,
    hardware_mode: str,
    summary_path: Path,
) -> None:
    import numpy as np
    from alkaid.codegen import RTLModel
    from alkaid.converter import trace_model
    from alkaid.trace import FVArray, trace
    from fqtree import FQTreeClassifier

    X_train, y_train, X_test, y_test = data
    log(f"start {label}: {config}, xls_opt={xls_opt}")

    model = FQTreeClassifier(
        scale=config["scale"],
        bias=config["bias"],
        num_class=10,
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        eta=0.8,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_test_acc = float(np.mean(model.predict(X_test) == y_test))
    log(f"{label}: XGB/FQTree test accuracy {xgb_test_acc:.6f}")

    inp = FVArray.new(int(X_train.shape[1])).quantize(0, 1, 0).as_new()
    _, out = trace_model(model.ibooster(), inputs=inp, mode=hardware_mode)
    comb = trace(inp, out)

    hw_train_acc = float(np.mean(np.argmax(comb.predict(X_train), axis=1) == y_train))
    hw_test_acc = float(np.mean(np.argmax(comb.predict(X_test, n_threads=1), axis=1) == y_test))
    log(f"{label}: traced hardware accuracy train={hw_train_acc:.6f}, test={hw_test_acc:.6f}")

    rtl_path = output_dir / "rtl" / (
        f"{label}_n_estimators={config['n_estimators']}_max_depth={config['max_depth']}"
        f"_scale={config['scale']}_bias={config['bias']}"
    )
    rtl = RTLModel(
        comb,
        rtl_path,
        "model",
        n_stages=config["n_stages"],
        clock_period=config["clock_period"],
        clock_uncertainty=0,
        part_name=part_name,
    )

    start = time.time()
    rtl.write(
        xls_opt=xls_opt,
        metadata={
            "label": label,
            "dataset": "MNIST",
            "prediction_mode": "multiclass",
            "xgb_test_acc": xgb_test_acc,
            "comb_metric": hw_test_acc,
            "hw_train_acc": hw_train_acc,
            "source": "FQTree MNIST Table I QAT reproduction",
        },
    )
    rtl_write_seconds = time.time() - start
    log(f"{label}: wrote RTL to {rtl_path} in {rtl_write_seconds:.1f}s")

    write_jsonl(
        summary_path,
        {
            "label": label,
            "config": {**config, "part_name": part_name, "xls_opt": xls_opt},
            "xgb_test_acc": xgb_test_acc,
            "hw_train_acc": hw_train_acc,
            "hw_test_acc": hw_test_acc,
            "rtl_path": str(rtl_path),
            "rtl_write_seconds": rtl_write_seconds,
        },
    )


def main() -> None:
    args = parse_args()
    cases = selected_cases(args.case)

    if args.dry_run:
        print(json.dumps({label: config for label, config in cases}, indent=2, sort_keys=True))
        return

    if args.xls_opt:
        require_xls_python()

    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "reproduce_summary.jsonl"
    log(f"output_dir={output_dir}")
    log(f"part_name={args.part_name}")

    data = prepare_mnist_data(args.data_cache, fetch_openml=not args.no_fetch_openml)
    for label, config in cases:
        run_case(
            label,
            config,
            output_dir=output_dir,
            data=data,
            part_name=args.part_name,
            xls_opt=args.xls_opt,
            hardware_mode=args.hardware_mode,
            summary_path=summary_path,
        )


if __name__ == "__main__":
    main()
