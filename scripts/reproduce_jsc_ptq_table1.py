#!/usr/bin/env python3
"""Reproduce the JSC HLF PTQ/FQTree rows from Table I.

This script trains the paper-facing FQTree PTQ configurations for the JSC HLF
benchmark, traces them through Alkaid leaf quantization, and writes RTL
projects. Vivado and Verilator are handled by the separate reporting and
simulation scripts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable


JSC_PTQ_CASES: dict[str, dict[str, Any]] = {
    "accuracy": {
        "n_estimators": 13,
        "max_depth": 5,
        "leaf_quantizer": "leaf_quantizer_1",
        "n_stages": 2,
        "clock_period": 2,
        "precision": 3,
    },
    "low_cost": {
        "n_estimators": 4,
        "max_depth": 5,
        "leaf_quantizer": "leaf_quantizer_2",
        "n_stages": 1,
        "clock_period": 2,
        "precision": 4,
    },
}

LABELS = {
    "g": 0,
    "q": 1,
    "w": 2,
    "z": 3,
    "t": 4,
}


def leaf_quantizer_1(x: Any) -> tuple[Any, int]:
    import numpy as np

    values = np.round(x * 7)
    offset = np.sort(values.ravel())[1] + 1
    return np.maximum(values - offset, 0), -offset


def leaf_quantizer_2(x: Any) -> tuple[Any, int]:
    import numpy as np

    values = np.floor(x * 3)
    offset = np.sort(values.ravel())[1]
    return np.maximum(values - offset, 0), -offset


LEAF_QUANTIZERS: dict[str, Callable[[Any], tuple[Any, int]]] = {
    "leaf_quantizer_1": leaf_quantizer_1,
    "leaf_quantizer_2": leaf_quantizer_2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=["all", *JSC_PTQ_CASES.keys()],
        default="all",
        help="JSC PTQ Table I configuration to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summaries and RTL output. Defaults to runs/jsc_table1_ptq_<timestamp>.",
    )
    parser.add_argument(
        "--data-cache",
        type=Path,
        default=Path("/tmp/jsc.npz"),
        help="Cached JSC HLF train/test arrays.",
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
        return list(JSC_PTQ_CASES.items())
    return [(case, JSC_PTQ_CASES[case])]


def default_output_dir() -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"jsc_table1_ptq_{ts}"


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def prepare_jsc_data(data_cache: Path, *, fetch_openml: bool) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    if not data_cache.exists():
        if not fetch_openml:
            raise FileNotFoundError(f"JSC cache not found: {data_cache}")

        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split

        log("fetching OpenML dataset hls4ml_lhc_jets_hlf")
        data = fetch_openml("hls4ml_lhc_jets_hlf")
        X, y = np.array(data["data"]), data["target"]
        y = np.array([LABELS[label] for label in y])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=3,
        )
        x_min, x_max = X_train.min(axis=0), X_train.max(axis=0)
        X_train = np.floor((X_train - x_min) / (x_max - x_min) * 255)
        X_test = np.floor((X_test - x_min) / (x_max - x_min) * 255)
        data_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(data_cache, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        log(f"cached JSC arrays to {data_cache}")

    arrays = np.load(data_cache)
    return arrays["X_train"], arrays["y_train"], arrays["X_test"], arrays["y_test"]


def run_case(
    label: str,
    config: dict[str, Any],
    *,
    output_dir: Path,
    data: tuple[Any, Any, Any, Any],
    part_name: str,
    hardware_mode: str,
    summary_path: Path,
) -> None:
    import numpy as np
    from alkaid.codegen import RTLModel
    from alkaid.trace import FVArray, trace
    from fqtree import FQTreeClassifier

    X_train, y_train, X_test, y_test = data
    quantizer_name = config["leaf_quantizer"]
    leaf_quantizer = LEAF_QUANTIZERS[quantizer_name]
    log(f"start {label}: {config}")

    model = FQTreeClassifier(
        training_mode="ptq",
        leaf_quantizer=leaf_quantizer,
        num_class=5,
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        eta=0.8,
    )
    model.fit(X_train, y_train)
    sw_train_acc = float(np.mean(model.predict(X_train) == y_train))
    sw_test_acc = float(np.mean(model.predict(X_test) == y_test))
    log(f"{label}: XGBoost/PTQ software accuracy train={sw_train_acc:.6f}, test={sw_test_acc:.6f}")

    inp = FVArray.new(16).quantize(0, 8, 0).as_new()
    _, out = model.trace(inputs=inp, mode=hardware_mode)
    comb = trace(inp, out)

    hw_train_acc = float(np.mean(np.argmax(comb.predict(X_train), axis=1) == y_train))
    hw_test_acc = float(np.mean(np.argmax(comb.predict(X_test, n_threads=1), axis=1) == y_test))
    log(f"{label}: traced hardware accuracy train={hw_train_acc:.6f}, test={hw_test_acc:.6f}")

    rtl_path = output_dir / "rtl" / (
        f"{label}_n_estimators={config['n_estimators']}_max_depth={config['max_depth']}"
        f"_{quantizer_name}"
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
        xls_opt=False,
        metadata={
            "label": label,
            "dataset": "JSC HLF",
            "training_mode": "ptq",
            "prediction_mode": "multiclass",
            "leaf_quantizer": quantizer_name,
            "xgb_train_acc": sw_train_acc,
            "xgb_test_acc": sw_test_acc,
            "comb_metric": hw_test_acc,
            "hw_train_acc": hw_train_acc,
            "source": "FQTree JSC Table I PTQ reproduction",
        },
    )
    rtl_write_seconds = time.time() - start
    log(f"{label}: wrote RTL to {rtl_path} in {rtl_write_seconds:.1f}s")

    write_jsonl(
        summary_path,
        {
            "label": label,
            "config": {**config, "part_name": part_name},
            "xgb_train_acc": sw_train_acc,
            "xgb_test_acc": sw_test_acc,
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

    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "reproduce_summary.jsonl"
    log(f"output_dir={output_dir}")
    log(f"part_name={args.part_name}")

    data = prepare_jsc_data(args.data_cache, fetch_openml=not args.no_fetch_openml)
    for label, config in cases:
        run_case(
            label,
            config,
            output_dir=output_dir,
            data=data,
            part_name=args.part_name,
            hardware_mode=args.hardware_mode,
            summary_path=summary_path,
        )


if __name__ == "__main__":
    main()
