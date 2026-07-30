# FQTree

FQTree is the paper-facing package for fine-grained quantization-aware boosted
decision trees targeting FPGA deployment.

The public API is named after the paper algorithm, while the current QAT engine
is implemented through the pinned [`qxgb`](https://github.com/calad0i/qxgb)
backend.

See [docs/backend_mapping.md](docs/backend_mapping.md) for the detailed mapping
between FQTree, qxgb, Alkaid, Vivado, and Verilator.

## Installation

Recommended conda setup:

```bash
conda env create -f environment.yml
conda activate fqtree
```

This installs the local `fqtree` package in editable mode, notebook utilities,
`verilator` for RTL simulation, and explicit Python dependencies including the
pinned qxgb backend:

```text
qxgb==0.1.0
```

Vivado/Vitis are still external FPGA toolchain dependencies. Source
the local Xilinx setup before running Vivado synthesis, such as:

```bash
source ~/tools/xilinx_vitis.sh
```

If you already have a suitable Python environment, install the package directly:

```bash
pip install -e .[notebooks]
```

## Quick Start

```python
from fqtree import FQTreeClassifier

model = FQTreeClassifier(
    scale=3,
    bias=-2.5,
    n_estimators=24,
    max_depth=4,
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
hw_bst = model.ibooster()
```

For hardware tracing through Alkaid:

```python
from alkaid.trace import FVArray, trace
from fqtree import trace_fqtree_model

inp = FVArray.new(16).quantize(0, 8, 0).as_new()
_, out = trace_fqtree_model(model, inputs=inp, mode="mux")
comb = trace(inp, out)
```

## Reproducing Table I Benchmarks

The QAT/FQTree rows from Table I are represented by dataset-specific scripts:

- `scripts/reproduce_jsc_table1.py`: JSC HLF, with `accuracy` and `low_cost` cases.
- `scripts/reproduce_mnist_table1.py`: MNIST, with `accuracy`, `balanced`, and `low_cost` cases.
- `scripts/reproduce_nid_table1.py`: NID, with `accuracy`, `balanced`, and `low_cost` cases.

Inspect the selected configurations without running training:

```bash
python scripts/reproduce_jsc_table1.py --dry-run
python scripts/reproduce_mnist_table1.py --dry-run
python scripts/reproduce_nid_table1.py --dry-run
```

Generate RTL for the Table I QAT configurations:

```bash
python scripts/reproduce_jsc_table1.py --output-dir runs/jsc_table1_qat
python scripts/reproduce_mnist_table1.py --output-dir runs/mnist_table1_qat
python scripts/reproduce_nid_table1.py --output-dir runs/nid_table1_qat
```

Run or parse Vivado reports for generated RTL directories:

```bash
python scripts/run_vivado_reports.py runs/jsc_table1_qat/rtl/*
python scripts/run_vivado_reports.py runs/mnist_table1_qat/rtl/*
python scripts/run_vivado_reports.py runs/nid_table1_qat/rtl/*
```

If Vivado has already been run and only report parsing is needed:

```bash
python scripts/run_vivado_reports.py --parse-only runs/jsc_table1_qat/rtl/*
python scripts/run_vivado_reports.py --parse-only runs/mnist_table1_qat/rtl/*
python scripts/run_vivado_reports.py --parse-only runs/nid_table1_qat/rtl/*
```

Run Verilator bit-exact RTL checks with the matching dataset cache:

```bash
python scripts/run_verilator_check.py runs/jsc_table1_qat/rtl/* --data-cache /tmp/jsc.npz
python scripts/run_verilator_check.py runs/mnist_table1_qat/rtl/* --data-cache /tmp/mnist.npz
python scripts/run_verilator_check.py runs/nid_table1_qat/rtl/* --data-cache /tmp/nid.npz
```

`RTLModel.predict()` is the runtime simulation API, but only after
`RTLModel._compile()` or `RTLModel.compile()` has built the Verilator shared
library. The script handles that compile step before calling `predict()`.

## Repository Layout

- `src/fqtree/`: FQTree public Python package.
- `src/fqtree/backends/`: backend adapters, currently `QXGBBackend`.
- `examples/`: JSC, MNIST, and NID notebooks matching paper experiments.
- `scripts/`: reproducibility scripts for Table I RTL generation, Vivado reports, and Verilator checks.
- `docs/backend_mapping.md`: mapping between FQTree, qxgb, Alkaid, and hardware tools.
- `environment.yml`: conda environment for local development and RTL simulation.
- `requirements.txt`: editable install helper for local notebook use.

## Compatibility

`FQTreeClassifier` delegates to the default `qxgb` backend, so existing training
arguments such as `scale`, `bias`, `n_estimators`, and `max_depth` remain
available. The hardware integer booster is exposed through both
`to_integer_booster()` and the compatibility method `ibooster()`.

A compatibility alias is also provided:

```python
from fqtree import QXGBClassifier
```

New code should prefer:

```python
from fqtree import FQTreeClassifier
```
