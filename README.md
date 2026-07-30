# FQTree

FQTree is a paper-facing package for fine-grained quantization-aware boosted
decision trees targeting FPGA deployment.

The implementation currently uses [`qxgb`](https://github.com/calad0i/qxgb)
behind a backend adapter for quantization-aware XGBoost training and
integer-booster export. User code should import `fqtree`; the `qxgb` dependency
is treated as an implementation backend.

## Installation

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

## Repository Layout

- `src/fqtree/`: FQTree public Python package.
- `src/fqtree/backends/`: backend adapters. `qxgb` is imported only here.
- `examples/`: JSC, MNIST, and NID notebooks matching the paper experiments.
- `requirements.txt`: editable install helper for local notebook use.

## Backend Boundary

`FQTreeClassifier` owns the public API and delegates training, prediction, and
integer-booster export to a backend object. The default backend is `QXGBBackend`,
which translates `FQTreeConfig` into `qxgb.QXGBClassifier` parameters.

This keeps the paper-facing API stable:

```python
from fqtree import FQTreeClassifier, FQTreeConfig

config = FQTreeConfig(
    scale=3,
    bias=-2.5,
    xgb_params={
        "n_estimators": 24,
        "max_depth": 4,
        "num_class": 5,
        "eta": 0.8,
    },
)

model = FQTreeClassifier(config)
```

Direct keyword construction is also supported:

```python
model = FQTreeClassifier(scale=3, bias=-2.5, n_estimators=24, max_depth=4)
```

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
