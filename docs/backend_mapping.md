# FQTree, qxgb, and Hardware Backend Mapping

This repository is the paper-facing FQTree package. The implementation is
intentionally split from the lower-level `qxgb` project:

- FQTree owns the public API, paper experiment configurations, and
  reproducibility workflow.
- qxgb is the quantized XGBoost backend used for QAT training and integer
  booster export.
- Alkaid lowers the integer booster into combinational logic and RTL projects.
- Vivado/Vitis and Verilator are external hardware tools used to report FPGA
  resources and check RTL behavior.

## Responsibility Boundary

| Layer | Package or tool | Responsibility |
| --- | --- | --- |
| Paper API | `fqtree.FQTreeClassifier` | Stable user-facing class named after the algorithm in the paper. |
| Configuration | `fqtree.FQTreeConfig` | Stores FQTree-owned settings such as `scale`, `bias`, QAT/PTQ mode, and hardware trace mode. |
| Backend adapter | `fqtree.backends.QXGBBackend` | Translates FQTree configuration into `qxgb.QXGBClassifier` arguments. |
| QAT engine | `qxgb` | Performs quantization-aware XGBoost training and exports the integer booster via `ibooster()`. |
| Hardware trace | `alkaid.converter.trace_model` and `alkaid.trace.trace` | Converts the qxgb integer booster into Alkaid combinational logic. |
| RTL generation | `alkaid.codegen.RTLModel` | Writes Verilog/VHDL RTL projects from Alkaid logic. |
| FPGA implementation | Vivado/Vitis | Produces post-route timing, power, and resource reports. |
| RTL simulation | Verilator | Compiles generated RTL into a shared library used by `RTLModel.predict()`. |

## Algorithm-to-Code Mapping

The FQTree paper describes fine-grained quantization-aware boosted decision
training. In this repository, the paper algorithm maps to code as follows:

1. `FQTreeClassifier(...)` receives paper-level hyperparameters, such as
   `scale`, `bias`, `n_estimators`, `max_depth`, `num_class`, and `eta`.
2. `FQTreeConfig` separates FQTree settings from generic XGBoost/qxgb training
   parameters.
3. `QXGBBackend` constructs `qxgb.QXGBClassifier` and delegates `fit`,
   `predict`, and `predict_proba`.
4. During `fit`, qxgb performs the QAT loop and quantized leaf-value handling.
5. `FQTreeClassifier.ibooster()` delegates to qxgb and returns the integer
   booster used for hardware lowering.
6. `trace_model(..., mode="mux")` lowers the integer booster to Alkaid logic.
7. `RTLModel.write(...)` writes the RTL project used for the hardware flow.
8. `RTLModel._compile()` invokes Verilator; `RTLModel.predict()` then runs the
   compiled RTL model for bit-exact simulation.

## Why Keep FQTree and qxgb Separate?

The split is useful only if FQTree remains more than a rename of qxgb. The
intended boundary is:

- qxgb remains a reusable quantized-BDT engine.
- FQTree remains the reproducible, paper-facing package with the algorithm name,
  the Table I configurations, hardware scripts, and documentation.

This lets users import the paper API:

```python
from fqtree import FQTreeClassifier
```

while still keeping the backend implementation reusable and isolated:

```python
model = FQTreeClassifier(scale=3, bias=-2.5, n_estimators=24, max_depth=4)
backend_model = model.to_qxgb()
```

## Current Limitations

- The qxgb backend currently implements the QAT path used by the FQTree rows.
  PTQ is not available through this backend yet.
- Vivado/Vitis remain external toolchain dependencies and are not installed by
  the conda environment.
