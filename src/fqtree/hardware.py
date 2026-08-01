from alkaid.trace import FVArray
from qxgb import QXGBClassifier
from xgboost import Booster

from .classifier import FQTreeClassifier


def trace_fqtree_model(
    model_or_booster: FQTreeClassifier | Booster | FQTreeClassifier,
    *,
    inputs: FVArray,
    mode: str = 'mux',
    **kwargs,
):
    """Trace a fitted FQTree model or integer booster with Alkaid."""
    from alkaid.converter import trace_model

    if isinstance(model_or_booster, (FQTreeClassifier, QXGBClassifier)):
        booster = model_or_booster.ibooster()
    else:
        booster = model_or_booster

    return trace_model(booster, inputs=inputs, mode=mode, **kwargs)
