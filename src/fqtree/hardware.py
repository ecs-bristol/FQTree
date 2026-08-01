from alkaid.trace import FVArray
from qxgb import QXGBClassifier
from xgboost import Booster

from .classifier import FQTreeClassifier


def to_integer_booster(model: FQTreeClassifier | QXGBClassifier) -> object:
    """Return the integer booster used by FQTree hardware generation."""
    if isinstance(model, (FQTreeClassifier, QXGBClassifier)):
        return model.ibooster()

    raise TypeError(
        "Expected a fitted FQTreeClassifier or qxgb.QXGBClassifier with ibooster()."
    )


def trace_fqtree_model(
    model_or_booster: FQTreeClassifier | QXGBClassifier | Booster,
    *,
    inputs: FVArray,
    mode: str = 'mux',
    **kwargs: object,
) -> object:
    """Trace a fitted FQTree model or integer booster with Alkaid."""
    from alkaid.converter import trace_model

    if isinstance(model_or_booster, (FQTreeClassifier, QXGBClassifier)):
        booster = to_integer_booster(model_or_booster)
    else:
        booster = model_or_booster

    return trace_model(booster, inputs=inputs, mode=mode, **kwargs)
