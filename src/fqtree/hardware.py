from typing import Protocol, runtime_checkable

from .classifier import FQTreeClassifier


@runtime_checkable
class SupportsIntegerBooster(Protocol):
    def ibooster(self) -> object:
        """Return an Alkaid-compatible integer booster."""
        ...


def to_integer_booster(model: object) -> object:
    """Return the integer booster used by FQTree hardware generation."""
    if isinstance(model, FQTreeClassifier):
        return model.to_integer_booster()

    if isinstance(model, SupportsIntegerBooster):
        return model.ibooster()

    raise TypeError(
        "Expected a fitted FQTreeClassifier or qxgb-compatible model with ibooster()."
    )


def trace_fqtree_model(
    model_or_booster: object,
    *,
    inputs: object,
    mode: str = "mux",
    **kwargs: object,
) -> object:
    """Trace a fitted FQTree model or integer booster with Alkaid."""
    from alkaid.converter import trace_model

    if isinstance(model_or_booster, (FQTreeClassifier, SupportsIntegerBooster)):
        booster = to_integer_booster(model_or_booster)
    else:
        booster = model_or_booster

    return trace_model(booster, inputs=inputs, mode=mode, **kwargs)
