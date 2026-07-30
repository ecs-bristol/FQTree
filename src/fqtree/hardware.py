from typing import Any


def to_integer_booster(model: Any) -> Any:
    """Return the integer booster used by FQTree hardware generation."""
    if hasattr(model, "to_integer_booster"):
        return model.to_integer_booster()
    if not hasattr(model, "ibooster"):
        raise TypeError(
            "Expected a fitted FQTreeClassifier or qxgb-compatible model with ibooster()."
        )
    return model.ibooster()


def trace_fqtree_model(model_or_booster: Any, *, inputs: Any, mode: str = "mux", **kwargs: Any) -> Any:
    """Trace a fitted FQTree model or integer booster with Alkaid."""
    from alkaid.converter import trace_model

    booster = to_integer_booster(model_or_booster) if hasattr(model_or_booster, "ibooster") else model_or_booster
    return trace_model(booster, inputs=inputs, mode=mode, **kwargs)
