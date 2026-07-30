from typing import Any, Optional, Union

from .backends import QXGBBackend
from .config import FQTreeConfig


class FQTreeClassifier:
    """Public FQTree classifier API backed by a pluggable implementation backend."""

    backend_name = "qxgb"

    def __init__(
        self,
        config: Optional[FQTreeConfig] = None,
        backend: Union[str, Any] = "qxgb",
        **kwargs: Any,
    ):
        self.config = FQTreeConfig.from_kwargs(config, **kwargs)
        self.backend = self._make_backend(backend)

    @property
    def backend_model_(self) -> Any:
        return self.backend.backend_model

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "FQTreeClassifier":
        self.backend.fit(X, y, **kwargs)
        return self

    def predict(self, X: Any, **kwargs: Any) -> Any:
        return self.backend.predict(X, **kwargs)

    def predict_proba(self, X: Any, **kwargs: Any) -> Any:
        return self.backend.predict_proba(X, **kwargs)

    def score(self, X: Any, y: Any, **kwargs: Any) -> Any:
        return self.backend.score(X, y, **kwargs)

    def ibooster(self) -> Any:
        return self.to_integer_booster()

    def to_integer_booster(self) -> Any:
        return self.backend.integer_booster()

    def trace(self, *, inputs: Any, mode: Optional[str] = None, **kwargs: Any) -> Any:
        from .hardware import trace_fqtree_model

        return trace_fqtree_model(
            self,
            inputs=inputs,
            mode=mode or self.config.hardware_mode,
            **kwargs,
        )

    def to_qxgb(self) -> Any:
        return self.backend_model_

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        params = self.config.to_qxgb_params()
        params["training_mode"] = self.config.training_mode
        params["hardware_mode"] = self.config.hardware_mode
        params["backend"] = self.backend_name
        return params

    def set_params(self, **params: Any) -> "FQTreeClassifier":
        backend = params.pop("backend", self.backend_name)
        self.config = FQTreeConfig.from_kwargs(self.config, **params)
        self.backend = self._make_backend(backend)
        return self

    def __getattr__(self, name: str) -> Any:
        backend = self.__dict__.get("backend")
        if backend is not None:
            backend_model = getattr(backend, "backend_model", None)
            if backend_model is not None and hasattr(backend_model, name):
                return getattr(backend_model, name)
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

    def _make_backend(self, backend: Union[str, Any]) -> Any:
        if backend == "qxgb":
            self.backend_name = QXGBBackend.name
            return QXGBBackend(self.config)
        if isinstance(backend, type):
            created = backend(self.config)
            self.backend_name = getattr(created, "name", type(created).__name__)
            return created
        if hasattr(backend, "fit") and hasattr(backend, "integer_booster"):
            self.backend_name = getattr(backend, "name", type(backend).__name__)
            return backend
        raise ValueError("backend must be 'qxgb', a backend class, or a backend instance.")


# Compatibility alias for older examples and downstream code.
QXGBClassifier = FQTreeClassifier

__all__ = ["FQTreeClassifier", "QXGBClassifier"]
