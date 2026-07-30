from typing import Any

from ..config import FQTreeConfig


class QXGBBackend:
    """Adapter from FQTree's public API to the qxgb implementation."""

    name = "qxgb"

    def __init__(self, config: FQTreeConfig):
        self.config = config
        self.model = self._build_model(config)

    @property
    def backend_model(self) -> Any:
        return self.model

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "QXGBBackend":
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X: Any, **kwargs: Any) -> Any:
        return self.model.predict(X, **kwargs)

    def predict_proba(self, X: Any, **kwargs: Any) -> Any:
        return self.model.predict_proba(X, **kwargs)

    def score(self, X: Any, y: Any, **kwargs: Any) -> Any:
        return self.model.score(X, y, **kwargs)

    def integer_booster(self) -> Any:
        return self.model.ibooster()

    def _build_model(self, config: FQTreeConfig) -> Any:
        if config.training_mode != "qat":
            raise NotImplementedError("The qxgb backend currently implements only FQTree QAT mode.")

        from qxgb import QXGBClassifier

        return QXGBClassifier(**config.to_qxgb_params())
