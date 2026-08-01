from typing import Any

from ..config import FQTreeConfig


class XGBoostPTQBackend:
    """Adapter for post-training quantization through Alkaid leaf quantizers."""

    name = "xgboost_ptq"

    def __init__(self, config: FQTreeConfig):
        self.config = config
        self.model = self._build_model(config)

    @property
    def backend_model(self) -> Any:
        return self.model

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "XGBoostPTQBackend":
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X: Any, **kwargs: Any) -> Any:
        return self.model.predict(X, **kwargs)

    def predict_proba(self, X: Any, **kwargs: Any) -> Any:
        return self.model.predict_proba(X, **kwargs)

    def score(self, X: Any, y: Any, **kwargs: Any) -> Any:
        return self.model.score(X, y, **kwargs)

    def integer_booster(self) -> Any:
        return self.model.get_booster()

    def trace_kwargs(self) -> dict[str, Any]:
        if self.config.quantizer is None:
            return {}
        return {"leaf_quantizer": self.config.quantizer}

    def _build_model(self, config: FQTreeConfig) -> Any:
        if config.training_mode != "ptq":
            raise NotImplementedError(
                "The xgboost_ptq backend implements only FQTree PTQ mode."
            )

        from xgboost import XGBClassifier

        return XGBClassifier(**config.to_xgboost_params())
