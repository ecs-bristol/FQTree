from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Tuple, Union


_CONFIG_KEYS = {"scale", "bias", "quantizer", "training_mode", "hardware_mode", "xgb_params"}


@dataclass
class FQTreeConfig:
    """Configuration owned by FQTree and translated to the active backend."""

    scale: Union[float, Callable[..., float]] = 1.0
    bias: Union[float, Callable[..., float]] = 0.0
    quantizer: Optional[Callable[..., Any]] = None
    training_mode: Literal["qat", "ptq"] = "qat"
    hardware_mode: Literal["mux", "masking"] = "mux"
    xgb_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_kwargs(
        cls,
        config: Optional[Union["FQTreeConfig", Mapping[str, Any]]] = None,
        **kwargs: Any,
    ) -> "FQTreeConfig":
        base_data: Dict[str, Any] = {}
        if config is None:
            pass
        elif isinstance(config, cls):
            base_data = config.to_dict()
        elif isinstance(config, Mapping):
            base_data = dict(config)
        else:
            raise TypeError("config must be an FQTreeConfig, a mapping, or None.")

        base_known, base_xgb = _split_config_params(base_data)
        override_known, override_xgb = _split_config_params(kwargs)

        xgb_params = dict(base_xgb)
        xgb_params.update(override_xgb)

        data = dict(base_known)
        data.update(override_known)
        data["xgb_params"] = xgb_params
        return cls(**data)

    def to_qxgb_params(self) -> Dict[str, Any]:
        params = dict(self.xgb_params)
        params["scale"] = self.scale
        params["bias"] = self.bias
        if self.quantizer is not None:
            params["quantizer"] = self.quantizer
        return params

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scale": self.scale,
            "bias": self.bias,
            "quantizer": self.quantizer,
            "training_mode": self.training_mode,
            "hardware_mode": self.hardware_mode,
            "xgb_params": dict(self.xgb_params),
        }


def _split_config_params(params: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    known: Dict[str, Any] = {}
    xgb_params: Dict[str, Any] = {}

    for key, value in params.items():
        if key == "xgb_params":
            if value:
                xgb_params.update(dict(value))
        elif key in _CONFIG_KEYS:
            known[key] = value
        elif key == "mode":
            if value in ("mux", "masking"):
                known["hardware_mode"] = value
            else:
                known["training_mode"] = value
        else:
            xgb_params[key] = value

    return known, xgb_params
