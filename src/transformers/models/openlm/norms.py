import numbers
from functools import partial
from typing import Union, List

import torch
from torch import Tensor, Size
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class LayerNorm(nn.Module):
    # NOTE: taken from official pytorch implementation and modified
    # to allow revoval of gain and bias independently

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 0.00001,
        elementwise_gain: bool = True,
        elementwise_bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_gain = elementwise_gain
        self.elementwise_bias = elementwise_bias

        if self.elementwise_gain:
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)

        if self.elementwise_bias:
            self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_gain:
            with torch.no_grad():
                self.weight.fill_(1.0)

        if self.elementwise_bias:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_gain={elementwise_gain}, "
            "elementwise_bias={elementwise_bias}".format(**self.__dict__)
        )


class LPLayerNorm(LayerNorm):
    """From MosaicML composer.

    See: https://github.com/mosaicml/composer/blob/6acca4c70425455be7280a5459dbf02e1ac5591d/composer/algorithms/low_precision_layernorm/low_precision_layernorm.py#L63
    """

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
        downcast_bias = _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
        with torch.autocast(enabled=False, device_type=module_device.type):
            return F.layer_norm(
                downcast_x,
                self.normalized_shape,
                downcast_weight,
                downcast_bias,
                self.eps,
            )


def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == "cuda":
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == "cpu":
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor


class RmsNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        self.reset_parameters()

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.fill_(1.0)

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps} ".format(**self.__dict__)


def get_norm_class(model_norm):
    if model_norm == "default_layer_norm":
        return torch.nn.LayerNorm
    elif model_norm == "lp_layer_norm":
        return LPLayerNorm
    elif model_norm == "gain_only_lp_layer_norm":
        return partial(LPLayerNorm, elementwise_gain=True, elementwise_bias=False)
    elif model_norm == "gain_only_layer_norm":
        return partial(LayerNorm, elementwise_gain=True, elementwise_bias=False)

    elif model_norm == "no_wb_layer_norm":
        return partial(LayerNorm, elementwise_gain=False, elementwise_bias=False)

    elif model_norm == "rms_norm":
        return RmsNorm

    else:
        raise ValueError(f"Unsupported model-norm: {model_norm}")