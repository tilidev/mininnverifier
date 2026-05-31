# Copyright (c) 2025 by David Boetius
# Licensed under the MIT Licensed.
import math

from .core import (
    avgpool,
    conv,
    elu,
    exp,
    gelu,
    leaky_relu,
    log,
    reduce_sum,
    relu,
    reshape,
    square,
)
from .eval import Array, zeros
from .nested_containers import flatten, map_structure
from .random import rand_uniform, split_rng_key


def linear(x, weight, bias):
    return x @ weight + bias


def mlp(x, params: list[dict[str, Array]]):
    x = reshape(x, (-1,))
    for layer_params in params[:-1]:
        x = linear(x, layer_params["weight"], layer_params["bias"])
        x = relu(x)
    return linear(x, params[-1]["weight"], params[-1]["bias"])


def conv_nn(x, params, input_shape, conv_layers):
    x = reshape(x, (x.shape[0],) + tuple(input_shape))
    for layer_params, layer_cfg in zip(params["conv_layers"], conv_layers, strict=True):
        x = conv(x, layer_params["kernel"], stride=layer_cfg["stride"]) + layer_params["bias"]
        x = _activation(x, layer_cfg["activation"], layer_cfg["activation_slope"])
        if layer_cfg["pool_window_size"] is not None:
            x = avgpool(
                x, window_size=layer_cfg["pool_window_size"], stride=layer_cfg["pool_stride"]
            )

    x = reshape(x, (x.shape[0], -1))
    for layer_params in params["dense_layers"][:-1]:
        x = linear(x, layer_params["weight"], layer_params["bias"])
        x = relu(x)
    return linear(x, params["dense_layers"][-1]["weight"], params["dense_layers"][-1]["bias"])


def _activation(x, name, slope):
    if name == "relu":
        return relu(x)
    if name == "gelu":
        return gelu(x)
    if name == "elu":
        return elu(x)
    if name == "leaky_relu":
        return leaky_relu(x, slope=slope)
    if name in ("identity", "none", None):
        return x
    raise ValueError(f"Unknown activation: {name}")


def softmax(x, axis: int):
    x_mean = reduce_sum(x, axis, keepaxes=True) / Array(x.shape[axis])
    exp_x = exp(x - x_mean)  # more numerically stable softmax
    return exp_x / reduce_sum(exp_x, axis, keepaxes=True)


# ======================================================================================================================


def reduce_mean(x):
    return reduce_sum(x) / Array(math.prod(x.shape))


def cross_entropy(y_pred, y_true):
    y_pred = softmax(y_pred, axis=-1)
    return -reduce_mean(reduce_sum(y_true * log(y_pred), axes=-1))


def weight_decay(params):
    param_norms = map_structure(lambda p: reduce_mean(square(p)), params)
    return sum(flatten(param_norms)[0], start=Array(0))


# ======================================================================================================================


def init_mlp(in_size, layer_sizes, rng_key):  # layer_sizes[-1] is output size
    in_sizes = (in_size,) + tuple(layer_sizes[:-1])
    rng_keys = split_rng_key(rng_key, len(layer_sizes))
    return [
        {"weight": kaiming_uniform((in_, out), in_, key), "bias": zeros((out,))}
        for in_, out, key in zip(in_sizes, layer_sizes, rng_keys)
    ]


def init_conv_nn(input_shape, conv_layers, dense_layer_sizes, rng_key):
    conv_params = []
    feature_shape = tuple(input_shape)
    rng_keys = iter(split_rng_key(rng_key, len(conv_layers) + len(dense_layer_sizes)))

    for layer_cfg in conv_layers:
        in_channels, height, width = feature_shape
        out_channels = layer_cfg["out_channels"]
        kernel_h, kernel_w = layer_cfg["kernel_size"]
        fan_in = in_channels * kernel_h * kernel_w
        conv_params.append(
            {
                "kernel": kaiming_uniform(
                    (out_channels, in_channels, kernel_h, kernel_w), fan_in, next(rng_keys)
                ),
                "bias": zeros((out_channels, 1, 1)),
            }
        )

        stride = layer_cfg["stride"]
        height = (height - kernel_h) // stride + 1
        width = (width - kernel_w) // stride + 1
        if layer_cfg["pool_window_size"] is not None:
            _, _, pool_h, pool_w = layer_cfg["pool_window_size"]
            _, _, pool_stride_h, pool_stride_w = layer_cfg["pool_stride"]
            height = (height - pool_h) // pool_stride_h + 1
            width = (width - pool_w) // pool_stride_w + 1
        feature_shape = (out_channels, height, width)

    dense_in_size = math.prod(feature_shape)
    dense_in_sizes = (dense_in_size,) + tuple(dense_layer_sizes[:-1])
    dense_params = [
        {"weight": kaiming_uniform((in_, out), in_, next(rng_keys)), "bias": zeros((out,))}
        for in_, out in zip(dense_in_sizes, dense_layer_sizes, strict=True)
    ]
    return {"conv_layers": conv_params, "dense_layers": dense_params}


def kaiming_uniform(shape, fan_in, rng_key):
    # kaiming_uniform initialization for ReLU
    bound = math.sqrt(2) * math.sqrt(3 / fan_in)
    return rand_uniform(shape, -bound, bound, rng_key=rng_key)
