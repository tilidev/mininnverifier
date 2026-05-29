# Copyright (c) 2025 by David Boetius
# Licensed under the MIT Licensed.
import numpy as np
import scipy.special as sp

from . import core


class Array(core.Value):
    def __init__(self, array_like):
        self.array = np.asarray(array_like, dtype=np.float64)
        super().__init__(EvalInterpreter(), self.array.shape)

    def item(self):
        return self.array.item()

    def __repr__(self):
        data_str = str(self.array).replace("\n", "\n" + " " * len("Array("))
        return f"Array({data_str})"


def full(shape, fill_value):
    return Array(np.full(shape, fill_value, dtype=np.float64))


def zeros(shape):
    return full(shape, 0.0)


def ones(shape):
    return full(shape, 1.0)


class EvalInterpreter(core.Interpreter[Array]):
    def __init__(self):
        super().__init__(0)

    def wrap(self, value):
        if not isinstance(value, core.Value):
            return Array(value)
        elif not isinstance(value, Array):
            raise ValueError("EvalInterpreter must be the bottom interpreter")
        return value

    def process(self, primitive, values: list[Array], options: dict):
        np_vals = [v.array for v in values]
        np_out = eval_rules[primitive](*np_vals, **options)
        return Array(np_out)


def np_dot(x, y):  # np.dot doesn't broadcast
    if y.ndim <= 1:
        return np.dot(x, y)
    return np.einsum("...j,...jk", x, y)


def np_normalcdf(x):
    return (1 + sp.erf(x / np.sqrt(2))) / 2


def _normalize_axes(axes, ndim):
    return tuple(ax + ndim if ax < 0 else ax for ax in axes)


def np_pad(x, config, axes, value):
    left, right, interior = config
    axes = _normalize_axes(axes, x.ndim)
    out_shape = list(x.shape)
    for axis in axes:
        out_shape[axis] = left + x.shape[axis] + (x.shape[axis] - 1) * interior + right

    out = np.full(out_shape, value, dtype=np.float64)
    slices = [slice(None)] * x.ndim
    for axis in axes:
        stop = left + (interior + 1) * x.shape[axis]
        slices[axis] = slice(left, stop, interior + 1)
    out[tuple(slices)] = x
    return out


def np_unpad(tangent, primal, config, axes):
    left, _, interior = config
    axes = _normalize_axes(axes, tangent.ndim)
    slices = [slice(None)] * tangent.ndim
    for axis in axes:
        stop = left + (interior + 1) * primal.shape[axis]
        slices[axis] = slice(left, stop, interior + 1)
    return tangent[tuple(slices)]


def _conv_windows(x, kernel_shape, stride):
    _, _, kernel_h, kernel_w = kernel_shape
    # shape: (N, Cin, Hout, Wout, kH, kW), matching einsum computation 
    windows = np.lib.stride_tricks.sliding_window_view(x, (kernel_h, kernel_w), axis=(2, 3))
    return windows[:, :, ::stride, ::stride, :, :]


def np_conv(x, kernel, stride):
    windows = _conv_windows(x, kernel.shape, stride)
    return np.einsum("nchwkl,ockl->nohw", windows, kernel)


def np_conv_input_grad(tangent, kernel, primal, stride):
    grad = np.zeros_like(primal, dtype=np.float64)
    _, _, kernel_h, kernel_w = kernel.shape
    out_h, out_w = tangent.shape[2:]

    for row in range(kernel_h):
        row_stop = row + stride * out_h
        for col in range(kernel_w):
            col_stop = col + stride * out_w
            # scatter each output tangent back to the input pixels touched by kernel offset
            grad[:, :, row:row_stop:stride, col:col_stop:stride] += np.einsum(
                "nohw,oc->nchw", tangent, kernel[:, :, row, col]
            )
    return grad


def np_conv_kernel_grad(tangent, primal, kernel, stride):
    windows = _conv_windows(primal, kernel.shape, stride)
    return np.einsum("nohw,nchwkl->ockl", tangent, windows)


eval_rules = {
    core.expand_dims: lambda x, axes: np.expand_dims(x, axes),
    core.moveaxis: np.moveaxis,
    core.reshape: lambda x, new_shape: np.reshape(x, new_shape),
    core.neg: lambda x: -x,
    core.add: lambda x, y: x + y,
    core.reduce_sum: lambda x, axes: x.sum(axes),
    core.dot: np_dot,
    core.mul: lambda x, y: x * y,
    core.reciprocal: lambda x: 1 / x,
    core.relu: lambda x: np.maximum(x, 0.0),
    core.square: np.square,
    core.sqrt: np.sqrt,
    core.exp: np.exp,
    core.log: np.log,
    core.where: np.where,
    core.leaky_relu: lambda x, slope: np.where(x > 0, x, slope*x),
    core.normalcdf: np_normalcdf,
    core.elu: lambda x: np.where(x > 0, x, np.exp(x) - 1),
    core.gelu: lambda x: x * np_normalcdf(x),
    core.pad: np_pad,
    core.unpad: np_unpad,
    core.conv: np_conv,
    core.conv_input_grad: np_conv_input_grad,
    core.conv_kernel_grad: np_conv_kernel_grad,
}
