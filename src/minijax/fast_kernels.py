import numpy as np
from numba import njit, prange


def _njit_fast(fn):
    try:
        return njit(cache=True, fastmath=True, parallel=True)(fn)
    except RuntimeError:
        return njit(fastmath=True, parallel=True)(fn)


@_njit_fast
def conv2d_nchw(x, kernel, stride):
    batch, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = kernel.shape
    out_h = (height - kernel_h) // stride + 1
    out_w = (width - kernel_w) // stride + 1
    out = np.empty((batch, out_channels, out_h, out_w), dtype=np.float64)

    total = batch * out_channels * out_h * out_w
    for idx in prange(total):
        # flatten 4D output index so each parallel iteration has one output cell
        out_col = idx % out_w
        tmp = idx // out_w
        out_row = tmp % out_h
        tmp //= out_h
        out_channel = tmp % out_channels
        batch_index = tmp // out_channels

        acc = 0.0
        in_row_start = out_row * stride
        in_col_start = out_col * stride
        for in_channel in range(in_channels):
            for kernel_row in range(kernel_h):
                in_row = in_row_start + kernel_row
                for kernel_col in range(kernel_w):
                    acc += (
                        x[batch_index, in_channel, in_row, in_col_start + kernel_col]
                        * kernel[out_channel, in_channel, kernel_row, kernel_col]
                    )
        out[batch_index, out_channel, out_row, out_col] = acc

    return out


@_njit_fast
def conv2d_input_grad_nchw(tangent, kernel, primal, stride):
    batch, in_channels, height, width = primal.shape
    out_channels, _, kernel_h, kernel_w = kernel.shape
    out_h = tangent.shape[2]
    out_w = tangent.shape[3]
    grad = np.empty_like(primal)

    total = batch * in_channels * height * width
    for idx in prange(total):
        # compute one input gradient cell at a time to avoid race condiions
        in_col = idx % width
        tmp = idx // width
        in_row = tmp % height
        tmp //= height
        in_channel = tmp % in_channels
        batch_index = tmp // in_channels

        acc = 0.0
        for out_channel in range(out_channels):
            for kernel_row in range(kernel_h):
                row_delta = in_row - kernel_row
                if row_delta < 0 or row_delta % stride != 0:
                    continue
                out_row = row_delta // stride
                if out_row >= out_h:
                    continue
                for kernel_col in range(kernel_w):
                    col_delta = in_col - kernel_col
                    if col_delta < 0 or col_delta % stride != 0:
                        continue
                    out_col = col_delta // stride
                    if out_col < out_w:
                        acc += (
                            tangent[batch_index, out_channel, out_row, out_col]
                            * kernel[out_channel, in_channel, kernel_row, kernel_col]
                        )
        grad[batch_index, in_channel, in_row, in_col] = acc

    return grad


@_njit_fast
def conv2d_kernel_grad_nchw(tangent, primal, kernel, stride):
    batch, in_channels, _, _ = primal.shape
    out_channels, _, kernel_h, kernel_w = kernel.shape
    out_h = tangent.shape[2]
    out_w = tangent.shape[3]
    grad = np.empty_like(kernel)

    total = out_channels * in_channels * kernel_h * kernel_w
    for idx in prange(total):
        # each kernel gradient cell sums all batches and output positions that used it
        kernel_col = idx % kernel_w
        tmp = idx // kernel_w
        kernel_row = tmp % kernel_h
        tmp //= kernel_h
        in_channel = tmp % in_channels
        out_channel = tmp // in_channels

        acc = 0.0
        for batch_index in range(batch):
            for out_row in range(out_h):
                in_row = out_row * stride + kernel_row
                for out_col in range(out_w):
                    acc += (
                        tangent[batch_index, out_channel, out_row, out_col]
                        * primal[batch_index, in_channel, in_row, out_col * stride + kernel_col]
                    )
        grad[out_channel, in_channel, kernel_row, kernel_col] = acc

    return grad


@_njit_fast
def avgpool4d(x, window_size, stride):
    batch, channels, height, width = x.shape
    window_b, window_c, window_h, window_w = window_size
    stride_b, stride_c, stride_h, stride_w = stride
    out_b = (batch - window_b) // stride_b + 1
    out_c = (channels - window_c) // stride_c + 1
    out_h = (height - window_h) // stride_h + 1
    out_w = (width - window_w) // stride_w + 1
    out = np.empty((out_b, out_c, out_h, out_w), dtype=np.float64)
    scale = 1.0 / (window_b * window_c * window_h * window_w)

    total = out_b * out_c * out_h * out_w
    for idx in prange(total):
        # pooling expressed over all 4 NCHW axes, but batch and channel dim typically 1
        out_col = idx % out_w
        tmp = idx // out_w
        out_row = tmp % out_h
        tmp //= out_h
        out_channel = tmp % out_c
        out_batch = tmp // out_c

        acc = 0.0
        for batch_offset in range(window_b):
            in_batch = out_batch * stride_b + batch_offset
            for channel_offset in range(window_c):
                in_channel = out_channel * stride_c + channel_offset
                for row_offset in range(window_h):
                    in_row = out_row * stride_h + row_offset
                    for col_offset in range(window_w):
                        acc += x[in_batch, in_channel, in_row, out_col * stride_w + col_offset]
        out[out_batch, out_channel, out_row, out_col] = acc * scale

    return out


@_njit_fast
def avgpool4d_grad(tangent, primal, window_size, stride):
    batch, channels, height, width = primal.shape
    window_b, window_c, window_h, window_w = window_size
    stride_b, stride_c, stride_h, stride_w = stride
    out_b, out_c, out_h, out_w = tangent.shape
    grad = np.empty_like(primal)
    scale = 1.0 / (window_b * window_c * window_h * window_w)

    total = batch * channels * height * width
    for idx in prange(total):
        in_col = idx % width
        tmp = idx // width
        in_row = tmp % height
        tmp //= height
        in_channel = tmp % channels
        in_batch = tmp // channels

        acc = 0.0
        for batch_offset in range(window_b):
            batch_delta = in_batch - batch_offset
            if batch_delta < 0 or batch_delta % stride_b != 0:
                continue
            out_batch = batch_delta // stride_b
            if out_batch >= out_b:
                continue
            for channel_offset in range(window_c):
                channel_delta = in_channel - channel_offset
                if channel_delta < 0 or channel_delta % stride_c != 0:
                    continue
                out_channel = channel_delta // stride_c
                if out_channel >= out_c:
                    continue
                for row_offset in range(window_h):
                    row_delta = in_row - row_offset
                    if row_delta < 0 or row_delta % stride_h != 0:
                        continue
                    out_row = row_delta // stride_h
                    if out_row >= out_h:
                        continue
                    for col_offset in range(window_w):
                        col_delta = in_col - col_offset
                        if col_delta < 0 or col_delta % stride_w != 0:
                            continue
                        out_col = col_delta // stride_w
                        if out_col < out_w:
                            acc += tangent[out_batch, out_channel, out_row, out_col]
        grad[in_batch, in_channel, in_row, in_col] = acc * scale

    return grad
