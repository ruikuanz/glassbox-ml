# layers with hand-derived backward passes, channels-last (m, n_H, n_W, n_C)

from __future__ import annotations

import numpy as np

from glassbox.nn.init import he_normal
from glassbox.nn.module import Layer, Parameter


# numpy on apple accelerate raises spurious warnings from matmul
def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return a @ b


# one row per f x f patch, so convolution becomes a single matrix product
def im2col(a_pad: np.ndarray, f: int, n_H: int, n_W: int) -> np.ndarray:
    m, _, _, n_C = a_pad.shape
    cols = np.empty((m, n_H, n_W, f, f, n_C), dtype=a_pad.dtype)
    for i in range(f):
        for j in range(f):
            cols[:, :, :, i, j, :] = a_pad[:, i:i + n_H, j:j + n_W, :]
    return cols.reshape(m * n_H * n_W, f * f * n_C)


# adjoint of im2col: sum patch gradients back onto the padded input
def col2im(dcols: np.ndarray, pad_shape: tuple[int, ...], f: int,
           n_H: int, n_W: int) -> np.ndarray:
    m, _, _, n_C = pad_shape
    dcols = dcols.reshape(m, n_H, n_W, f, f, n_C)
    da_pad = np.zeros(pad_shape, dtype=dcols.dtype)
    for i in range(f):
        for j in range(f):
            da_pad[:, i:i + n_H, j:j + n_W, :] += dcols[:, :, :, i, j, :]
    return da_pad


# stride 1, same padding; W is (f, f, n_C_prev, n_C)
class Conv2D(Layer):
    def __init__(self, n_C_prev: int, n_C: int, f: int, rng: np.random.Generator,
                 dtype=np.float32, input_grad: bool = True):
        if f % 2 != 1:
            raise ValueError(f"filter size must be odd for same padding, got {f}")
        self.f = f
        self.pad = (f - 1) // 2
        self.input_grad = input_grad # False on the first layer skips dJ/d(image)
        self.W = Parameter(
            he_normal((f, f, n_C_prev, n_C), fan_in=f * f * n_C_prev, rng=rng, dtype=dtype),
            kind="weight",
        )
        self.b = Parameter(np.zeros(n_C, dtype=dtype), kind="bias")
        self.cache = None

    def parameters(self) -> dict[str, Parameter]:
        return {"W": self.W, "b": self.b}

    def forward(self, A_prev: np.ndarray, training: bool = False) -> np.ndarray:
        m, n_H, n_W, n_C_prev = A_prev.shape
        f, p = self.f, self.pad
        A_prev_pad = np.pad(A_prev, ((0, 0), (p, p), (p, p), (0, 0)))
        cols = im2col(A_prev_pad, f, n_H, n_W)
        W_mat = self.W.data.reshape(f * f * n_C_prev, -1) # one column per filter
        Z = matmul(cols, W_mat) + self.b.data
        self.cache = (cols, A_prev_pad.shape, n_H, n_W)
        return Z.reshape(m, n_H, n_W, -1)

    def backward(self, dZ: np.ndarray) -> np.ndarray | None:
        cols, pad_shape, n_H, n_W = self.cache
        f, p = self.f, self.pad
        n_C = dZ.shape[-1]
        dZ = dZ.reshape(-1, n_C)

        self.W.grad[...] = matmul(cols.T, dZ).reshape(self.W.data.shape)
        self.b.grad[...] = dZ.sum(axis=0)
        if not self.input_grad:
            return None

        dcols = matmul(dZ, self.W.data.reshape(-1, n_C).T)
        dA_prev_pad = col2im(dcols, pad_shape, f, n_H, n_W)
        if p == 0:
            return dA_prev_pad
        return dA_prev_pad[:, p:-p, p:-p, :]


# non-overlapping f x f windows
class MaxPool2D(Layer):
    def __init__(self, f: int = 2):
        self.f = f
        self.cache = None

    def forward(self, A_prev: np.ndarray, training: bool = False) -> np.ndarray:
        m, n_H_prev, n_W_prev, n_C = A_prev.shape
        f = self.f
        if n_H_prev % f or n_W_prev % f:
            raise ValueError(f"input {n_H_prev}x{n_W_prev} is not divisible by pool size {f}")
        windows = A_prev.reshape(m, n_H_prev // f, f, n_W_prev // f, f, n_C)
        A = windows.max(axis=(2, 4))
        self.cache = (windows, A)
        return A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        windows, A = self.cache
        mask = (windows == A[:, :, None, :, None, :]).astype(dA.dtype)
        share = mask / mask.sum(axis=(2, 4), keepdims=True) # ties share the gradient
        dA_prev = share * dA[:, :, None, :, None, :]
        m, n_H, f, n_W, _, n_C = windows.shape
        return dA_prev.reshape(m, n_H * f, n_W * f, n_C)


class LeakyReLU(Layer):
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.cache = None

    def forward(self, Z: np.ndarray, training: bool = False) -> np.ndarray:
        positive = Z > 0
        self.cache = positive
        return np.where(positive, Z, self.alpha * Z).astype(Z.dtype, copy=False)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        positive = self.cache
        return np.where(positive, dA, self.alpha * dA).astype(dA.dtype, copy=False)


class ReLU(LeakyReLU):
    def __init__(self):
        super().__init__(alpha=0.0)
