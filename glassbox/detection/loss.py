# detection cost and its gradient: objectness bce, box squared error, class cross-entropy

from __future__ import annotations

import numpy as np

from glassbox.detection.encoding import CLS, H, OBJ, W, X, Y, sigmoid


class YoloLoss:
    def __init__(self, lambda_obj: float = 1.0, lambda_noobj: float = 0.5,
                 lambda_coord: float = 5.0, lambda_class: float = 1.0):
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord
        self.lambda_class = lambda_class
        self.cache = None

    def forward(self, t: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, float]]:
        m = t.shape[0]
        has_obj = y[..., OBJ] == 1

        t_o = t[..., OBJ]
        bce = np.maximum(t_o, 0) - t_o * y[..., OBJ] + np.log1p(np.exp(-np.abs(t_o))) # overflow-safe
        J_obj = self.lambda_obj * bce[has_obj].sum() / m
        J_noobj = self.lambda_noobj * bce[~has_obj].sum() / m

        xy = sigmoid(t[..., X:Y + 1])
        d_xy = (xy - y[..., X:Y + 1]) * has_obj[..., None]
        d_wh = (t[..., W:H + 1] - y[..., W:H + 1]) * has_obj[..., None]
        J_coord = self.lambda_coord * ((d_xy ** 2).sum() + (d_wh ** 2).sum()) / m

        probs = None
        J_class = 0.0
        if t.shape[-1] > CLS:
            logits = t[..., CLS:] - t[..., CLS:].max(axis=-1, keepdims=True)
            log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
            probs = np.exp(log_probs)
            J_class = self.lambda_class * -(y[..., CLS:] * log_probs)[has_obj].sum() / m

        self.cache = (t, y, has_obj, xy, d_xy, d_wh, probs)
        parts = {"obj": float(J_obj), "noobj": float(J_noobj),
                 "coord": float(J_coord), "class": float(J_class)}
        return float(J_obj + J_noobj + J_coord + J_class), parts

    # dJ/dt
    def backward(self) -> np.ndarray:
        t, y, has_obj, xy, d_xy, d_wh, probs = self.cache
        m = t.shape[0]
        dt = np.zeros_like(t)

        weight = np.where(has_obj, self.lambda_obj, self.lambda_noobj)
        dt[..., OBJ] = weight * (sigmoid(t[..., OBJ]) - y[..., OBJ]) # d bce / dt = sigmoid(t) - y

        dt[..., X:Y + 1] = self.lambda_coord * 2 * d_xy * xy * (1 - xy) # chain rule through the sigmoid
        dt[..., W:H + 1] = self.lambda_coord * 2 * d_wh

        if probs is not None:
            dt[..., CLS:] = self.lambda_class * (probs - y[..., CLS:]) * has_obj[..., None]

        return dt / m
