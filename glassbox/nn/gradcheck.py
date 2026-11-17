# gradient checking against two-sided finite differences (run in float64)

from __future__ import annotations

from typing import Callable

import numpy as np


# J reads theta in place; returns the flat indices checked and the gradient there
def numerical_gradient(J: Callable[[], float], theta: np.ndarray, eps: float = 1e-6,
                       max_entries: int | None = None,
                       rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    flat = theta.reshape(-1) # view, so writes perturb theta
    indices = np.arange(flat.size)
    if max_entries is not None and flat.size > max_entries:
        rng = rng or np.random.default_rng(0)
        indices = rng.choice(flat.size, size=max_entries, replace=False)

    gradapprox = np.zeros(len(indices), dtype=np.float64)
    for k, i in enumerate(indices):
        original = flat[i]
        flat[i] = original + eps
        J_plus = J()
        flat[i] = original - eps
        J_minus = J()
        flat[i] = original
        gradapprox[k] = (J_plus - J_minus) / (2 * eps)
    return indices, gradapprox


# about 1e-7 means correct, 1e-3 or above means a bug
def relative_error(grad: np.ndarray, gradapprox: np.ndarray) -> float:
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
