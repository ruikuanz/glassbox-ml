# weight initialization

from __future__ import annotations

import numpy as np


# W ~ N(0, 2 / fan_in)
def he_normal(shape: tuple[int, ...], fan_in: int, rng: np.random.Generator,
              dtype=np.float32) -> np.ndarray:
    return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(dtype)
