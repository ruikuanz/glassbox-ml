# detector: [conv 3x3 -> leaky relu -> maxpool 2x2] per stage, then conv 3x3 and conv 1x1

from __future__ import annotations

import numpy as np

from glassbox.detection.encoding import GridSpec
from glassbox.nn import Conv2D, LeakyReLU, MaxPool2D, Sequential


def build_detector(spec: GridSpec, stages: list[int], head_channels: int,
                   rng: np.random.Generator, in_channels: int = 3,
                   dtype=np.float32) -> Sequential:
    if spec.image_size != spec.grid_size * 2 ** len(stages):
        raise ValueError(
            f"{len(stages)} pooling stages turn a {spec.image_size}px image into a "
            f"{spec.image_size // 2 ** len(stages)}-cell grid, but the spec says {spec.grid_size}"
        )
    layers = []
    n_C_prev = in_channels
    for k, n_C in enumerate(stages):
        layers += [
            Conv2D(n_C_prev, n_C, 3, rng, dtype, input_grad=k > 0),
            LeakyReLU(0.1),
            MaxPool2D(2),
        ]
        n_C_prev = n_C
    layers += [
        Conv2D(n_C_prev, head_channels, 3, rng, dtype),
        LeakyReLU(0.1),
        Conv2D(head_channels, spec.num_anchors * spec.depth, 1, rng, dtype),
    ]
    return Sequential(layers)


# (m, S, S, A * depth) -> (m, S, S, A, depth)
def to_grid(output: np.ndarray, spec: GridSpec) -> np.ndarray:
    m, S, _, _ = output.shape
    return output.reshape(m, S, S, spec.num_anchors, spec.depth)
