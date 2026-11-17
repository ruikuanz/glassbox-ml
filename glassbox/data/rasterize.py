# scanline polygon fill

from __future__ import annotations

import numpy as np


# rings are (K, 2) vertices in pixel coordinates
def rasterize_polygon(rings: list[np.ndarray], out: np.ndarray, value: int) -> None:
    H, W = out.shape
    starts = np.concatenate([ring for ring in rings])
    ends = np.concatenate([np.roll(ring, -1, axis=0) for ring in rings])
    x1, y1, x2, y2 = starts[:, 0], starts[:, 1], ends[:, 0], ends[:, 1]

    r_min = max(int(np.floor(starts[:, 1].min() - 0.5)), 0)
    r_max = min(int(np.ceil(starts[:, 1].max() - 0.5)), H - 1)
    c_min = max(int(np.floor(starts[:, 0].min() - 0.5)), 0)
    c_max = min(int(np.ceil(starts[:, 0].max() - 0.5)), W - 1)
    if r_min > r_max or c_min > c_max:
        return
    n_rows, n_cols = r_max - r_min + 1, c_max - c_min + 1

    y = (np.arange(r_min, r_max + 1) + 0.5)[:, None]

    crosses = (np.minimum(y1, y2) <= y) & (y < np.maximum(y1, y2)) # half-open so a vertex counts once
    row_idx, edge_idx = np.nonzero(crosses)
    if len(row_idx) == 0:
        return
    e = edge_idx
    x_cross = x1[e] + (y[row_idx, 0] - y1[e]) * (x2[e] - x1[e]) / (y2[e] - y1[e])

    first_col = np.ceil(x_cross - 0.5).astype(np.int64) - c_min # first pixel center at or right of x
    first_col = np.clip(first_col, 0, n_cols)
    flips = np.zeros((n_rows, n_cols + 1), dtype=np.int32)
    np.add.at(flips, (row_idx, first_col), 1)
    inside = np.cumsum(flips[:, :n_cols], axis=1) % 2 == 1

    out[r_min:r_max + 1, c_min:c_max + 1][inside] = value
