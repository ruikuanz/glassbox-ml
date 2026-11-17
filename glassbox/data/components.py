# turn connected blobs in pixel-level mask into smallest satisfying bounding boxes

from __future__ import annotations

import numpy as np


# boolean mask (horizontal)
def find_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, W = mask.shape
    padded = np.zeros((H, W + 2), dtype=np.int8) # pad for blobs on border
    padded[:, 1:-1] = mask
    step = np.diff(padded, axis=1)
    rows, starts = np.nonzero(step == 1)
    _, ends = np.nonzero(step == -1)
    return rows, starts, ends


# label runs in boolean mask, returning the label of each run and the number of components
def label_runs(rows: np.ndarray, starts: np.ndarray, ends: np.ndarray,
               connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Give every run the label (0 .. n - 1) of the component it belongs to."""
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    reach = 1 if connectivity == 8 else 0

    parent = list(range(len(rows)))

    def find(k: int) -> int:
        """The representative run of k's component."""
        root = k
        while parent[root] != root:
            root = parent[root]
        while parent[k] != root:
            parent[k], k = root, parent[k]
        return root

    n_rows = int(rows.max()) + 1 if len(rows) else 0
    first_run = np.searchsorted(rows, np.arange(n_rows + 1)).tolist()
    s, e = starts.tolist(), ends.tolist()

    for r in range(1, n_rows):
        i, i_end = first_run[r - 1], first_run[r] 
        j, j_end = first_run[r], first_run[r + 1] 
        while i < i_end and j < j_end: # sorted left to right
            if s[i] < e[j] + reach and s[j] < e[i] + reach:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_j] = root_i
            if e[i] < e[j]:
                i += 1
            else:
                j += 1

    roots = np.array([find(k) for k in range(len(rows))], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels, (int(labels.max()) + 1 if len(labels) else 0)


def mask_to_boxes(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rows, starts, ends = find_runs(mask)
    labels, n = label_runs(rows, starts, ends, connectivity)
    x1 = np.full(n, np.iinfo(np.int64).max)
    y1 = np.full(n, np.iinfo(np.int64).max)
    x2 = np.zeros(n, dtype=np.int64)
    y2 = np.zeros(n, dtype=np.int64)
    areas = np.zeros(n, dtype=np.int64)
    np.minimum.at(x1, labels, starts)
    np.minimum.at(y1, labels, rows)
    np.maximum.at(x2, labels, ends)
    np.maximum.at(y2, labels, rows + 1)
    np.add.at(areas, labels, ends - starts)
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32).reshape(n, 4)
    return boxes, areas


def label_image(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    rows, starts, ends = find_runs(mask)
    labels, n = label_runs(rows, starts, ends, connectivity)
    out = np.zeros(mask.shape, dtype=np.int32)
    for r, s, e, lab in zip(rows, starts, ends, labels):
        out[r, s:e] = lab + 1
    return out, n
