# iou and non-max suppression; boxes are (x1, y1, x2, y2) in continuous pixel coordinates

from __future__ import annotations

import numpy as np


def box_area(boxes: np.ndarray) -> np.ndarray:
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


# (K, 4) and (L, 4) boxes -> (K, L)
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a[:, None, :]
    b = b[None, :, :]
    xi1 = np.maximum(a[..., 0], b[..., 0])
    yi1 = np.maximum(a[..., 1], b[..., 1])
    xi2 = np.minimum(a[..., 2], b[..., 2])
    yi2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(xi2 - xi1, 0) * np.maximum(yi2 - yi1, 0)
    union = box_area(a) + box_area(b) - inter
    return inter / np.maximum(union, 1e-12)


# iou of (width, height) shapes as if they shared a center
def iou_wh(wh: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    inter = (np.minimum(wh[:, None, 0], anchors[None, :, 0])
             * np.minimum(wh[:, None, 1], anchors[None, :, 1]))
    union = (wh[:, 0] * wh[:, 1])[:, None] + (anchors[:, 0] * anchors[:, 1])[None, :] - inter
    return inter / np.maximum(union, 1e-12)


# keep the highest-scoring box, drop the others that overlap it, repeat
def non_max_suppression(boxes: np.ndarray, scores: np.ndarray,
                        iou_threshold: float) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size > 0:
        best = order[0]
        keep.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        overlaps = iou_matrix(boxes[best:best + 1], boxes[rest])[0]
        order = rest[overlaps < iou_threshold]
    return np.array(keep, dtype=np.int64)
