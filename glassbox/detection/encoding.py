# translate between boxes and the S x S grid of anchors the network predicts on

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from glassbox.detection.boxes import iou_wh, non_max_suppression

# channels of the last axis; class logits only when C > 1
OBJ = 0
X, Y, W, H = 1, 2, 3, 4
CLS = 5


@dataclass(frozen=True)
class GridSpec:
    image_size: int
    grid_size: int
    anchors: np.ndarray # (A, 2) width, height in pixels
    num_classes: int

    @property
    def cell_size(self) -> float:
        return self.image_size / self.grid_size

    @property
    def num_anchors(self) -> int:
        return len(self.anchors)

    @property
    def depth(self) -> int:
        return 5 + (self.num_classes if self.num_classes > 1 else 0)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60))) # clip guards exp against overflow


# label tensor (m, S, S, A, depth), and count of boxes dropped in slot collisions
def encode_targets(boxes_per_image: list[np.ndarray], classes_per_image: list[np.ndarray],
                   spec: GridSpec, dtype=np.float32) -> tuple[np.ndarray, int]:
    m = len(boxes_per_image)
    S, A = spec.grid_size, spec.num_anchors
    y = np.zeros((m, S, S, A, spec.depth), dtype=dtype)
    dropped = 0
    for i, (boxes, classes) in enumerate(zip(boxes_per_image, classes_per_image)):
        if len(boxes) == 0:
            continue
        wh = boxes[:, 2:4] - boxes[:, 0:2]
        center = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
        for k in np.argsort(-(wh[:, 0] * wh[:, 1]), kind="stable"): # larger box wins a collision
            gx, gy = center[k] / spec.cell_size
            col = min(int(gx), S - 1)
            row = min(int(gy), S - 1)
            a = int(iou_wh(wh[k:k + 1], spec.anchors)[0].argmax())
            if y[i, row, col, a, OBJ] == 1:
                dropped += 1
                continue
            y[i, row, col, a, OBJ] = 1
            y[i, row, col, a, X] = gx - col
            y[i, row, col, a, Y] = gy - row
            y[i, row, col, a, W] = np.log(wh[k, 0] / spec.anchors[a, 0])
            y[i, row, col, a, H] = np.log(wh[k, 1] / spec.anchors[a, 1])
            if spec.num_classes > 1:
                y[i, row, col, a, CLS + int(classes[k])] = 1
    return y, dropped


# raw outputs (m, S, S, A, depth) -> boxes, scores, classes per slot
def decode_boxes(t: np.ndarray, spec: GridSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, S, _, A, _ = t.shape
    cols = np.arange(S).reshape(1, 1, S, 1)
    rows = np.arange(S).reshape(1, S, 1, 1)
    b_x = (cols + sigmoid(t[..., X])) * spec.cell_size
    b_y = (rows + sigmoid(t[..., Y])) * spec.cell_size
    b_w = spec.anchors[:, 0] * np.exp(np.clip(t[..., W], -10, 10))
    b_h = spec.anchors[:, 1] * np.exp(np.clip(t[..., H], -10, 10))
    boxes = np.stack([b_x - b_w / 2, b_y - b_h / 2, b_x + b_w / 2, b_y + b_h / 2], axis=-1)
    boxes = np.clip(boxes, 0, spec.image_size)

    scores = sigmoid(t[..., OBJ])
    if spec.num_classes > 1:
        logits = t[..., CLS:]
        logits = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=-1, keepdims=True)
        classes = probs.argmax(axis=-1)
        scores = scores * probs.max(axis=-1)
    else:
        classes = np.zeros(scores.shape, dtype=np.int64)
    return boxes.reshape(m, -1, 4), scores.reshape(m, -1), classes.reshape(m, -1)


# decode, threshold, then non-max suppression per class
def decode_predictions(t: np.ndarray, spec: GridSpec, score_threshold: float,
                       nms_iou: float = 0.5, max_detections: int = 100
                       ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    boxes, scores, classes = decode_boxes(t, spec)
    detections = []
    for b, s, c in zip(boxes, scores, classes):
        confident = s >= score_threshold
        b, s, c = b[confident], s[confident], c[confident]
        keep = []
        for cls in np.unique(c):
            idx = np.nonzero(c == cls)[0]
            keep.append(idx[non_max_suppression(b[idx], s[idx], nms_iou)])
        keep = np.concatenate(keep) if keep else np.zeros(0, dtype=np.int64)
        keep = keep[np.argsort(-s[keep], kind="stable")][:max_detections]
        detections.append((b[keep], s[keep], c[keep]))
    return detections
