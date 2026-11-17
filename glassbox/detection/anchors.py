# anchor shapes from k-means on training boxes, with 1 - iou as the distance

from __future__ import annotations

import numpy as np

from glassbox.detection.boxes import iou_wh


# returns anchors sorted by area and the mean iou of each box with its best anchor
def kmeans_anchors(wh: np.ndarray, k: int, rng: np.random.Generator,
                   iterations: int = 100) -> tuple[np.ndarray, float]:
    wh = np.asarray(wh, dtype=np.float64)
    if len(wh) < k:
        raise ValueError(f"need at least {k} boxes to fit {k} anchors, got {len(wh)}")
    centroids = wh[rng.integers(len(wh))][None, :] # k-means++ seeding
    while len(centroids) < k:
        distance = 1 - iou_wh(wh, centroids).max(axis=1)
        chosen = rng.choice(len(wh), p=distance ** 2 / (distance ** 2).sum())
        centroids = np.vstack([centroids, wh[chosen]])
    assignment = np.full(len(wh), -1)
    for _ in range(iterations):
        new_assignment = iou_wh(wh, centroids).argmax(axis=1)
        if np.array_equal(new_assignment, assignment):
            break
        assignment = new_assignment
        for j in range(k):
            members = wh[assignment == j]
            if len(members):
                centroids[j] = np.median(members, axis=0) # robust to tile-filling boxes
    centroids = centroids[np.argsort(centroids[:, 0] * centroids[:, 1])]
    mean_best_iou = float(iou_wh(wh, centroids).max(axis=1).mean())
    return centroids, mean_best_iou
