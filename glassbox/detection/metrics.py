# precision, recall, f1 and average precision of detections against ground truth

from __future__ import annotations

import numpy as np

from glassbox.detection.boxes import iou_matrix

Detections = tuple[np.ndarray, np.ndarray, np.ndarray] # boxes, scores, classes
GroundTruth = tuple[np.ndarray, np.ndarray] # boxes, classes


# highest score first
def match_image(det_boxes: np.ndarray, det_scores: np.ndarray, gt_boxes: np.ndarray,
                iou_threshold: float) -> np.ndarray:
    is_tp = np.zeros(len(det_boxes), dtype=bool)
    if len(det_boxes) == 0 or len(gt_boxes) == 0:
        return is_tp
    ious = iou_matrix(det_boxes, gt_boxes)
    claimed = np.zeros(len(gt_boxes), dtype=bool)
    for d in np.argsort(-det_scores, kind="stable"):
        candidates = np.where(claimed, -1.0, ious[d])
        g = int(candidates.argmax())
        if candidates[g] >= iou_threshold:
            claimed[g] = True
            is_tp[d] = True
    return is_tp


def precision_recall_curve(scores: np.ndarray, is_tp: np.ndarray, num_gt: int
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(is_tp[order])
    fp = np.cumsum(~is_tp[order])
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(num_gt, 1)
    return scores[order], precision, recall


# area under the PR curve
def average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    if len(precision) == 0:
        return 0.0
    r = np.concatenate([[0.0], recall, [recall[-1]]])
    p = np.concatenate([[0.0], precision, [0.0]])
    p = np.maximum.accumulate(p[::-1])[::-1]
    steps = np.nonzero(r[1:] != r[:-1])[0]
    return float(((r[steps + 1] - r[steps]) * p[steps + 1]).sum())


# per-class curves and ap, plus mAP and the best f1 on the pooled curve
def evaluate_detections(detections: list[Detections], ground_truth: list[GroundTruth],
                        num_classes: int, iou_threshold: float = 0.5) -> dict:
    per_class = []
    pooled_scores, pooled_tp, pooled_gt = [], [], 0
    for cls in range(num_classes):
        scores, is_tp, num_gt = [], [], 0
        for (d_boxes, d_scores, d_classes), (g_boxes, g_classes) in zip(detections, ground_truth):
            d = d_classes == cls
            g = g_classes == cls
            num_gt += int(g.sum())
            scores.append(d_scores[d])
            is_tp.append(match_image(d_boxes[d], d_scores[d], g_boxes[g], iou_threshold))
        scores = np.concatenate(scores) if scores else np.zeros(0)
        is_tp = np.concatenate(is_tp) if is_tp else np.zeros(0, dtype=bool)
        thresholds, precision, recall = precision_recall_curve(scores, is_tp, num_gt)
        per_class.append({
            "num_gt": num_gt, "thresholds": thresholds, "precision": precision,
            "recall": recall, "ap": average_precision(precision, recall),
        })
        pooled_scores.append(scores)
        pooled_tp.append(is_tp)
        pooled_gt += num_gt

    with_gt = [c["ap"] for c in per_class if c["num_gt"] > 0]
    result = {
        "per_class": per_class,
        "mAP": float(np.mean(with_gt)) if with_gt else 0.0,
        "num_gt": pooled_gt,
        "best_f1": {"f1": 0.0, "threshold": 1.0, "precision": 0.0, "recall": 0.0},
    }
    thresholds, precision, recall = precision_recall_curve(
        np.concatenate(pooled_scores), np.concatenate(pooled_tp), pooled_gt)
    if len(thresholds):
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        k = int(f1.argmax())
        result["best_f1"] = {"f1": float(f1[k]), "threshold": float(thresholds[k]),
                             "precision": float(precision[k]), "recall": float(recall[k])}
    return result


# for reporting test metrics at a threshold chosen on dev
def precision_recall_at(detections: list[Detections], ground_truth: list[GroundTruth],
                        score_threshold: float, iou_threshold: float = 0.5) -> dict:
    tp = fp = num_gt = 0
    for (d_boxes, d_scores, d_classes), (g_boxes, g_classes) in zip(detections, ground_truth):
        keep = d_scores >= score_threshold
        d_boxes, d_scores, d_classes = d_boxes[keep], d_scores[keep], d_classes[keep]
        num_gt += len(g_boxes)
        for cls in np.unique(np.concatenate([d_classes, g_classes])):
            d = d_classes == cls
            is_tp = match_image(d_boxes[d], d_scores[d], g_boxes[g_classes == cls], iou_threshold)
            tp += int(is_tp.sum())
            fp += int((~is_tp).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(num_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": num_gt - tp}
