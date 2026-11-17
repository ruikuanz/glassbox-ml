# boxes, anchors, grid encoding, loss and metrics

import numpy as np
import pytest

from glassbox.detection.anchors import kmeans_anchors
from glassbox.detection.boxes import iou_matrix, iou_wh, non_max_suppression
from glassbox.detection.encoding import (CLS, OBJ, GridSpec, decode_boxes,
                                         decode_predictions, encode_targets)
from glassbox.detection.loss import YoloLoss
from glassbox.detection.metrics import (average_precision, evaluate_detections,
                                        match_image, precision_recall_at)
from glassbox.nn import numerical_gradient, relative_error

ANCHORS = np.array([[10.0, 10.0], [30.0, 20.0], [60.0, 70.0]])


def spec(num_classes=1):
    return GridSpec(image_size=64, grid_size=4, anchors=ANCHORS, num_classes=num_classes)


def test_iou_values():
    a = np.array([[0.0, 0.0, 2.0, 2.0]])
    b = np.array([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 4.0, 4.0], [5.0, 5.0, 6.0, 6.0]])
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 1 / 7, 0.0, 0.0])


def test_iou_wh_ignores_position():
    np.testing.assert_allclose(iou_wh(np.array([[4.0, 2.0]]), np.array([[2.0, 4.0], [4.0, 2.0]]))[0],
                               [4 / 12, 1.0])


def test_nms_keeps_one_box_per_cluster():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 10, 9]], dtype=float)
    scores = np.array([0.8, 0.9, 0.3, 0.7])
    np.testing.assert_array_equal(non_max_suppression(boxes, scores, 0.5), [1, 2])
    np.testing.assert_array_equal(non_max_suppression(boxes, scores, 0.95), [1, 0, 3, 2])


def test_kmeans_recovers_planted_shapes():
    rng = np.random.default_rng(0)
    planted = np.array([[8.0, 8.0], [20.0, 60.0], [70.0, 30.0]])
    wh = np.concatenate([p * rng.uniform(0.9, 1.1, (200, 2)) for p in planted])
    anchors, mean_iou = kmeans_anchors(wh, 3, rng)
    np.testing.assert_allclose(anchors, planted[np.argsort(planted.prod(axis=1))], rtol=0.05)
    assert mean_iou > 0.85


# center (40, 24) is cell row 1, column 2; 28 x 22 is closest to the second anchor
def test_encode_places_box_in_the_right_slot():
    box = np.array([[26.0, 13.0, 54.0, 35.0]])
    y, dropped = encode_targets([box], [np.array([0])], spec())
    assert dropped == 0 and y.shape == (1, 4, 4, 3, 5) and y[..., OBJ].sum() == 1
    np.testing.assert_allclose(y[0, 1, 2, 1], [1, 0.5, 0.5, np.log(28 / 30), np.log(22 / 20)], rtol=1e-6)


def test_encode_then_decode_recovers_the_boxes():
    rng = np.random.default_rng(1)
    for num_classes in (1, 3):
        s = spec(num_classes)
        boxes = np.array([[2.0, 3.0, 11.0, 14.0], [20.0, 30.0, 52.0, 49.0], [0.0, 0.0, 64.0, 64.0]])
        classes = rng.integers(0, num_classes, 3)
        y, dropped = encode_targets([boxes], [classes], s, dtype=np.float64)
        assert dropped == 0
        t = y.copy() # logits that reproduce the label exactly
        t[..., OBJ] = np.where(y[..., OBJ] == 1, 20.0, -20.0)
        offsets = np.clip(y[..., 1:3], 1e-9, 1 - 1e-9)
        t[..., 1:3] = np.log(offsets / (1 - offsets))
        t[..., CLS:] = y[..., CLS:] * 20.0
        (d_boxes, d_scores, d_classes), = decode_predictions(t, s, score_threshold=0.5)
        assert len(d_boxes) == 3
        for box, cls in zip(boxes, classes):
            k = int(np.abs(d_boxes - box).sum(axis=1).argmin())
            np.testing.assert_allclose(d_boxes[k], box, atol=1e-5)
            assert d_classes[k] == cls


def test_encode_collision_keeps_the_larger_box():
    boxes = np.array([[36.0, 20.0, 44.0, 28.0], [34.0, 18.0, 46.0, 30.0]]) # same cell, same anchor
    y, dropped = encode_targets([boxes], [np.zeros(2, dtype=int)], spec())
    assert dropped == 1 and y[..., OBJ].sum() == 1
    assert y[0, 1, 2, 0, 3] == pytest.approx(np.log(12 / 10))


def test_decode_clips_boxes_and_survives_extreme_logits():
    t = np.full((1, 4, 4, 3, 5), 500.0)
    boxes, scores, _ = decode_boxes(t, spec())
    assert np.isfinite(boxes).all() and boxes.min() >= 0 and boxes.max() <= 64
    assert np.all(scores == 1.0)


@pytest.mark.parametrize("num_classes", [1, 3])
def test_loss_gradient(num_classes):
    rng = np.random.default_rng(num_classes)
    s = spec(num_classes)
    boxes = [np.array([[2.0, 3.0, 11.0, 14.0], [20.0, 30.0, 52.0, 49.0]]), np.zeros((0, 4)),
             np.array([[30.0, 1.0, 62.0, 20.0]])]
    classes = [rng.integers(0, num_classes, len(b)) for b in boxes]
    y, _ = encode_targets(boxes, classes, s, dtype=np.float64)
    t = rng.standard_normal(y.shape)
    loss = YoloLoss(lambda_obj=1.3, lambda_noobj=0.4, lambda_coord=5.0, lambda_class=0.7)

    J, parts = loss.forward(t, y)
    assert J == pytest.approx(sum(parts.values()))
    dt = loss.backward()
    _, dt_approx = numerical_gradient(lambda: loss.forward(t, y)[0], t)
    assert relative_error(dt.ravel(), dt_approx) < 1e-7


def test_loss_is_zero_only_for_a_perfect_prediction():
    s = spec(3)
    y, _ = encode_targets([np.array([[20.0, 30.0, 52.0, 49.0]])], [np.array([2])], s, dtype=np.float64)
    t = y.copy()
    t[..., OBJ] = np.where(y[..., OBJ] == 1, 40.0, -40.0)
    offsets = np.clip(y[..., 1:3], 1e-9, 1 - 1e-9)
    t[..., 1:3] = np.log(offsets / (1 - offsets))
    t[..., CLS:] = y[..., CLS:] * 40.0
    loss = YoloLoss()
    assert loss.forward(t, y)[0] == pytest.approx(0.0, abs=1e-9)
    t[y[..., OBJ] == 1, 3] += 1.0 # width off by a factor of e
    assert loss.forward(t, y)[0] == pytest.approx(5.0)


def test_loss_is_stable_for_huge_logits():
    y = np.zeros((1, 4, 4, 3, 5))
    t = np.full_like(y, 1e4)
    J, _ = YoloLoss().forward(t, y)
    assert np.isfinite(J)


def test_duplicate_detection_is_a_false_positive():
    gt = np.array([[0.0, 0.0, 10.0, 10.0]])
    det = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 9.0], [30.0, 30.0, 40.0, 40.0]])
    np.testing.assert_array_equal(match_image(det, np.array([0.6, 0.9, 0.8]), gt, 0.5),
                                  [False, True, False])


# TP, FP, TP against 4 ground-truth boxes: AP = 0.25 * 1 + 0.25 * 2/3
def test_average_precision_by_hand():
    precision = np.array([1.0, 0.5, 2 / 3])
    recall = np.array([0.25, 0.25, 0.5])
    assert average_precision(precision, recall) == pytest.approx(0.25 + 0.25 * 2 / 3)
    assert average_precision(np.zeros(0), np.zeros(0)) == 0.0


def test_evaluate_perfect_and_empty():
    gt = [(np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]), np.array([0, 1])),
          (np.zeros((0, 4)), np.zeros(0, dtype=int))]
    perfect = [(gt[0][0].copy(), np.array([0.9, 0.8]), np.array([0, 1])),
               (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))]
    result = evaluate_detections(perfect, gt, num_classes=2)
    assert result["mAP"] == pytest.approx(1.0) and result["best_f1"]["f1"] == pytest.approx(1.0)

    wrong_class = [(gt[0][0].copy(), np.array([0.9, 0.8]), np.array([1, 0])), perfect[1]]
    assert evaluate_detections(wrong_class, gt, num_classes=2)["mAP"] == 0.0

    nothing = [perfect[1], perfect[1]]
    assert evaluate_detections(nothing, gt, num_classes=2)["mAP"] == 0.0
    assert precision_recall_at(nothing, gt, 0.5)["fn"] == 2


def test_precision_recall_at_threshold():
    gt = [(np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]), np.array([0, 0]))]
    det = [(np.array([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0], [20.0, 20.0, 30.0, 30.0]]),
            np.array([0.9, 0.6, 0.3]), np.array([0, 0, 0]))]
    at = precision_recall_at(det, gt, score_threshold=0.5)
    assert (at["tp"], at["fp"], at["fn"]) == (1, 1, 1)
    assert at["precision"] == 0.5 and at["recall"] == 0.5
    best = evaluate_detections(det, gt, num_classes=1)["best_f1"]
    assert best["f1"] == pytest.approx(0.8) and best["threshold"] == pytest.approx(0.3)
