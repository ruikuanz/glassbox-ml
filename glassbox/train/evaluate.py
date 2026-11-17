# run a trained detector over a dataset

from __future__ import annotations

from glassbox.data.dataset import DetectionDataset, Normalizer, minibatches
from glassbox.detection.encoding import GridSpec, decode_predictions, encode_targets
from glassbox.detection.loss import YoloLoss
from glassbox.detection.metrics import Detections
from glassbox.detection.model import to_grid
from glassbox.nn import Sequential


# low score threshold keeps the whole ranked list for average precision
def predict(model: Sequential, dataset: DetectionDataset, spec: GridSpec,
            normalizer: Normalizer, batch_size: int = 128, score_threshold: float = 0.05,
            nms_iou: float = 0.5, loss: YoloLoss | None = None
            ) -> tuple[list[Detections], float | None]:
    detections = []
    total_cost = 0.0
    for batch in minibatches(len(dataset), batch_size):
        t = to_grid(model.forward(normalizer(dataset.images[batch]), training=False), spec)
        detections += decode_predictions(t, spec, score_threshold, nms_iou)
        if loss is not None:
            y, _ = encode_targets([dataset.boxes[i] for i in batch],
                                  [dataset.classes[i] for i in batch], spec)
            total_cost += loss.forward(t, y)[0] * len(batch)
    return detections, (total_cost / len(dataset) if loss is not None else None)
