# score a run's checkpoint on each split, at the best-f1 threshold on dev unless the config sets one

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glassbox.config import config_from_cli, dataset_dir, run_dir
from glassbox.data.dataset import DetectionDataset
from glassbox.detection.loss import YoloLoss
from glassbox.detection.metrics import evaluate_detections, precision_recall_at
from glassbox.train.checkpoint import load_checkpoint
from glassbox.train.evaluate import predict


def main() -> None:
    config = config_from_cli("score a run's checkpoint on each split, at the best-f1 threshold on dev unless the config sets one")
    evaluation = config["evaluation"]
    data_dir, out_dir = dataset_dir(config), run_dir(config)
    checkpoint = out_dir / evaluation["checkpoint"]
    if not checkpoint.exists():
        raise FileNotFoundError(f"no checkpoint at {checkpoint}; run scripts/train.py with the same config")
    tag = checkpoint.stem
    model, spec, normalizer, _ = load_checkpoint(checkpoint)
    loss = YoloLoss(**config["loss"])

    def detect(dataset: DetectionDataset):
        return predict(model, dataset, spec, normalizer, score_threshold=evaluation["min_score"],
                       nms_iou=evaluation["nms_iou"], loss=loss)

    threshold = evaluation["score_threshold"]
    if threshold is None:
        dev = DetectionDataset.load(data_dir, "dev")
        best = evaluate_detections(detect(dev)[0], dev.ground_truth(), spec.num_classes, 0.5)["best_f1"]
        threshold = best["threshold"]
        print(f"score threshold {threshold:.3f}: best F1 on dev at IoU 0.5 ({best['f1']:.3f})")

    for split in evaluation["splits"]:
        dataset = DetectionDataset.load(data_dir, split)
        detections, cost = detect(dataset)
        ground_truth = dataset.ground_truth()
        report = {"checkpoint": str(checkpoint), "split": split, "tiles": len(dataset),
                  "boxes": dataset.num_boxes, "cost": cost, "score_threshold": threshold}
        print(f"\n{split}: {len(dataset)} tiles, {dataset.num_boxes} boxes, cost {cost:.4f}")
        for iou in evaluation["iou_thresholds"]:
            result = evaluate_detections(detections, ground_truth, spec.num_classes, iou)
            at = precision_recall_at(detections, ground_truth, threshold, iou)
            report[f"iou_{iou}"] = {"mAP": result["mAP"], **at, "sites": {}}
            print(f"IoU >= {iou}:  mAP {result['mAP']:.3f}   "
                  f"precision {at['precision']:.3f}  recall {at['recall']:.3f}  F1 {at['f1']:.3f}  "
                  f"(TP {at['tp']}, FP {at['fp']}, FN {at['fn']})")
            for k, site in enumerate(dataset.site_names or []):
                here = np.nonzero(dataset.tile_site == k)[0]
                if len(here) == 0:
                    continue
                d, g = [detections[i] for i in here], [ground_truth[i] for i in here]
                site_at = precision_recall_at(d, g, threshold, iou)
                site_map = evaluate_detections(d, g, spec.num_classes, iou)["mAP"]
                report[f"iou_{iou}"]["sites"][site] = {"tiles": len(here), "mAP": site_map, **site_at}
                print(f"    {site}: {len(here)} tiles  mAP {site_map:.3f}  precision {site_at['precision']:.3f}  "
                      f"recall {site_at['recall']:.3f}  F1 {site_at['f1']:.3f}")
        (out_dir / f"{tag}_{split}_metrics.json").write_text(json.dumps(report, indent=1))

        if evaluation["draw_count"] > 0:
            from glassbox.viz import save_tile_grid
            rng = np.random.default_rng(0)
            chosen = np.sort(rng.choice(len(dataset), min(evaluation["draw_count"], len(dataset)), replace=False))
            shown = []
            for i in chosen:
                boxes, scores, _ = detections[i]
                keep = scores >= threshold
                shown.append((boxes[keep], scores[keep]))
            drawing = out_dir / f"{tag}_{split}_predictions.png"
            save_tile_grid(drawing, dataset.images[chosen], [dataset.boxes[i] for i in chosen], shown)
            print(f"drew {len(chosen)} tiles to {drawing} (green: ground truth, red: predictions)")


if __name__ == "__main__":
    main()
