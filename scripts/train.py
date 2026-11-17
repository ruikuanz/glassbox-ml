# train the detector; normalization, anchors and weights all come from the train split

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glassbox.config import config_from_cli, dataset_dir, run_dir, save_config
from glassbox.data.dataset import DetectionDataset, Normalizer
from glassbox.detection.anchors import kmeans_anchors
from glassbox.detection.encoding import GridSpec, encode_targets
from glassbox.detection.loss import YoloLoss
from glassbox.detection.metrics import evaluate_detections
from glassbox.detection.model import build_detector
from glassbox.nn import Parameter
from glassbox.optim import SGD, Adam, Momentum, Optimizer, RMSprop
from glassbox.train.checkpoint import save_checkpoint
from glassbox.train.evaluate import predict
from glassbox.train.trainer import Trainer


def make_optimizer(training: dict, params: list[Parameter]) -> Optimizer:
    lr, eps = training["learning_rate"], training["epsilon"]
    if training["optimizer"] == "sgd":
        return SGD(params, lr)
    if training["optimizer"] == "momentum":
        return Momentum(params, lr, training["momentum_beta"])
    if training["optimizer"] == "rmsprop":
        return RMSprop(params, lr, training["rmsprop_beta"], eps)
    return Adam(params, lr, training["adam_beta1"], training["adam_beta2"], eps)


def main() -> None:
    config = config_from_cli("train the detector; normalization, anchors and weights all come from the train split")
    training, model_cfg, evaluation = config["training"], config["model"], config["evaluation"]
    data_dir, out_dir = dataset_dir(config), run_dir(config)
    if not (data_dir / "meta.json").exists():
        raise FileNotFoundError(f"no dataset at {data_dir}; run scripts/build_dataset.py with the same config")
    built_with = json.loads((data_dir / "meta.json").read_text())["config"]
    wanted = json.loads(json.dumps({"data": config["data"], "classes": config["classes"]}))
    if built_with != wanted:
        raise ValueError(f"{data_dir} was built with different data/classes settings; "
                         "rebuild it or point paths.dataset_folder elsewhere")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir / "config.yaml")
    rng = np.random.default_rng(config["project"]["random_seed"])

    train = DetectionDataset.load(data_dir, "train")
    dev = DetectionDataset.load(data_dir, "dev")
    if training["limit_train"] is not None:
        train = train.subset(np.arange(min(training["limit_train"], len(train))))
    # fixed train sample the size of dev, to compare the two each epoch
    train_sample = train.subset(np.sort(rng.choice(len(train), min(len(train), len(dev)), replace=False)))
    print(f"train: {len(train)} tiles, {train.num_boxes} boxes   dev: {len(dev)} tiles, {dev.num_boxes} boxes")

    normalizer = Normalizer.fit(train.images)
    all_boxes = np.concatenate(train.boxes)
    anchors, anchor_iou = kmeans_anchors(all_boxes[:, 2:4] - all_boxes[:, 0:2], model_cfg["num_anchors"], rng)
    spec = GridSpec(image_size=train.image_size,
                    grid_size=train.image_size // 2 ** len(model_cfg["stages"]),
                    anchors=anchors.astype(np.float32), num_classes=len(train.class_names))
    _, dropped = encode_targets(train.boxes, train.classes, spec)
    print(f"grid {spec.grid_size}x{spec.grid_size}, anchors (w, h): {anchors.round(1).tolist()}")
    print(f"mean IoU of a box with its best anchor: {anchor_iou:.3f}; "
          f"boxes lost to slot collisions: {dropped} of {train.num_boxes}")

    architecture = {"stages": model_cfg["stages"], "head_channels": model_cfg["head_channels"]}
    model = build_detector(spec, rng=rng, **architecture)
    print(f"model: {model.num_parameters():,} parameters")
    loss = YoloLoss(**config["loss"])
    optimizer = make_optimizer(training, list(model.parameters().values()))
    trainer = Trainer(model, loss, optimizer, spec, normalizer)

    def score(dataset: DetectionDataset) -> dict:
        detections, cost = predict(model, dataset, spec, normalizer, score_threshold=evaluation["min_score"],
                                   nms_iou=evaluation["nms_iou"], loss=loss)
        result = evaluate_detections(detections, dataset.ground_truth(), spec.num_classes, 0.5)
        return {"cost": cost, "mAP50": result["mAP"], "best_f1": result["best_f1"]}

    best = {"mAP50": -1.0}

    def on_epoch_end(epoch: int, record: dict) -> None:
        record["train_sample"] = score(train_sample)
        record["dev"] = score(dev)
        record["summary"] = (f"  dev cost {record['dev']['cost']:.4f}"
                             f"  mAP@0.5 train {record['train_sample']['mAP50']:.3f}"
                             f" / dev {record['dev']['mAP50']:.3f}")
        save_checkpoint(out_dir / "last.npz", model, spec, normalizer, architecture, train.class_names)
        if record["dev"]["mAP50"] > best["mAP50"]:
            best.update(mAP50=record["dev"]["mAP50"], epoch=epoch)
            save_checkpoint(out_dir / "best.npz", model, spec, normalizer, architecture, train.class_names)
        (out_dir / "history.json").write_text(json.dumps(history_so_far + [record], indent=1))
        history_so_far.append(record)

    history_so_far: list[dict] = []
    trainer.fit(train, training["num_epochs"], training["batch_size"], rng, on_epoch_end)
    print(f"best dev mAP@0.5 {best['mAP50']:.3f} at epoch {best['epoch']}; checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
