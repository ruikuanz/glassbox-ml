# training loop: mini-batch gradient descent on the detection cost

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np

from glassbox.data.dataset import DetectionDataset, Normalizer, minibatches
from glassbox.detection.encoding import GridSpec, encode_targets
from glassbox.detection.loss import YoloLoss
from glassbox.detection.model import to_grid
from glassbox.nn import Parameter, Sequential
from glassbox.optim import Optimizer

Penalty = Callable[[dict[str, Parameter]], float] # returns extra cost, adds its gradient to param.grad


class Trainer:
    def __init__(self, model: Sequential, loss: YoloLoss, optimizer: Optimizer,
                 spec: GridSpec, normalizer: Normalizer,
                 penalties: Sequence[Penalty] = (),
                 lr_schedule: Callable[[int], float] | None = None):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.spec = spec
        self.normalizer = normalizer
        self.penalties = list(penalties)
        self.lr_schedule = lr_schedule

    def train_step(self, images: np.ndarray, boxes: list[np.ndarray],
                   classes: list[np.ndarray]) -> tuple[float, dict[str, float]]:
        x = self.normalizer(images)
        y, _ = encode_targets(boxes, classes, self.spec)

        output = self.model.forward(x, training=True)
        J, parts = self.loss.forward(to_grid(output, self.spec), y)

        dt = self.loss.backward()
        self.model.backward(dt.reshape(output.shape))

        for penalty in self.penalties: # after backprop, before the update
            extra = penalty(self.model.parameters())
            parts["penalty"] = parts.get("penalty", 0.0) + extra
            J += extra

        self.optimizer.step()
        return J, parts

    # on_epoch_end(epoch, record) may add to the record; returning True stops training
    def fit(self, train: DetectionDataset, epochs: int, batch_size: int,
            rng: np.random.Generator,
            on_epoch_end: Callable[[int, dict], bool | None] | None = None,
            log: Callable[[str], None] = print) -> list[dict]:
        history = []
        for epoch in range(1, epochs + 1):
            if self.lr_schedule is not None:
                self.optimizer.lr = self.lr_schedule(epoch)
            start = time.time()
            totals: dict[str, float] = {}
            for batch in minibatches(len(train), batch_size, rng):
                J, parts = self.train_step(train.images[batch],
                                           [train.boxes[i] for i in batch],
                                           [train.classes[i] for i in batch])
                if not np.isfinite(J):
                    raise FloatingPointError(f"cost became {J} in epoch {epoch}; lower the learning rate")
                for key, value in {"cost": J, **parts}.items():
                    totals[key] = totals.get(key, 0.0) + value * len(batch) # per-example mean
            record = {"epoch": epoch, "lr": self.optimizer.lr,
                      "train": {k: v / len(train) for k, v in totals.items()},
                      "seconds": time.time() - start}
            stop = on_epoch_end(epoch, record) if on_epoch_end is not None else False
            history.append(record)
            log(f"epoch {epoch:3d}  train cost {record['train']['cost']:.4f}  "
                f"({record['seconds']:.0f}s)" + record.get("summary", ""))
            if stop:
                break
        return history
