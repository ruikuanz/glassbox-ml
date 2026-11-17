# stream and preprocessing utils

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from glassbox.data.tiling import SPLITS


class DetectionDataset:
    def __init__(self, images: np.ndarray, boxes: list[np.ndarray],
                 classes: list[np.ndarray], class_names: list[str],
                 tile_xy: np.ndarray | None = None, tile_site: np.ndarray | None = None,
                 site_names: list[str] | None = None):
        self.images = images
        self.boxes = boxes
        self.classes = classes 
        self.class_names = class_names
        self.tile_xy = tile_xy
        self.tile_site = tile_site
        self.site_names = site_names

    def __len__(self) -> int:
        return len(self.images)

    @property
    def image_size(self) -> int:
        return self.images.shape[1]

    @property
    def num_boxes(self) -> int:
        return sum(len(b) for b in self.boxes)

    def ground_truth(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return list(zip(self.boxes, self.classes))

    def subset(self, indices: np.ndarray) -> "DetectionDataset":
        return DetectionDataset(
            self.images[indices], [self.boxes[i] for i in indices],
            [self.classes[i] for i in indices], self.class_names,
            None if self.tile_xy is None else self.tile_xy[indices],
            None if self.tile_site is None else self.tile_site[indices], self.site_names,
        )

    @classmethod
    def load(cls, directory: str | Path, split: str) -> "DetectionDataset":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        chosen = np.nonzero(np.load(directory / "split.npy") == SPLITS.index(split))[0]
        images = np.load(directory / "images.npy", mmap_mode="r")[chosen]
        offsets = np.load(directory / "box_offsets.npy")
        all_boxes = np.load(directory / "boxes.npy")
        all_classes = np.load(directory / "box_classes.npy")
        boxes = [all_boxes[offsets[i]:offsets[i + 1]] for i in chosen]
        classes = [all_classes[offsets[i]:offsets[i + 1]].astype(np.int64) for i in chosen]
        tile_xy = np.load(directory / "tile_xy.npy")[chosen]
        tile_site = np.load(directory / "tile_site.npy")[chosen]
        return cls(np.ascontiguousarray(images), boxes, classes, meta["class_names"], tile_xy,
                   tile_site, meta.get("site_names"))


def write_dataset(directory: str | Path, images: np.ndarray, boxes: list[np.ndarray],
                  classes: list[np.ndarray], areas: list[np.ndarray], tile_xy: np.ndarray,
                  split: np.ndarray, meta: dict, tile_site: np.ndarray | None = None) -> None:
    directory = Path(directory)
    if tile_site is None:
        tile_site = np.zeros(len(images))
    directory.mkdir(parents=True, exist_ok=True)
    offsets = np.concatenate([[0], np.cumsum([len(b) for b in boxes])]).astype(np.int64)
    np.save(directory / "images.npy", images)
    np.save(directory / "box_offsets.npy", offsets)
    np.save(directory / "boxes.npy", np.concatenate(boxes).astype(np.float32).reshape(-1, 4))
    np.save(directory / "box_classes.npy", np.concatenate(classes).astype(np.int16))
    np.save(directory / "box_areas.npy", np.concatenate(areas).astype(np.int64))
    np.save(directory / "tile_xy.npy", np.asarray(tile_xy, dtype=np.int64))
    np.save(directory / "tile_site.npy", np.asarray(tile_site, dtype=np.int16))
    np.save(directory / "split.npy", np.asarray(split, dtype=np.uint8))
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, images: np.ndarray) -> "Normalizer":
        # Summing uint8 values as float64 in chunks avoids a float copy of
        # the whole training set.
        total = np.zeros(3)
        total_sq = np.zeros(3)
        for start in range(0, len(images), 1024):
            chunk = images[start:start + 1024].reshape(-1, 3).astype(np.float64) / 255.0
            total += chunk.sum(axis=0)
            total_sq += (chunk ** 2).sum(axis=0)
        count = images.shape[0] * images.shape[1] * images.shape[2]
        mean = total / count
        std = np.sqrt(total_sq / count - mean ** 2)
        return cls(mean, std)

    def __call__(self, images: np.ndarray, dtype=np.float32) -> np.ndarray:
        return ((images.astype(dtype) / 255.0 - self.mean) / self.std).astype(dtype, copy=False)


def minibatches(num_examples: int, batch_size: int,
                rng: np.random.Generator | None = None) -> Iterator[np.ndarray]:
    order = np.arange(num_examples) if rng is None else rng.permutation(num_examples)
    for start in range(0, num_examples, batch_size):
        yield order[start:start + batch_size]
