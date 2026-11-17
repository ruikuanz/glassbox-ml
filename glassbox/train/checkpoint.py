# save and restore a trained detector as a single .npz

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from glassbox.data.dataset import Normalizer
from glassbox.detection.encoding import GridSpec
from glassbox.detection.model import build_detector
from glassbox.nn import Sequential


def save_checkpoint(path: str | Path, model: Sequential, spec: GridSpec,
                    normalizer: Normalizer, architecture: dict, class_names: list[str]) -> None:
    config = {
        "architecture": architecture, # kwargs of build_detector
        "image_size": spec.image_size,
        "grid_size": spec.grid_size,
        "class_names": class_names,
    }
    arrays = {f"model/{name}": array for name, array in model.state_dict().items()}
    np.savez(path, config=json.dumps(config), anchors=spec.anchors,
             mean=normalizer.mean, std=normalizer.std, **arrays)


def load_checkpoint(path: str | Path) -> tuple[Sequential, GridSpec, Normalizer, list[str]]:
    with np.load(path) as data:
        config = json.loads(str(data["config"]))
        spec = GridSpec(config["image_size"], config["grid_size"], data["anchors"],
                        len(config["class_names"]))
        model = build_detector(spec, rng=np.random.default_rng(0), **config["architecture"])
        model.load_state_dict({name[len("model/"):]: data[name]
                               for name in data.files if name.startswith("model/")})
        normalizer = Normalizer(data["mean"], data["std"])
    return model, spec, normalizer, config["class_names"]
