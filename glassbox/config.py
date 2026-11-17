# layered yaml configs: a base file plus overrides merged in order (the only module that imports yaml)

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from glassbox.data.build import BuildConfig
from glassbox.data.tiling import TilingConfig

REPO = Path(__file__).resolve().parents[1]
REPLACED = ("label_remap", "class_names") # their keys are data, so an override replaces them whole
OPTIMIZERS = ("sgd", "momentum", "rmsprop", "adam")
NUMBERS = {
    "data": ("tile_size", "overlap", "decimation", "block_tiles", "min_blob_area", "min_box_side",
             "connectivity", "background_sample_ratio", "dev_fraction", "test_fraction"),
    "model": ("head_channels", "num_anchors"),
    "loss": ("lambda_obj", "lambda_noobj", "lambda_coord", "lambda_class"),
    "training": ("num_epochs", "batch_size", "learning_rate", "momentum_beta", "rmsprop_beta",
                 "adam_beta1", "adam_beta2", "epsilon"),
    "evaluation": ("min_score", "nms_iou", "draw_count"),
}


# override wins; nested mappings merge, and a key the base lacks is an error
def merge_configs(base: dict, override: dict, where: str = "") -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if key not in merged:
            raise KeyError(f"unknown config key {where}{key}")
        if isinstance(merged[key], dict) and key not in REPLACED:
            if not isinstance(value, dict):
                raise ValueError(f"{where}{key} must be a mapping")
            merged[key] = merge_configs(merged[key], value, f"{where}{key}.")
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(base_path: str | Path, override_paths: list[str | Path] = ()) -> dict:
    config = yaml.safe_load(Path(base_path).read_text())
    for path in override_paths:
        override = yaml.safe_load(Path(path).read_text())
        if override is None:
            raise ValueError(f"override config is empty: {path}")
        config = merge_configs(config, override)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    for section, keys in NUMBERS.items():
        for key in keys:
            value = config[section][key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{section}.{key} must be a number, got {value!r} "
                                 "(yaml reads 1e-3 as text; write 1.0e-3)")
    data, model = config["data"], config["model"]
    if data["tile_size"] % data["decimation"] or (data["tile_size"] - data["overlap"]) % data["decimation"]:
        raise ValueError("data.tile_size and data.tile_size - data.overlap must be divisible by data.decimation")
    if (data["tile_size"] // data["decimation"]) % 2 ** len(model["stages"]):
        raise ValueError(f"{len(model['stages'])} pooling stages do not divide the "
                         f"{data['tile_size'] // data['decimation']} px stored tile")
    if config["training"]["optimizer"] not in OPTIMIZERS:
        raise ValueError(f"training.optimizer must be one of {OPTIMIZERS}")
    class_mapping(config)


# raw mask value -> zero-based class index, and the class names; unmapped raw values become ignore
def class_mapping(config: dict) -> tuple[dict[int, int], list[str]]:
    remap = {int(raw): int(index) for raw, index in config["classes"]["label_remap"].items()}
    names = {int(index): name for index, name in config["classes"]["class_names"].items()}
    if remap.get(0) != 0 or any(index == 0 for raw, index in remap.items() if raw != 0):
        raise ValueError("classes.label_remap must send raw 0, and only raw 0, to background (0)")
    num_classes = max(remap.values())
    if num_classes < 1 or set(remap.values()) != set(range(num_classes + 1)):
        raise ValueError("classes.label_remap must use class indices 1..C without gaps")
    if set(names) != set(range(num_classes + 1)):
        raise ValueError("classes.class_names must name background (0) and every class 1..C")
    mapping = {raw: index - 1 for raw, index in remap.items() if index > 0}
    return mapping, [names[index] for index in range(1, num_classes + 1)]


# (site name, image, mask) for every image with a matching mask, as drone_dengue pairs them
def image_mask_pairs(config: dict) -> list[tuple[str, Path, Path]]:
    paths = config["paths"]
    data_dir = Path(paths["base_path"]) / paths["data_folder"]
    image_dir, mask_dir = data_dir / paths["image_folder"], data_dir / paths["mask_folder"]
    for directory in (image_dir, mask_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"data folder not found: {directory}")
    suffix = paths["image_suffix"]
    names = sorted(f.name[:-len(suffix)] for f in image_dir.iterdir() if f.name.endswith(suffix))
    if paths["include_images"] is not None:
        missing = set(paths["include_images"]) - set(names)
        if missing:
            raise FileNotFoundError(f"paths.include_images not found in {image_dir}: {sorted(missing)}")
        names = [name for name in names if name in paths["include_images"]]
    pairs = [(name, image_dir / (name + suffix), mask_dir / (name + paths["mask_suffix"])) for name in names]
    pairs = [pair for pair in pairs if pair[2].exists()]
    if not pairs:
        raise FileNotFoundError(f"no image/mask pairs in {image_dir} and {mask_dir}")
    return pairs


def build_config(config: dict) -> BuildConfig:
    data = config["data"]
    return BuildConfig(
        tiling=TilingConfig(data["tile_size"], data["overlap"], data["decimation"], data["block_tiles"]),
        min_blob_area=data["min_blob_area"], min_box_side=data["min_box_side"],
        connectivity=data["connectivity"], background_ratio=data["background_sample_ratio"],
        dev_fraction=data["dev_fraction"], test_fraction=data["test_fraction"],
        seed=config["project"]["random_seed"],
    )


def dataset_dir(config: dict) -> Path:
    return Path(config["paths"]["output_path"]) / config["paths"]["dataset_folder"]


def run_dir(config: dict) -> Path:
    return Path(config["paths"]["output_path"]) / config["paths"]["runs_folder"] / config["project"]["run_name"]


def save_config(config: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False))


# every script takes the same two flags, as in drone_dengue
def config_from_cli(description: str) -> dict:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(REPO / "configs" / "base_config.yaml"))
    parser.add_argument("--override", action="append", default=[], help="override yaml, merged in order; repeatable")
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    print(f"config: {args.config}" + "".join(f" + {path}" for path in args.override))
    return config
