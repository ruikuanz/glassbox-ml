# yaml config loading: layered overrides, strict keys, class mapping, site pairing

from pathlib import Path

import pytest
import yaml

from glassbox.config import (build_config, class_mapping, dataset_dir, image_mask_pairs,
                             load_config, merge_configs, run_dir)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
BASE = CONFIGS / "base_config.yaml"


def write(path, content):
    path.write_text(yaml.safe_dump(content))
    return path


def test_overrides_merge_in_order(tmp_path):
    first = write(tmp_path / "a.yaml", {"training": {"learning_rate": 0.01, "batch_size": 8}})
    second = write(tmp_path / "b.yaml", {"training": {"learning_rate": 0.1}})
    config = load_config(BASE, [first, second])
    assert config["training"]["learning_rate"] == 0.1 and config["training"]["batch_size"] == 8
    assert config["training"]["optimizer"] == "adam" # untouched keys keep the base value


def test_unknown_key_is_rejected(tmp_path):
    typo = write(tmp_path / "typo.yaml", {"training": {"learing_rate": 0.01}})
    with pytest.raises(KeyError, match="training.learing_rate"):
        load_config(BASE, [typo])


def test_label_remap_is_replaced_not_merged():
    base = {"classes": {"label_remap": {0: 0, 1: 1, 2: 1}, "class_names": {0: "background", 1: "trash"}}}
    merged = merge_configs(base, {"classes": {"label_remap": {0: 0, 7: 1}}})
    assert merged["classes"]["label_remap"] == {0: 0, 7: 1}
    assert class_mapping(merged) == ({7: 0}, ["trash"])


def test_invalid_values_are_rejected(tmp_path):
    cases = [
        ({"training": {"learning_rate": "1e-3"}}, "must be a number"), # yaml reads 1e-3 as text
        ({"training": {"optimizer": "adagrad"}}, "training.optimizer"),
        ({"model": {"stages": [8, 8, 8, 8, 8, 8, 8, 8]}}, "pooling stages"),
        ({"classes": {"label_remap": {0: 0, 1: 2}}}, "without gaps"),
        ({"classes": {"label_remap": {0: 0, 1: 0}}}, "only raw 0"),
    ]
    for k, (override, message) in enumerate(cases):
        with pytest.raises(ValueError, match=message):
            load_config(BASE, [write(tmp_path / f"{k}.yaml", override)])
    with pytest.raises(ValueError, match="empty"):
        (tmp_path / "empty.yaml").write_text("# nothing\n")
        load_config(BASE, [tmp_path / "empty.yaml"])


def test_every_shipped_config_loads():
    overrides = sorted(CONFIGS.glob("sherlock/*.yaml")) + sorted(CONFIGS.glob("experiments/*.yaml"))
    assert len(overrides) >= 4
    for override in overrides:
        load_config(BASE, [override])
    config = load_config(BASE, [CONFIGS / "sherlock" / "sherlock.yaml", CONFIGS / "experiments" / "multiclass.yaml"])
    mapping, names = class_mapping(config)
    assert len(names) == 9 and mapping[9] == 8 and 10 not in mapping # tire stays ignored
    assert dataset_dir(config).name == "multiclass" and run_dir(config).name == "multiclass"
    assert build_config(config).tiling.output_size == 128


def test_image_mask_pairs_follow_the_reference_layout(tmp_path):
    images, masks = tmp_path / "data" / "images", tmp_path / "data" / "masks"
    images.mkdir(parents=True), masks.mkdir(parents=True)
    for name in ("Kisumu_Jan2023", "Ukunda_Jan2023", "Unlabelled"):
        (images / f"{name}_image.tif").touch()
    for name in ("Kisumu_Jan2023_mask.tif", "Ukunda_Jan2023_mask.tif", "Ukunda_Jan2023_mask_tires.tif"):
        (masks / name).touch()
    config = load_config(BASE, [write(tmp_path / "p.yaml", {"paths": {"base_path": str(tmp_path)}})])
    assert [name for name, _, _ in image_mask_pairs(config)] == ["Kisumu_Jan2023", "Ukunda_Jan2023"]
    config["paths"]["include_images"] = ["Ukunda_Jan2023"]
    (name, image, mask), = image_mask_pairs(config)
    assert image == images / "Ukunda_Jan2023_image.tif" and mask == masks / "Ukunda_Jan2023_mask.tif"
    config["paths"]["include_images"] = ["Nowhere"]
    with pytest.raises(FileNotFoundError, match="Nowhere"):
        image_mask_pairs(config)
