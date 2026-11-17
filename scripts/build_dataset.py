# build one tile dataset from every image/mask pair the config names

from __future__ import annotations

import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glassbox.config import build_config, class_mapping, config_from_cli, dataset_dir, image_mask_pairs
from glassbox.data.build import build_tiles
from glassbox.data.dataset import write_dataset
from glassbox.data.mask_sources import RasterMaskSource
from glassbox.data.raster_io import RasterReader
from glassbox.data.tiling import SPLITS


def main() -> None:
    config = config_from_cli("build one tile dataset from every image/mask pair the config names")
    cfg = build_config(config)
    mapping, class_names = class_mapping(config)
    pairs = image_mask_pairs(config)
    output_dir = dataset_dir(config)

    start = time.time()
    parts, stats = [], {"sites": {}}
    for k, (name, image_path, mask_path) in enumerate(pairs):
        print(f"site {name}: {image_path}")
        image = RasterReader(image_path)
        mask_source = RasterMaskSource(mask_path, image, mapping, class_names)
        built = build_tiles(image, mask_source, replace(cfg, seed=cfg.seed + k)) # each site is split on its own
        stats["sites"][name] = built.pop("stats")
        built["tile_site"] = np.full(len(built["images"]), k)
        parts.append(built)

    images = np.concatenate([part["images"] for part in parts])
    boxes = [b for part in parts for b in part["boxes"]]
    classes = [c for part in parts for c in part["classes"]]
    areas = [a for part in parts for a in part["areas"]]
    tile_xy = np.concatenate([part["tile_xy"] for part in parts])
    tile_site = np.concatenate([part["tile_site"] for part in parts])
    split = np.concatenate([part["split"] for part in parts])
    if len(images) == 0:
        raise ValueError("no tile has a box; check classes.label_remap against the masks")

    for s, name in enumerate(SPLITS):
        in_split = split == s
        n_boxes = sum(len(b) for b, keep in zip(boxes, in_split) if keep)
        n_fg = sum(len(b) > 0 for b, keep in zip(boxes, in_split) if keep)
        stats[name] = {"tiles": int(in_split.sum()), "foreground_tiles": int(n_fg), "boxes": int(n_boxes)}
        print(f"{name:5s}: {stats[name]}")

    meta = {
        "class_names": class_names,
        "site_names": [name for name, _, _ in pairs],
        "images": [str(image_path) for _, image_path, _ in pairs],
        "config": {"data": config["data"], "classes": config["classes"]},
        "build": asdict(cfg),
        "stats": stats,
    }
    write_dataset(output_dir, images, boxes, classes, areas, tile_xy, split, meta, tile_site)
    print(f"wrote {len(images)} tiles from {len(pairs)} sites to {output_dir} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
