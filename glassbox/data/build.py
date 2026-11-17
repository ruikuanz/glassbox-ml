# turn orthomosaic and pixel-level mask into tiles with bounding boxes

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from glassbox.data.components import mask_to_boxes
from glassbox.data.raster_io import RasterReader
from glassbox.data.tiling import Block, TilingConfig, assign_splits, plan_blocks


@dataclass(frozen=True)
class BuildConfig:
    tiling: TilingConfig = TilingConfig()
    min_blob_area: int = 256
    min_box_side: float = 2.0
    connectivity: int = 8
    background_ratio: float = 0.5 # severe imbalance
    dev_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 0


# based on contiguous blobs in the mask, return boxes, classes, areas, and count of rejected blobs
def mask_tile_to_boxes(mask: np.ndarray, num_classes: int, cfg: BuildConfig
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    boxes, classes, areas, rejected = [], [], [], 0
    for c in range(num_classes):
        b, a = mask_to_boxes(mask == c + 1, cfg.connectivity)
        b = b / cfg.tiling.decimation
        side = np.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
        keep = (a >= cfg.min_blob_area) & (side >= cfg.min_box_side)
        rejected += int((~keep).sum())
        boxes.append(b[keep])
        classes.append(np.full(int(keep.sum()), c, dtype=np.int64))
        areas.append(a[keep])
    return np.concatenate(boxes), np.concatenate(classes), np.concatenate(areas), rejected


# classify into nodata / background / foreground / ambiguous, and extract boxes for foreground tiles
def label_block(block: Block, image: RasterReader, mask_source, cfg: BuildConfig) -> list[dict]:
    t = cfg.tiling
    x0, y0, w, h = block.window(t.tile_size)
    mask = mask_source.read(x0, y0, w, h)
    if image.count >= 4:
        alpha = image.read(x0, y0, w, h, bands=[4], decimation=t.decimation)[:, :, 0]
    else:
        alpha = None

    results = []
    n = t.output_size
    for x, y in block.tiles:
        tile = {"xy": (x, y), "status": "background", "boxes": None}
        if alpha is not None:
            ax, ay = (x - x0) // t.decimation, (y - y0) // t.decimation
            if alpha[ay:ay + n, ax:ax + n].min() == 0:
                tile["status"] = "nodata"
                results.append(tile)
                continue
        mask_tile = mask[y - y0:y - y0 + t.tile_size, x - x0:x - x0 + t.tile_size]
        if mask_tile.any():
            boxes, classes, areas, rejected = mask_tile_to_boxes(
                mask_tile, len(mask_source.class_names), cfg)
            if len(boxes):
                tile.update(status="foreground", boxes=boxes, classes=classes,
                            areas=areas, rejected=rejected)
            else:
                tile["status"] = "ambiguous"
        results.append(tile)
    return results


def read_block_tiles(block: Block, tiles: list[tuple[int, int]], image: RasterReader,
                     t: TilingConfig) -> np.ndarray:
    """Pass 2 for one block: RGB pixels of the chosen tiles, at reduced resolution."""
    x0, y0, w, h = block.window(t.tile_size)
    rgb = image.read(x0, y0, w, h, bands=[1, 2, 3], decimation=t.decimation)
    n = t.output_size
    out = np.empty((len(tiles), n, n, 3), dtype=np.uint8)
    for k, (x, y) in enumerate(tiles):
        ax, ay = (x - x0) // t.decimation, (y - y0) // t.decimation
        out[k] = rgb[ay:ay + n, ax:ax + n]
    return out


def build_tiles(image: RasterReader, mask_source, cfg: BuildConfig, log=print) -> dict:
    """Run both passes. Returns the arrays `write_dataset` expects, plus stats."""
    t = cfg.tiling
    rng = np.random.default_rng(cfg.seed)
    blocks = plan_blocks(image.width, image.height, mask_source.extents, t)
    block_split = assign_splits(len(blocks), cfg.dev_fraction, cfg.test_fraction, rng)
    log(f"{len(blocks)} blocks, {sum(len(b.tiles) for b in blocks)} candidate tiles")

    labelled = [label_block(block, image, mask_source, cfg) for block in blocks]
    status_counts = {"foreground": 0, "background": 0, "ambiguous": 0, "nodata": 0}
    foreground, background = [], []   # (block index, tile dict)
    for bi, tiles in enumerate(labelled):
        for tile in tiles:
            status_counts[tile["status"]] += 1
            if tile["status"] == "foreground":
                foreground.append((bi, tile))
            elif tile["status"] == "background":
                background.append((bi, tile))
    log(f"tile statuses: {status_counts}")

    chosen = list(foreground) # keep same split for each block
    for s in range(3):
        fg = [item for item in foreground if block_split[item[0]] == s]
        bg = [item for item in background if block_split[item[0]] == s]
        n_bg = min(len(bg), int(round(cfg.background_ratio * len(fg))))
        chosen += [bg[k] for k in rng.choice(len(bg), size=n_bg, replace=False)]
    chosen.sort(key=lambda item: (item[0], item[1]["xy"][1], item[1]["xy"][0]))

    by_block: dict[int, list[dict]] = {}
    for bi, tile in chosen:
        by_block.setdefault(bi, []).append(tile)
    images = [read_block_tiles(blocks[bi], [tile["xy"] for tile in tiles], image, t)
              for bi, tiles in by_block.items()]

    empty_boxes = np.zeros((0, 4), dtype=np.float32)
    empty_ints = np.zeros(0, dtype=np.int64)
    ordered = [(bi, tile) for bi, tiles in by_block.items() for tile in tiles]
    return {
        "images": np.concatenate(images) if images else np.zeros((0, t.output_size, t.output_size, 3), np.uint8),
        "boxes": [tile["boxes"] if tile["boxes"] is not None else empty_boxes for _, tile in ordered],
        "classes": [tile.get("classes", empty_ints) for _, tile in ordered],
        "areas": [tile.get("areas", empty_ints) for _, tile in ordered],
        "tile_xy": np.array([tile["xy"] for _, tile in ordered], dtype=np.int64).reshape(-1, 2),
        "split": np.array([block_split[bi] for bi, _ in ordered], dtype=np.uint8),
        "stats": {
            "blocks": len(blocks),
            "tile_statuses": status_counts,
            "blobs_rejected_as_too_small": int(sum(tile.get("rejected", 0) for _, tile in ordered)),
        },
    }
