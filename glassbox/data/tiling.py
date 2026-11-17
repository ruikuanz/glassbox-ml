# cut the image into tiles and split them train/dev/test by block

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Rect = tuple[int, int, int, int] # x0, y0, x1, y1 in pixels, x1/y1 exclusive

SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class TilingConfig:
    tile_size: int = 512
    overlap: int = 0
    decimation: int = 4 # tiles stored at 1/decimation resolution
    block_tiles: int = 8

    @property
    def stride(self) -> int:
        return self.tile_size - self.overlap

    @property
    def block_size(self) -> int:
        return self.tile_size + (self.block_tiles - 1) * self.stride

    @property
    def output_size(self) -> int:
        return self.tile_size // self.decimation


@dataclass
class Block:
    index: tuple[int, int]
    tiles: list[tuple[int, int]] = field(default_factory=list) # top-left (x, y) of each tile

    def window(self, tile_size: int) -> tuple[int, int, int, int]:
        xs = [t[0] for t in self.tiles]
        ys = [t[1] for t in self.tiles]
        return (min(xs), min(ys),
                max(xs) - min(xs) + tile_size, max(ys) - min(ys) + tile_size)


# keep tiles that lie fully inside the image and an annotated extent
def plan_blocks(width: int, height: int, extents: list[Rect], cfg: TilingConfig) -> list[Block]:
    if cfg.tile_size % cfg.decimation or cfg.stride % cfg.decimation:
        raise ValueError("tile_size and tile_size - overlap must be divisible by decimation")
    blocks = []
    for by in range(height // cfg.block_size + 1):
        for bx in range(width // cfg.block_size + 1):
            block = Block(index=(bx, by))
            for ty in range(cfg.block_tiles):
                for tx in range(cfg.block_tiles):
                    x = bx * cfg.block_size + tx * cfg.stride
                    y = by * cfg.block_size + ty * cfg.stride
                    x1, y1 = x + cfg.tile_size, y + cfg.tile_size
                    if x1 > width or y1 > height:
                        continue
                    if any(x >= ex0 and y >= ey0 and x1 <= ex1 and y1 <= ey1
                           for ex0, ey0, ex1, ey1 in extents):
                        block.tiles.append((x, y))
            if block.tiles:
                blocks.append(block)
    return blocks


# whole blocks go to train (0), dev (1) or test (2)
def assign_splits(num_blocks: int, dev_fraction: float, test_fraction: float,
                  rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(num_blocks)
    n_dev = int(round(dev_fraction * num_blocks))
    n_test = int(round(test_fraction * num_blocks))
    split = np.zeros(num_blocks, dtype=np.uint8)
    split[order[:n_dev]] = 1
    split[order[n_dev:n_dev + n_test]] = 2
    return split
