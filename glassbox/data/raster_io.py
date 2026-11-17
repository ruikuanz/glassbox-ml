# windowed reads from GeoTIFF

from __future__ import annotations

from pathlib import Path

import numpy as np


class RasterReader:
    def __init__(self, path: str | Path):
        import rasterio

        self.path = str(path)
        self._dataset = None
        with rasterio.open(self.path) as ds:
            t = ds.transform
            if t.b != 0 or t.d != 0:
                raise ValueError(f"{path}: rotated rasters are not supported")
            self.width, self.height, self.count = ds.width, ds.height, ds.count
            self.origin_x, self.origin_y = t.c, t.f # map coordinates of the top-left corner
            self.pixel_width, self.pixel_height = t.a, -t.e

    # (K, 2) map coordinates -> pixel coordinates from the top left corner
    def map_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        px = (xy[:, 0] - self.origin_x) / self.pixel_width
        py = (self.origin_y - xy[:, 1]) / self.pixel_height
        return np.stack([px, py], axis=1)

    # (H, W, bands) window
    def read(self, x: int, y: int, w: int, h: int, bands: list[int],
             decimation: int = 1) -> np.ndarray:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import Window

        if x < 0 or y < 0 or x + w > self.width or y + h > self.height:
            raise ValueError(f"window ({x}, {y}, {w}, {h}) leaves the {self.width}x{self.height} raster")
        if w % decimation or h % decimation:
            raise ValueError(f"window {w}x{h} is not divisible by decimation {decimation}")
        if self._dataset is None:
            self._dataset = rasterio.open(self.path)
        data = self._dataset.read(
            bands, window=Window(x, y, w, h),
            out_shape=(len(bands), h // decimation, w // decimation),
            resampling=Resampling.average,
        )
        return np.transpose(data, (1, 2, 0))
