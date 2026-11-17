# classes for masks, rasterizing from polygon shapefiles

from __future__ import annotations

from pathlib import Path

import numpy as np

from glassbox.data.raster_io import RasterReader
from glassbox.data.rasterize import rasterize_polygon
from glassbox.data.shapefile import read_dbf, read_shp

IGNORE = 255
Rect = tuple[int, int, int, int]  # x0, y0, x1, y1 in pixels, x1/y1 exclusive


class PolygonMaskSource:

    def __init__(self, layers: list[dict], image: RasterReader, class_name: str = "object"):
        self.class_names = [class_name]
        self.extents: list[Rect] = []
        self._rings: list[list[np.ndarray]] = [] 
        self._values: list[int] = []
        bboxes = []
        for layer in layers:
            polygons = read_shp(layer["shp"])
            values = np.ones(len(polygons), dtype=np.int64)
            if layer.get("label_field"):
                rows = read_dbf(Path(layer["shp"]).with_suffix(".dbf"))
                if len(rows) != len(polygons):
                    raise ValueError(f"{layer['shp']}: {len(polygons)} shapes but {len(rows)} attribute rows")
                ignore = {name.lower() for name in layer.get("ignore_labels", [])}
                for k, row in enumerate(rows):
                    if row[layer["label_field"]].lower() in ignore:
                        values[k] = IGNORE
            layer_boxes = []
            for rings, value in zip(polygons, values):
                if not rings:
                    continue
                rings = [image.map_to_pixel(ring) for ring in rings]
                points = np.concatenate(rings)
                self._rings.append(rings)
                self._values.append(int(value))
                layer_boxes.append([*points.min(axis=0), *points.max(axis=0)])
            layer_boxes = np.array(layer_boxes)
            bboxes.append(layer_boxes)
            extent = (
                max(int(np.ceil(layer_boxes[:, 0].min())), 0),
                max(int(np.ceil(layer_boxes[:, 1].min())), 0),
                min(int(np.floor(layer_boxes[:, 2].max())), image.width),
                min(int(np.floor(layer_boxes[:, 3].max())), image.height),
            )
            if extent[0] >= extent[2] or extent[1] >= extent[3]:
                raise ValueError(
                    f"{layer['shp']} does not overlap the image; "
                    "its polygons must be in the image's coordinate system")
            self.extents.append(extent)
        self._bboxes = np.concatenate(bboxes)
        self._values_array = np.array(self._values)

    def read(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        b = self._bboxes
        touching = np.nonzero((b[:, 0] < x + w) & (b[:, 2] > x) & (b[:, 1] < y + h) & (b[:, 3] > y))[0]
        offset = np.array([x, y], dtype=np.float64)
        # Foreground first, IGNORE second: where the two overlap, ignore wins.
        for value in (1, IGNORE):
            for k in touching[self._values_array[touching] == value]:
                rasterize_polygon([ring - offset for ring in self._rings[k]], mask, value)
        return mask


# single band GEOTIFF
class RasterMaskSource:

    def __init__(self, mask_path: str | Path, image: RasterReader,
                 class_mapping: dict[int, int], class_names: list[str]):
        self.mask = RasterReader(mask_path)
        if (self.mask.width, self.mask.height) != (image.width, image.height):
            raise ValueError(
                f"mask is {self.mask.width}x{self.mask.height} but image is {image.width}x{image.height}")
        self.class_names = class_names
        self.extents: list[Rect] = [(0, 0, image.width, image.height)]
        self._lookup = np.full(256, IGNORE, dtype=np.uint8)
        self._lookup[0] = 0
        for raw, index in class_mapping.items():
            if not 0 <= int(index) < len(class_names):
                raise ValueError(f"class index {index} has no name in {class_names}")
            self._lookup[int(raw)] = int(index) + 1

    def read(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        return self._lookup[self.mask.read(x, y, w, h, bands=[1])[:, :, 0]]
