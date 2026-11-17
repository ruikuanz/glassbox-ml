# read polygons (.shp) and attributes (.dbf) of esri shapefile

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

NULL_SHAPE = 0
POLYGON_TYPES = (5, 15, 25)

Polygon = list[np.ndarray] # rings, each (K, 2) of (x, y) in map coordinates


# rings of every feature in file order; a null shape gives an empty list
def read_shp(path: str | Path) -> list[Polygon]:
    data = Path(path).read_bytes()
    file_code, = struct.unpack(">i", data[0:4])
    if file_code != 9994:
        raise ValueError(f"{path} is not a shapefile (file code {file_code})")

    features = []
    pos = 100
    while pos < len(data):
        _, content_words = struct.unpack(">ii", data[pos:pos + 8])
        content = data[pos + 8:pos + 8 + 2 * content_words] # length is in 16-bit words
        pos += 8 + 2 * content_words

        shape_type, = struct.unpack("<i", content[0:4])
        if shape_type == NULL_SHAPE:
            features.append([])
            continue
        if shape_type not in POLYGON_TYPES:
            raise ValueError(f"{path}: unsupported shape type {shape_type} (expected polygons)")

        num_parts, num_points = struct.unpack("<ii", content[36:44])
        starts = np.frombuffer(content, dtype="<i4", count=num_parts, offset=44)
        points = np.frombuffer(content, dtype="<f8", count=2 * num_points,
                               offset=44 + 4 * num_parts).reshape(num_points, 2)
        ends = np.append(starts[1:], num_points)
        features.append([points[s:e] for s, e in zip(starts, ends)])
    return features


# rows as dicts of stripped strings
def read_dbf(path: str | Path) -> list[dict[str, str]]:
    data = Path(path).read_bytes()
    num_records, header_size, record_size = struct.unpack("<IHH", data[4:12])

    fields = []
    pos = 32
    while data[pos] != 0x0D:
        name = data[pos:pos + 11].split(b"\x00")[0].decode("latin-1")
        fields.append((name, data[pos + 16]))
        pos += 32

    rows = []
    for r in range(num_records):
        record = data[header_size + r * record_size:header_size + (r + 1) * record_size]
        row, offset = {}, 1
        for name, width in fields:
            row[name] = record[offset:offset + width].decode("latin-1").strip()
            offset += width
        rows.append(row)
    return rows
