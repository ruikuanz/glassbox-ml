# writers for tiny shapefiles

import struct

import numpy as np


# features is a list of polygons, each a list of (K, 2) rings
def write_shp(path, features):
    records = b""
    for number, rings in enumerate(features, start=1):
        points = np.concatenate(rings)
        starts = np.cumsum([0] + [len(r) for r in rings[:-1]])
        content = struct.pack("<i4d2i", 5, *points.min(0), *points.max(0), len(rings), len(points))
        content += struct.pack(f"<{len(rings)}i", *starts) + points.astype("<f8").tobytes()
        records += struct.pack(">2i", number, len(content) // 2) + content
    header = struct.pack(">i5i", 9994, 0, 0, 0, 0, 0) + struct.pack(">i", (100 + len(records)) // 2)
    header += struct.pack("<2i8d", 1000, 5, *([0.0] * 8))
    path.write_bytes(header + records)


def write_dbf(path, field, values, width=12):
    descriptor = field.encode().ljust(11, b"\x00") + b"C" + bytes(4) + bytes([width, 0]) + bytes(14)
    header_size = 32 + 32 + 1
    header = struct.pack("<4BIHH", 3, 26, 1, 1, len(values), header_size, 1 + width) + bytes(20)
    rows = b"".join(b" " + value.encode().ljust(width) for value in values)
    path.write_bytes(header + descriptor + b"\x0d" + rows + b"\x1a")
