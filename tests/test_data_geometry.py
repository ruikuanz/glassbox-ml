# label geometry checked against scipy, gdal and geopandas

import numpy as np
import pytest

from glassbox.data.components import label_image, mask_to_boxes
from glassbox.data.rasterize import rasterize_polygon
from glassbox.data.shapefile import read_dbf, read_shp
from helpers import write_dbf, write_shp


def test_mask_to_boxes_is_the_tight_box():
    mask = np.zeros((20, 30), dtype=bool)
    mask[10:13, 3:8] = True
    mask[2, 20] = True
    boxes, areas = mask_to_boxes(mask)
    order = np.argsort(boxes[:, 0])
    np.testing.assert_array_equal(boxes[order], [[3, 10, 8, 13], [20, 2, 21, 3]])
    np.testing.assert_array_equal(areas[order], [15, 1])


def test_diagonal_pixels_join_only_with_8_connectivity():
    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    assert len(mask_to_boxes(mask, connectivity=8)[0]) == 1
    assert len(mask_to_boxes(mask, connectivity=4)[0]) == 2


# the arms are separate runs until the bottom row joins them
def test_u_shape_is_one_blob():
    mask = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    boxes, areas = mask_to_boxes(mask, connectivity=4)
    np.testing.assert_array_equal(boxes, [[0, 0, 3, 3]])
    assert areas[0] == 7


def test_empty_mask():
    boxes, areas = mask_to_boxes(np.zeros((5, 5), dtype=bool))
    assert boxes.shape == (0, 4) and areas.shape == (0,)


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("density", [0.2, 0.45, 0.6])
def test_components_match_scipy(connectivity, density):
    ndimage = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(int(density * 100) + connectivity)
    mask = rng.random((64, 80)) < density
    structure = np.ones((3, 3)) if connectivity == 8 else None
    expected, n_expected = ndimage.label(mask, structure=structure)

    labels, n = label_image(mask, connectivity)
    assert n == n_expected
    pairs = np.unique(np.stack([labels[mask], expected[mask]]), axis=1) # same partition, any numbering
    assert pairs.shape[1] == n

    boxes, areas = mask_to_boxes(mask, connectivity)
    ours = sorted(map(tuple, np.column_stack([boxes, areas]).tolist()))
    theirs = []
    for k, (rows, cols) in enumerate(ndimage.find_objects(expected), start=1):
        theirs.append((cols.start, rows.start, cols.stop, rows.stop, int((expected == k).sum())))
    assert ours == sorted(map(lambda t: tuple(map(float, t)), theirs))


# covers the centers of columns 1..3 and rows 2..3
def test_rasterize_square_uses_pixel_centers():
    out = np.zeros((6, 6), dtype=np.uint8)
    rasterize_polygon([np.array([[1.2, 1.6], [3.9, 1.6], [3.9, 3.7], [1.2, 3.7]])], out, 7)
    expected = np.zeros((6, 6), dtype=np.uint8)
    expected[2:4, 1:4] = 7
    np.testing.assert_array_equal(out, expected)


def test_rasterize_hole_is_left_empty():
    out = np.zeros((10, 10), dtype=np.uint8)
    outer = np.array([[1, 1], [9, 1], [9, 9], [1, 9]], dtype=float)
    hole = np.array([[4, 4], [4, 6], [6, 6], [6, 4]], dtype=float)
    rasterize_polygon([outer, hole], out, 1)
    assert out[1:9, 1:9].sum() == 64 - 4
    assert out[4:6, 4:6].sum() == 0


def test_rasterize_polygon_partly_outside_the_array():
    out = np.zeros((4, 4), dtype=np.uint8)
    rasterize_polygon([np.array([[-5, -5], [2, -5], [2, 2], [-5, 2]], dtype=float)], out, 1)
    expected = np.zeros((4, 4), dtype=np.uint8)
    expected[:2, :2] = 1
    np.testing.assert_array_equal(out, expected)


def test_rasterize_matches_gdal():
    features = pytest.importorskip("rasterio.features")
    rng = np.random.default_rng(3)
    for _ in range(25): # random star-shaped polygons, possibly leaving the array
        n = int(rng.integers(3, 30))
        angles = np.sort(rng.uniform(0, 2 * np.pi, n))
        radii = rng.uniform(3, 30, n)
        center = rng.uniform(5, 45, 2)
        ring = center + np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        ours = np.zeros((50, 60), dtype=np.uint8)
        rasterize_polygon([ring], ours, 1)
        geometry = {"type": "Polygon", "coordinates": [ring.tolist() + [ring[0].tolist()]]}
        theirs = features.rasterize([(geometry, 1)], out_shape=(50, 60), dtype="uint8")
        np.testing.assert_array_equal(ours, theirs)


def test_read_shp_round_trip(tmp_path):
    features = [
        [np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [0.0, 0.0]])],
        [np.array([[5.5, 5.0], [5.0, 9.0], [9.0, 9.0], [5.5, 5.0]]),
         np.array([[6.0, 7.0], [7.0, 8.0], [6.0, 8.0], [6.0, 7.0]])],
    ]
    write_shp(tmp_path / "a.shp", features)
    read = read_shp(tmp_path / "a.shp")
    assert [len(rings) for rings in read] == [1, 2]
    for got, want in zip(read, features):
        for g, w in zip(got, want):
            np.testing.assert_array_equal(g, w)


def test_read_shp_and_dbf_match_geopandas(tmp_path):
    gpd = pytest.importorskip("geopandas")
    geometry = pytest.importorskip("shapely.geometry")
    polygons = [
        geometry.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)], holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]]),
        geometry.MultiPolygon([geometry.box(10, 10, 12, 13), geometry.box(20, 20, 21, 21)]),
        geometry.Polygon([(5, 5), (6, 9), (9, 6)]),
    ]
    frame = gpd.GeoDataFrame({"Abbr": ["Scat g", None, "Mimic"], "Id": [1, 2, 3]}, geometry=polygons)
    frame.to_file(tmp_path / "layer.shp")

    read = read_shp(tmp_path / "layer.shp")
    assert [len(rings) for rings in read] == [2, 2, 1]
    for rings, polygon in zip(read, polygons): # holes wind the other way
        signed = sum(0.5 * np.sum(r[:-1, 0] * r[1:, 1] - r[1:, 0] * r[:-1, 1]) for r in rings)
        assert abs(signed) == pytest.approx(polygon.area)

    rows = read_dbf(tmp_path / "layer.dbf")
    assert [row["Abbr"] for row in rows] == ["Scat g", "", "Mimic"]
    assert [int(row["Id"]) for row in rows] == [1, 2, 3]


def test_read_dbf_round_trip(tmp_path):
    write_dbf(tmp_path / "a.dbf", "Abbr", ["Scat g", "", "Mimic"])
    assert [row["Abbr"] for row in read_dbf(tmp_path / "a.dbf")] == ["Scat g", "", "Mimic"]
