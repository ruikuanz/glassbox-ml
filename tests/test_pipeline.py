# data pipeline and training loop end to end on a tiny hand-made raster

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from glassbox.data.build import BuildConfig, build_tiles
from glassbox.data.dataset import DetectionDataset, Normalizer, minibatches, write_dataset
from glassbox.data.mask_sources import IGNORE, PolygonMaskSource, RasterMaskSource
from glassbox.data.raster_io import RasterReader
from glassbox.data.tiling import TilingConfig, assign_splits, plan_blocks
from glassbox.detection.encoding import GridSpec
from glassbox.detection.loss import YoloLoss
from glassbox.detection.metrics import evaluate_detections
from glassbox.detection.model import build_detector
from glassbox.optim import Adam
from glassbox.train.checkpoint import load_checkpoint, save_checkpoint
from glassbox.train.evaluate import predict
from glassbox.train.trainer import Trainer
from helpers import write_dbf, write_shp

rasterio = pytest.importorskip("rasterio")
REPO = Path(__file__).resolve().parents[1]

# 64 px tiles at half resolution in 2 x 2 blocks: 8 x 6 tiles in 4 x 3 blocks
WIDTH, HEIGHT = 512, 384
TILING = TilingConfig(tile_size=64, overlap=0, decimation=2, block_tiles=2)
CONFIG = BuildConfig(tiling=TILING, min_blob_area=16, min_box_side=2.0,
                     background_ratio=1.0, dev_fraction=0.25, test_fraction=0.25, seed=0)


# rgba image whose right-hand column of tiles is nodata, and a mask
def make_rasters(directory):
    from rasterio.transform import from_origin
    transform = from_origin(1000.0, 2000.0, 0.5, 0.5) # top-left (1000, 2000), 0.5 m pixels
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    # constant over 2 x 2 blocks, so 2x averaging is exact
    rgb = np.stack([(yy // 2 * 3 + xx // 2 * 5) % 251, (xx // 2 * 7) % 251, (yy // 2 * 11) % 251])
    alpha = np.full((HEIGHT, WIDTH), 255)
    alpha[:, 448:] = 0

    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[20:50, 10:40] = 3
    mask[5:25, 50:80] = 3 # cut by the tile edge at x = 64
    mask[10:12, 140:142] = 3 # too small to box
    mask[10:40, 200:230] = 7 # class missing from the mapping
    mask[70:100, 5:25] = 3 # touches the next object, of another class
    mask[70:100, 25:45] = 5
    for ty in range(2, 6): # one object in every other remaining tile
        for tx in range(7):
            if (tx + ty) % 2 == 0:
                mask[ty * 64 + 8:ty * 64 + 40, tx * 64 + 12:tx * 64 + 36 + 2 * tx] = 3

    profile = dict(driver="GTiff", width=WIDTH, height=HEIGHT, dtype="uint8", transform=transform)
    with rasterio.open(directory / "site_image.tif", "w", count=4, **profile) as dst:
        dst.write(np.concatenate([rgb, alpha[None]]).astype(np.uint8))
    with rasterio.open(directory / "site_mask.tif", "w", count=1, **profile) as dst:
        dst.write(mask, 1)
    return np.transpose(rgb, (1, 2, 0)).astype(np.uint8), mask


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    directory = tmp_path_factory.mktemp("site")
    rgb, mask = make_rasters(directory)
    image = RasterReader(directory / "site_image.tif")
    source = RasterMaskSource(directory / "site_mask.tif", image, {3: 0, 5: 1}, ["trash", "tire"])
    return {"dir": directory, "rgb": rgb, "mask": mask, "image": image, "source": source,
            "built": build_tiles(image, source, CONFIG, log=lambda _: None)}


def test_plan_blocks_keeps_only_fully_annotated_tiles():
    blocks = plan_blocks(WIDTH, HEIGHT, [(0, 0, WIDTH, HEIGHT)], TILING)
    assert len(blocks) == 12 and sum(len(b.tiles) for b in blocks) == 48
    partial = plan_blocks(WIDTH, HEIGHT, [(0, 0, 200, HEIGHT)], TILING) # three columns of tiles
    assert sorted({x for b in partial for x, _ in b.tiles}) == [0, 64, 128]


def test_overlapping_tiles_never_cross_into_another_block():
    cfg = TilingConfig(tile_size=64, overlap=16, decimation=2, block_tiles=2)
    blocks = plan_blocks(WIDTH, HEIGHT, [(0, 0, WIDTH, HEIGHT)], cfg)
    assert cfg.block_size == 112
    for block in blocks:
        x0, y0 = block.index[0] * cfg.block_size, block.index[1] * cfg.block_size
        for x, y in block.tiles:
            assert x0 <= x and x + 64 <= x0 + cfg.block_size
            assert y0 <= y and y + 64 <= y0 + cfg.block_size


def test_assign_splits_fractions():
    split = assign_splits(20, 0.1, 0.2, np.random.default_rng(0))
    assert np.bincount(split, minlength=3).tolist() == [14, 2, 4]


def test_raster_mask_source_remaps_classes(site):
    window = site["source"].read(0, 0, 256, 128)
    raw = site["mask"][0:128, 0:256]
    np.testing.assert_array_equal(window == 1, raw == 3)
    np.testing.assert_array_equal(window == 2, raw == 5)
    np.testing.assert_array_equal(window == IGNORE, raw == 7)


def test_polygon_mask_source(site, tmp_path):
    def square(c0, r0, c1, r1): # map x = 1000 + column / 2, y = 2000 - row / 2
        return [np.array([[1000 + c0 / 2, 2000 - r0 / 2], [1000 + c1 / 2, 2000 - r0 / 2],
                          [1000 + c1 / 2, 2000 - r1 / 2], [1000 + c0 / 2, 2000 - r1 / 2]])]
    write_shp(tmp_path / "layer.shp", [square(10, 20, 40, 50), square(100, 30, 130, 60), square(30, 40, 60, 70)])
    write_dbf(tmp_path / "layer.dbf", "Abbr", ["Scat g", "", "MIMIC"])
    source = PolygonMaskSource(
        [{"shp": str(tmp_path / "layer.shp"), "label_field": "Abbr", "ignore_labels": ["mimic"]}],
        site["image"], "trash")
    assert source.extents == [(10, 20, 130, 70)] and source.class_names == ["trash"]

    expected = np.zeros((100, 150), dtype=np.uint8)
    expected[20:50, 10:40] = 1
    expected[30:60, 100:130] = 1
    expected[40:70, 30:60] = IGNORE # ignore wins on overlap
    np.testing.assert_array_equal(source.read(0, 0, 150, 100), expected)
    np.testing.assert_array_equal(source.read(25, 35, 60, 40), expected[35:75, 25:85])

    write_shp(tmp_path / "far.shp", [[np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])]])
    with pytest.raises(ValueError, match="does not overlap"):
        PolygonMaskSource([{"shp": str(tmp_path / "far.shp")}], site["image"])


def tile_index(built, x, y):
    return int(np.nonzero((built["tile_xy"] == (x, y)).all(axis=1))[0][0])


# 6 nodata tiles at x = 448, two ambiguous in the top row, 3 + 14 foreground
def test_build_tile_statuses(site):
    assert site["built"]["stats"]["tile_statuses"] == {
        "foreground": 17, "background": 23, "ambiguous": 2, "nodata": 6}
    xy = site["built"]["tile_xy"].tolist()
    assert [128, 0] not in xy and [192, 0] not in xy and all(x < 448 for x, _ in xy)


# full-resolution extents / 2; the second object is cut at x = 64
def test_build_boxes_are_tight_and_in_stored_pixels(site):
    built = site["built"]
    k = tile_index(built, 0, 0)
    np.testing.assert_array_equal(sorted(built["boxes"][k].tolist()),
                                  [[5, 10, 20, 25], [25, 2.5, 32, 12.5]])
    np.testing.assert_array_equal(sorted(built["areas"][k].tolist()), [280, 900])
    np.testing.assert_array_equal(built["boxes"][tile_index(built, 64, 0)], [[0, 2.5, 8, 12.5]])


def test_build_separates_touching_objects_of_different_classes(site):
    built = site["built"]
    k = tile_index(built, 0, 64)
    order = np.argsort(built["classes"][k])
    np.testing.assert_array_equal(built["classes"][k][order], [0, 1])
    np.testing.assert_array_equal(built["boxes"][k][order], [[2.5, 3, 12.5, 18], [12.5, 3, 22.5, 18]])


def test_build_images_are_the_decimated_source(site):
    built = site["built"]
    half = site["rgb"][::2, ::2]
    for (x, y), image in zip(built["tile_xy"], built["images"]):
        np.testing.assert_array_equal(image, half[y // 2:y // 2 + 32, x // 2:x // 2 + 32])


def test_build_splits_by_block_and_balances_background(site):
    built = site["built"]
    block_of = built["tile_xy"] // TILING.block_size
    for block in np.unique(block_of, axis=0):
        assert len(set(built["split"][(block_of == block).all(axis=1)])) == 1
    foreground = np.array([len(b) > 0 for b in built["boxes"]])
    assert foreground.sum() == 17
    for s in range(3):
        n_fg = int((foreground & (built["split"] == s)).sum())
        n_bg = int((~foreground & (built["split"] == s)).sum())
        assert n_bg <= n_fg # background_ratio = 1.0


def test_build_with_no_boxes_returns_an_empty_dataset(site):
    source = RasterMaskSource(site["dir"] / "site_mask.tif", site["image"], {9: 0}, ["absent"])
    built = build_tiles(site["image"], source, CONFIG, log=lambda _: None)
    assert built["images"].shape == (0, 32, 32, 3) and built["tile_xy"].shape == (0, 2)


def test_dataset_round_trip(site, tmp_path):
    built = site["built"]
    write_dataset(tmp_path, built["images"], built["boxes"], built["classes"], built["areas"],
                  built["tile_xy"], built["split"], {"class_names": ["trash", "tire"]})
    total = 0
    for s, name in enumerate(("train", "dev", "test")):
        dataset = DetectionDataset.load(tmp_path, name)
        chosen = np.nonzero(built["split"] == s)[0]
        total += len(dataset)
        np.testing.assert_array_equal(dataset.images, built["images"][chosen])
        for got, k in zip(dataset.ground_truth(), chosen):
            np.testing.assert_array_equal(got[0], built["boxes"][k])
            np.testing.assert_array_equal(got[1], built["classes"][k])
    assert total == len(built["images"])


def test_normalizer_and_minibatches():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (10, 8, 8, 3), dtype=np.uint8)
    x = Normalizer.fit(images)(images)
    assert x.dtype == np.float32
    np.testing.assert_allclose(x.mean(axis=(0, 1, 2)), 0, atol=1e-5)
    np.testing.assert_allclose(x.std(axis=(0, 1, 2)), 1, atol=1e-5)
    batches = list(minibatches(10, 4, rng))
    assert [len(b) for b in batches] == [4, 4, 2]
    assert sorted(np.concatenate(batches).tolist()) == list(range(10))


def tiny_training_setup(site):
    built = site["built"]
    keep = np.nonzero(np.array([len(b) > 0 for b in built["boxes"]]))[0][:8]
    data = DetectionDataset(built["images"][keep], [built["boxes"][k] for k in keep],
                            [built["classes"][k] for k in keep], ["trash", "tire"])
    spec = GridSpec(image_size=32, grid_size=8,
                    anchors=np.array([[10.0, 15.0], [14.0, 16.0]], dtype=np.float32), num_classes=2)
    architecture = {"stages": [16, 16], "head_channels": 32}
    model = build_detector(spec, rng=np.random.default_rng(0), **architecture)
    return data, spec, model, architecture, Normalizer.fit(data.images)


# overfitting 8 tiles reaches mAP 1.0 only if every piece agrees with the others
def test_training_reduces_the_cost_and_finds_the_boxes(site):
    data, spec, model, _, normalizer = tiny_training_setup(site)
    loss = YoloLoss()
    trainer = Trainer(model, loss, Adam(list(model.parameters().values()), lr=1e-2), spec, normalizer)
    history = trainer.fit(data, epochs=300, batch_size=8, rng=np.random.default_rng(0),
                          log=lambda _: None)
    assert history[-1]["train"]["cost"] < 0.01 * history[0]["train"]["cost"]
    detections, cost = predict(model, data, spec, normalizer, score_threshold=0.5, loss=loss)
    assert cost < 0.1
    assert evaluate_detections(detections, data.ground_truth(), 2, 0.5)["mAP"] == pytest.approx(1.0)


def test_trainer_extension_points(site):
    data, spec, model, _, normalizer = tiny_training_setup(site)

    # (lambd / 2) * sum of squared weights; gradient lambd * W
    def l2_penalty(parameters, lambd=0.1):
        total = 0.0
        for p in parameters.values():
            if p.kind == "weight":
                total += 0.5 * lambd * float(np.sum(p.data ** 2))
                p.grad += lambd * p.data
        return total

    class Recorder(Adam):
        def step(self):
            self.seen = [p.grad.copy() for p in self.params]

    plain = Trainer(model, YoloLoss(), Recorder(list(model.parameters().values()), 1e-3), spec, normalizer)
    J_plain, _ = plain.train_step(data.images, data.boxes, data.classes)
    decayed = Trainer(model, YoloLoss(), Recorder(list(model.parameters().values()), 1e-3), spec,
                      normalizer, penalties=[l2_penalty])
    J_decayed, parts = decayed.train_step(data.images, data.boxes, data.classes)

    weights = [p for p in model.parameters().values() if p.kind == "weight"]
    assert J_decayed - J_plain == pytest.approx(parts["penalty"])
    assert parts["penalty"] == pytest.approx(0.05 * sum(float(np.sum(p.data ** 2)) for p in weights), rel=1e-5)
    for p, g_plain, g_decayed in zip(model.parameters().values(), plain.optimizer.seen, decayed.optimizer.seen):
        expected = g_plain + (0.1 * p.data if p.kind == "weight" else 0) # biases are not decayed
        np.testing.assert_allclose(g_decayed, expected, rtol=1e-4, atol=1e-6)

    rates = []
    scheduled = Trainer(model, YoloLoss(), Adam(list(model.parameters().values()), 1.0), spec,
                        normalizer, lr_schedule=lambda epoch: 1e-3 / epoch)
    history = scheduled.fit(data, epochs=10, batch_size=4, rng=np.random.default_rng(0),
                            on_epoch_end=lambda epoch, record: rates.append(record["lr"]) or epoch == 3,
                            log=lambda _: None)
    assert len(history) == 3 and rates == pytest.approx([1e-3, 5e-4, 1e-3 / 3])


def test_checkpoint_round_trip(site, tmp_path):
    data, spec, model, architecture, normalizer = tiny_training_setup(site)
    save_checkpoint(tmp_path / "model.npz", model, spec, normalizer, architecture, data.class_names)
    model2, spec2, normalizer2, class_names = load_checkpoint(tmp_path / "model.npz")
    assert class_names == data.class_names and spec2.grid_size == 8 and spec2.num_classes == 2
    np.testing.assert_array_equal(spec2.anchors, spec.anchors)
    x = normalizer(data.images)
    np.testing.assert_array_equal(normalizer2(data.images), x)
    np.testing.assert_array_equal(model2.forward(x), model.forward(x))


def test_scripts_run_end_to_end(site, tmp_path):
    for folder, suffix in (("images", "_image.tif"), ("masks", "_mask.tif")):
        (tmp_path / "dd" / "data" / folder).mkdir(parents=True)
        for name in ("SiteA", "SiteB"): # two sites with the same pixels
            shutil.copy(site["dir"] / f"site{suffix}", tmp_path / "dd" / "data" / folder / f"{name}{suffix}")
    override = {
        "project": {"run_name": "tiny"},
        "paths": {"base_path": str(tmp_path / "dd"), "output_path": str(tmp_path / "out")},
        "data": {"tile_size": 64, "decimation": 2, "block_tiles": 2, "min_blob_area": 16,
                 "background_sample_ratio": 1.0, "dev_fraction": 0.25, "test_fraction": 0.25},
        "classes": {"label_remap": {0: 0, 3: 1, 5: 2}, "class_names": {0: "background", 1: "trash", 2: "tire"}},
        "model": {"stages": [8, 8], "head_channels": 8, "num_anchors": 2},
        "training": {"num_epochs": 2, "batch_size": 4},
    }
    (tmp_path / "tiny.yaml").write_text(yaml.safe_dump(override))

    def run(script, *overrides):
        command = [sys.executable, str(REPO / "scripts" / script)]
        for path in (tmp_path / "tiny.yaml", *overrides):
            command += ["--override", str(path)]
        result = subprocess.run(command, capture_output=True, text=True)
        return result

    built = run("build_dataset.py")
    assert built.returncode == 0, built.stderr
    meta = json.loads((tmp_path / "out" / "datasets" / "all_trash" / "meta.json").read_text())
    assert meta["site_names"] == ["SiteA", "SiteB"] and meta["class_names"] == ["trash", "tire"]
    assert [meta["stats"]["sites"][name]["tile_statuses"]["foreground"] for name in meta["site_names"]] == [17, 17]
    train = DetectionDataset.load(tmp_path / "out" / "datasets" / "all_trash", "train")
    assert set(train.tile_site.tolist()) == {0, 1} and train.site_names == ["SiteA", "SiteB"]

    trained = run("train.py")
    assert trained.returncode == 0, trained.stderr
    assert "best dev mAP@0.5" in trained.stdout
    run_dir = tmp_path / "out" / "runs" / "tiny"
    history = json.loads((run_dir / "history.json").read_text())
    assert len(history) == 2 and {"train", "train_sample", "dev"} <= set(history[0])
    assert yaml.safe_load((run_dir / "config.yaml").read_text())["model"]["stages"] == [8, 8]

    scored = run("evaluate.py")
    assert scored.returncode == 0, scored.stderr
    report = json.loads((run_dir / "best_test_metrics.json").read_text())
    assert report["split"] == "test" and "mAP" in report["iou_0.5"] and "score threshold" in scored.stdout
    assert set(report["iou_0.5"]["sites"]) <= {"SiteA", "SiteB"}
    assert (run_dir / "best_dev_predictions.png").stat().st_size > 0

    # a dataset built under other data settings is refused, not silently reused
    (tmp_path / "other.yaml").write_text(yaml.safe_dump({"data": {"min_blob_area": 32}}))
    refused = run("train.py", tmp_path / "other.yaml")
    assert refused.returncode != 0 and "different data/classes settings" in refused.stderr
