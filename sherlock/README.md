# Running on Sherlock

Every stage of the pipeline is an sbatch job. Nothing needs a GPU: the model is
numpy, so the jobs ask for CPU cores on the `normal` partition.

Jobs run inside the drone_dengue Singularity image
(`/home/groups/jrosser/drone_dengue/segmentation_container_multiclass.sif`).
It is used only because it already has numpy, rasterio, PyYAML and matplotlib;
its PyTorch is never imported.

## Data

Nothing is copied from a laptop. The jobs read the drone_dengue data where it
already lives, in the layout that repository uses:

```
/scratch/groups/jrosser/drone_dengue/data/
    images/<site>_image.tif
    masks/<site>_mask.tif
```

Every image with a matching mask becomes a site in the dataset. The paths are
set in `configs/sherlock/sherlock.yaml`, which every job applies on top of
`configs/base_config.yaml`. If scratch has been purged and the data is only in
group home, change `paths.base_path` there to `/home/groups/jrosser/drone_dengue`.

## One-time setup

Put the code on Sherlock. From your laptop, in the repository root:

```bash
rsync -av --exclude data --exclude runs --exclude 'logs/*.out' --exclude __pycache__ --exclude .pytest_cache ./ <sunet>@login.sherlock.stanford.edu:glassbox-ml/
```

That lands in `~/glassbox-ml`. Commit first if you want each job log to record
the commit it ran.

## Running

Submit from the repository root, so that `logs/` and the code resolve:

```bash
cd ~/glassbox-ml
T=$(sbatch --parsable sherlock/test.sbatch)
B=$(sbatch --parsable --dependency=afterok:$T sherlock/build_dataset.sbatch)
R=$(sbatch --parsable --dependency=afterok:$B sherlock/train.sbatch)
sbatch --dependency=afterok:$R sherlock/evaluate.sbatch
```

Each job only starts if the one before it succeeded.

| Job | What it does | Resources |
| --- | --- | --- |
| `test.sbatch` | The test suite, inside the container | 4 cores, 20 min |
| `build_dataset.sbatch` | Image/mask pairs to tiles + boxes | 2 cores, 32 GB, up to 4 h |
| `train.sbatch` | Trains the detector | 16 cores, 32 GB, up to 24 h |
| `evaluate.sbatch` | Dev and test metrics per site, prediction images | 8 cores, 1 h |

## Experiments

An experiment is an override YAML, not a set of flags. Name the overrides in
`OVERRIDES` (paths relative to the checkout, space-separated, applied in order
after the Sherlock paths):

```bash
OVERRIDES="configs/experiments/wide.yaml" sbatch sherlock/train.sbatch
```

```bash
OVERRIDES="configs/experiments/wide.yaml" sbatch sherlock/evaluate.sbatch
```

Evaluate with the same overrides the run was trained with: `project.run_name`
in the override is what locates the run. An override that changes `data` or
`classes` needs its own `paths.dataset_folder` and its own build job first
(`configs/experiments/multiclass.yaml` and `quick_test.yaml` do this); training
refuses a dataset that was built under different settings.

To bill another account, `sbatch --account=<name> ...`.

## Where things go

```
~/glassbox-ml/logs/gb_<job>_<jobid>.out          job logs
/scratch/groups/jrosser/glassbox_ml/
    datasets/<dataset_folder>/                   the built tiles, boxes and meta.json
    runs/<run_name>/
        config.yaml  history.json                the merged config; per-epoch costs and mAP
        last.npz  best.npz                       checkpoints
        best_{dev,test}_metrics.json             written by evaluate.sbatch
        best_{dev,test}_predictions.png
    pydeps/                                      pytest, installed by test.sbatch
```

Scratch is purged after 90 days without access. Copy any run worth keeping to
group home or Oak.

## Not yet verified on Sherlock

These scripts were dry-run on a laptop with a stand-in for `singularity`, and
the tests pass under numpy 1.26 to match the container's `numpy<2` pin. Nothing
has run on the cluster. What that leaves unchecked:

- that the image/mask pairs are at the path above, that each mask is a
  single-band 8-bit raster on the same pixel grid as its image, and that
  nodata is 255 in the mask or an alpha band in the image (the build assumes all three);
- that the container is readable by your account, and that `normal` +
  `--account=jrosser` is the right combination for you;
- the pytest install in `test.sbatch`, which needs the compute node to reach PyPI;
- how large the built dataset is, and so the memory and time a build and an
  epoch need. On an Apple M-series laptop a training step costs about 6 ms per
  128 x 128 tile.
