# draw tiles with their boxes (the only module that imports matplotlib)

from __future__ import annotations

from pathlib import Path

import numpy as np


# ground truth in green, predictions as (boxes, scores) per tile in red
def save_tile_grid(path: str | Path, images: np.ndarray, ground_truth: list[np.ndarray],
                   predictions: list[tuple[np.ndarray, np.ndarray]] | None = None,
                   columns: int = 6, titles: list[str] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows = int(np.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, image in enumerate(images):
        ax = axes[k // columns][k % columns]
        ax.imshow(image, interpolation="nearest")
        for x1, y1, x2, y2 in ground_truth[k]: # imshow centers pixels on integers, hence - 0.5
            ax.add_patch(Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="lime", linewidth=1.5))
        if predictions is not None:
            for (x1, y1, x2, y2), score in zip(*predictions[k]):
                ax.add_patch(Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1, y2 - y1,
                                       fill=False, edgecolor="red", linewidth=1.2, linestyle="--"))
                ax.text(x1, max(y1 - 2, 4), f"{score:.2f}", color="red", fontsize=7)
        if titles is not None:
            ax.set_title(titles[k], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
