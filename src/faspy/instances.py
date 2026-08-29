"""Instance route: Cellpose-SAM fine-tuned on vascular bundles.

Bundles are counted, not painted. Treating each one as a separate object gives a
native count, separates touching bundles, and tolerates the irregular outlines
that a distance- or shape-based rule cannot.

The single decisive setting is the scale. Cellpose-SAM trains on fixed 256 px
crops, and that window cannot be changed, so the data must be brought to it: see
``CELLPOSE_RESCALE`` in the configuration. Training and inference must use the
same value, otherwise recall on small objects collapses.
"""
from __future__ import annotations

import numpy as np

from . import imaging
from .config import (
    CELLPOSE_EPOCHS,
    CELLPOSE_LEARNING_RATE,
    CELLPOSE_RESCALE,
    CELLPOSE_WEIGHT_DECAY,
    CELLPROB_THRESHOLD,
    DIR_CELLPOSE_MODELS,
    FLOW_THRESHOLD,
    MIN_BUNDLE_AREA,
)
from .datasets import load_pair

MODELS_SUBDIR = "models"


def checkpoint_path(name: str):
    return DIR_CELLPOSE_MODELS / MODELS_SUBDIR / name


def gpu_available() -> bool:
    import torch

    return torch.cuda.is_available()


def load_model(checkpoint=None, gpu=None):
    """Load Cellpose-SAM, optionally from a fine-tuned checkpoint."""
    from cellpose.models import CellposeModel

    if gpu is None:
        gpu = gpu_available()
    if checkpoint is None:
        return CellposeModel(gpu=gpu)
    return CellposeModel(gpu=gpu, pretrained_model=str(checkpoint))


def train(rows, name: str, epochs=CELLPOSE_EPOCHS, rescale=CELLPOSE_RESCALE, gpu=None):
    """Fine-tune Cellpose-SAM on the given rows and save it under ``name``.

    Training images are held in memory as a list, as the Cellpose API requires,
    so peak memory grows with the number of sections and shrinks with ``rescale``.
    """
    from cellpose import train as cellpose_train

    images, labels = [], []
    for row in rows:
        rgb, instances = load_pair(row)
        rgb, instances = imaging.rescale_pair(rgb, instances, rescale)
        images.append(rgb)
        labels.append(instances)

    DIR_CELLPOSE_MODELS.mkdir(parents=True, exist_ok=True)
    model = load_model(gpu=gpu)
    cellpose_train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        channel_axis=2,
        n_epochs=epochs,
        learning_rate=CELLPOSE_LEARNING_RATE,
        weight_decay=CELLPOSE_WEIGHT_DECAY,
        batch_size=1,
        min_train_masks=1,
        normalize=True,
        save_path=str(DIR_CELLPOSE_MODELS),
        model_name=name,
    )
    return model


def predict(
    model,
    rgb: np.ndarray,
    rescale=CELLPOSE_RESCALE,
    cellprob=CELLPROB_THRESHOLD,
    flow=FLOW_THRESHOLD,
    min_size=None,
) -> np.ndarray:
    """Segment bundles and return an instance map at the input resolution."""
    if rescale == 1.0:
        return model.eval(
            rgb,
            channel_axis=2,
            min_size=MIN_BUNDLE_AREA if min_size is None else min_size,
            cellprob_threshold=cellprob,
            flow_threshold=flow,
        )[0].astype(np.int32)

    small = imaging.resize_rgb(rgb, rescale)
    # min_size doit suivre la reduction d'echelle, mais en partant du seuil
    # demande : sans cela le seuil propre a la coupe etait ignore des que
    # rescale != 1, c'est-a-dire en production.
    base = MIN_BUNDLE_AREA if min_size is None else min_size
    instances = model.eval(
        small,
        channel_axis=2,
        min_size=max(1, int(base * rescale ** 2)),
        cellprob_threshold=cellprob,
        flow_threshold=flow,
    )[0]
    return imaging.resize_labels(instances, rgb.shape[:2])


def median_diameter(instance_maps) -> float:
    """Median equivalent diameter, in pixels, over a set of instance maps.

    Used to check that training data actually sits inside the window Cellpose
    expects before committing to a run.
    """
    areas = []
    for instances in instance_maps:
        values = [int((instances == i).sum()) for i in np.unique(instances) if i > 0]
        if values:
            areas.append(np.median(values))
    if not areas:
        return float("nan")
    return float(2 * np.sqrt(np.median(areas) / np.pi))
