"""Image input/output, the project's colour conventions, and geometry helpers.

Everything that touches pixels directly lives here: reading the very large
source PNGs, turning each annotation convention into a boolean mask, removing
stray white background, rescaling image/label pairs together, and measuring how
deeply an object sits inside the section.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFile

from .config import (
    LUMEN_NORM_PERCENTILES,
    LUMEN_NORM_THRESHOLD,
    MIN_BUNDLE_AREA,
    NONBLACK_THRESHOLD,
    WHITE_BG_THRESHOLD,
)

# Sections routinely exceed PIL's decompression-bomb guard, and one source file
# is truncated. Both are tolerated deliberately.
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------
def read_rgb(path) -> np.ndarray:
    """Read any image as uint8 RGB, tolerating very large files."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def write_rgb(path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def read_bgr_as_rgb(path) -> np.ndarray:
    """Read a derived working-scale PNG written by :func:`write_rgb`."""
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def read_grey(path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


# ---------------------------------------------------------------------------
# Annotation conventions
# ---------------------------------------------------------------------------
def nonblack(rgb: np.ndarray) -> np.ndarray:
    """True where a pixel is annotated, for the "object on black" convention."""
    return rgb.astype(np.int32).sum(axis=2) > NONBLACK_THRESHOLD


def leaflet_mask(rgb: np.ndarray) -> np.ndarray:
    """Section footprint, taken from the leaflet image itself."""
    return nonblack(rgb).astype(np.uint8)


def clean_white_background(rgb: np.ndarray) -> np.ndarray:
    """Blacken white background that touches an image border.

    White regions fully enclosed by tissue are genuine air spaces and are kept.
    Only components reaching an edge are treated as background. The operation is
    idempotent and never modifies the file on disk: sections already on black
    pass through unchanged.
    """
    white = (
        (rgb[:, :, 0] > WHITE_BG_THRESHOLD)
        & (rgb[:, :, 1] > WHITE_BG_THRESHOLD)
        & (rgb[:, :, 2] > WHITE_BG_THRESHOLD)
    )
    if not white.any():
        return rgb

    height, width = white.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        white.astype(np.uint8), connectivity=8
    )
    outside = np.zeros((height, width), dtype=bool)
    for i in range(1, count):
        x, y, w, h, _area = stats[i]
        if x == 0 or y == 0 or x + w == width or y + h == height:
            outside |= labels == i
    if not outside.any():
        return rgb

    out = rgb.copy()
    out[outside] = 0
    return out


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------
def resize_rgb(rgb: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return rgb
    return cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def scaled_min_area(scale: float) -> int:
    """Minimum bundle area at a reduced scale; areas follow the square of it."""
    return max(1, int(MIN_BUNDLE_AREA * scale ** 2))


def resize_labels(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of an instance map to ``shape`` (height, width).

    OpenCV cannot resize int32, and the object count is far below 65535, so the
    map is routed through uint16.
    """
    height, width = shape
    resized = cv2.resize(
        labels.astype(np.uint16), (width, height), interpolation=cv2.INTER_NEAREST
    )
    return resized.astype(np.int32)


def rescale_pair(
    rgb: np.ndarray, instances: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink an image and its instance map together, then renumber objects.

    Objects falling below the rescaled minimum area are dropped, so that the
    training targets match what inference is able to return at the same scale.
    """
    if scale == 1.0:
        return rgb, instances

    small = resize_rgb(rgb, scale)
    labels = resize_labels(instances, small.shape[:2])
    minimum = scaled_min_area(scale)

    out = np.zeros_like(labels)
    next_id = 1
    for value in np.unique(labels):
        if value == 0:
            continue
        selected = labels == value
        if int(selected.sum()) >= minimum:
            out[selected] = next_id
            next_id += 1
    return small, out


# ---------------------------------------------------------------------------
# Instance geometry
# ---------------------------------------------------------------------------
def connected_instances(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Label a binary mask, keeping only components of at least ``min_area``."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    out = np.zeros(labels.shape, dtype=np.int32)
    next_id = 1
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = next_id
            next_id += 1
    return out


def depth_map(rgb: np.ndarray) -> np.ndarray:
    """Distance from every pixel to the section border; zero outside the section."""
    return cv2.distanceTransform(leaflet_mask(rgb), cv2.DIST_L2, 5)


def instance_geometry(instances: np.ndarray, depth: np.ndarray) -> list[tuple]:
    """Per instance, return ``(id, area, equivalent radius, centroid depth)``.

    Centroids are accumulated in a single pass rather than one mask per object,
    which matters on sections holding hundreds of bundles.
    """
    height, width = instances.shape
    flat = instances.ravel()
    filled = np.flatnonzero(flat)
    if filled.size == 0:
        return []

    labels = flat[filled]
    size = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=size).astype(float)
    sum_y = np.bincount(labels, weights=(filled // width).astype(float), minlength=size)
    sum_x = np.bincount(labels, weights=(filled % width).astype(float), minlength=size)

    out = []
    for i in range(1, size):
        if counts[i] == 0:
            continue
        cy = min(height - 1, max(0, int(round(sum_y[i] / counts[i]))))
        cx = min(width - 1, max(0, int(round(sum_x[i] / counts[i]))))
        radius = float(np.sqrt(counts[i] / np.pi))
        out.append((i, int(counts[i]), radius, float(depth[cy, cx])))
    return out


def filter_by_depth(
    instances: np.ndarray,
    depth: np.ndarray,
    min_depth: float | None = None,
    min_ratio: float | None = None,
) -> tuple[np.ndarray, int]:
    """Drop instances whose centroid sits too close to the section border.

    Trichomes are epidermal outgrowths, so their centroid is within roughly one
    radius of the cuticle, whereas a vascular bundle is buried in the mesophyll.
    ``min_ratio`` (depth divided by equivalent radius) is dimensionless and
    therefore survives a change of working scale; ``min_depth`` is in pixels at
    the current scale. Calibrate either one with ``faspy diagnose depth``.

    Returns the filtered map and the number of instances removed.
    """
    if min_depth is None and min_ratio is None:
        return instances, 0

    dropped = []
    for index, _area, radius, depth_value in instance_geometry(instances, depth):
        if min_depth is not None and depth_value < min_depth:
            dropped.append(index)
        elif min_ratio is not None and radius > 0 and depth_value / radius < min_ratio:
            dropped.append(index)

    if not dropped:
        return instances, 0
    out = instances.copy()
    out[np.isin(out, dropped)] = 0
    return out, len(dropped)


def draw_outline(image: np.ndarray, mask: np.ndarray, colour, thickness: int) -> None:
    """Draw the outline of ``mask`` onto a BGR image, in place."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, colour, thickness)


def lumen_mask(rgb: np.ndarray, inside: np.ndarray, leaflet: np.ndarray,
               threshold: float = LUMEN_NORM_THRESHOLD) -> np.ndarray:
    """Lumen pixels within ``inside``, by brightness relative to the section.

    A cavity stays bright in every channel while a stained wall does not, so the
    test is on the minimum of the three channels rather than on luminance.

    The threshold is a fraction of the section's own dynamic range, taken from
    percentiles of the darkest channel over the whole leaflet. An absolute level
    does not survive a change of acquisition pipeline: a level calibrated on one
    set of images under-detected another by a factor of five, the two differing
    only in exposure.
    """
    if not inside.any() or not leaflet.any():
        return np.zeros_like(inside)
    darkest = rgb.min(axis=2)
    low, high = np.percentile(darkest[leaflet], LUMEN_NORM_PERCENTILES)
    normalised = (darkest.astype(np.float32) - low) / max(float(high - low), 1.0)
    return (normalised > threshold) & inside
