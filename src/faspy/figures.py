"""Reproducible figures that show what the pipeline does to a real section.

Every panel is computed from the data, never drawn by hand, so a figure always
matches the code that produced it.

The figure reads as a flow diagram: each stage is an image, and the operation
that leads to the next stage is written on the arrow between them. Panels are
placed explicitly rather than through a grid, because straightened sections are
roughly seven times wider than they are tall while the detail views are square,
and no single grid accommodates both without leaving large gaps.
"""
from __future__ import annotations

import textwrap

import numpy as np

from . import imaging
from .config import (
    CELLPOSE_RESCALE,
    DIR_BUNDLE,
    DIR_EXAMPLES,
    DIR_LEAFLET,
    DIR_OUT,
    LUMEN_NORM_THRESHOLD,
    MIN_BUNDLE_AREA,
    MIN_BUNDLE_AREA_FULLRES,
    PATCH,
    WORKING_SCALE,
)

# Qualitative palette, colour-blind safe, cycled over bundle instances.
_PALETTE = np.array(
    [
        (230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66),
        (0, 114, 178), (213, 94, 0), (204, 121, 167), (153, 153, 153),
    ],
    dtype=np.uint8,
)

# Layout constants, in inches unless stated otherwise.
FIGURE_WIDTH = 13.5
MARGIN = 0.5
COLUMN_GAP = 0.36
BLOCK_GAP = 0.30
ARROW_GAP = 0.92          # vertical room reserved for a flow arrow
TITLE_HEIGHT = 0.28
CAPTION_LINE = 0.165
CAPTION_PAD = 0.09

TITLE_SIZE = 11.5
CAPTION_SIZE = 8.6
ARROW_SIZE = 9.2
INK = "#111111"
SUBDUED = "#5a5a5a"
FLOW = "#2f6f8f"

_CHARS_PER_INCH = 14.8    # measured at CAPTION_SIZE; deliberately conservative
                          # so a caption never runs past its column


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def colour_instances(instances: np.ndarray, background: np.ndarray | None = None) -> np.ndarray:
    """Render an instance map, one colour per object, over an optional backdrop."""
    if background is None:
        out = np.zeros((*instances.shape, 3), dtype=np.uint8)
    else:
        out = (background.astype(np.float32) * 0.35).astype(np.uint8)
    for position, index in enumerate(i for i in np.unique(instances) if i > 0):
        out[instances == index] = _PALETTE[position % len(_PALETTE)]
    return out


def _square(box, shape):
    """Expand a bounding box to a square, clamped to the image."""
    y0, y1, x0, x1 = box
    side = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0, x0 = cy - side // 2, cx - side // 2
    y0 = max(0, min(y0, shape[0] - side)) if side <= shape[0] else 0
    x0 = max(0, min(x0, shape[1] - side)) if side <= shape[1] else 0
    return y0, min(shape[0], y0 + side), x0, min(shape[1], x0 + side)


def _bundle_box(instances: np.ndarray, margin=0.25, pick="largest"):
    """Bounding box of one instance, expanded by ``margin``.

    ``pick`` selects the largest object, or the one closest to the median area.
    The median is what a size argument should rest on: the midrib is an outlier
    an order of magnitude above the rest, and illustrating anything with it
    misrepresents the population.
    """
    sizes = [(int((instances == i).sum()), i) for i in np.unique(instances) if i > 0]
    if not sizes:
        return None
    if pick == "median":
        ordered = sorted(sizes)
        _, index = ordered[len(ordered) // 2]
    else:
        _, index = max(sizes)
    ys, xs = np.where(instances == index)
    pad_y = int(margin * (ys.max() - ys.min())) + 5
    pad_x = int(margin * (xs.max() - xs.min())) + 5
    return (
        max(0, ys.min() - pad_y), min(instances.shape[0], ys.max() + pad_y),
        max(0, xs.min() - pad_x), min(instances.shape[1], xs.max() + pad_x),
    )


class Alignment:
    """Rotate a section onto its long axis so it fills a landscape panel.

    Sections are narrow strips lying at an arbitrary angle, so their bounding box
    covers the whole frame and cropping alone gains nothing. The long axis comes
    from the minimum-area rectangle of the section mask, and the same rotation is
    applied to every layer, which keeps the panels comparable.
    """

    def __init__(self, mask: np.ndarray):
        import cv2

        points = cv2.findNonZero(mask.astype(np.uint8))
        (centre_x, centre_y), (width, height), angle = cv2.minAreaRect(points)
        if width < height:
            angle += 90.0

        source_height, source_width = mask.shape
        matrix = cv2.getRotationMatrix2D((centre_x, centre_y), angle, 1.0)
        cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
        self.size = (
            int(source_height * sine + source_width * cosine),
            int(source_height * cosine + source_width * sine),
        )
        matrix[0, 2] += self.size[0] / 2 - centre_x
        matrix[1, 2] += self.size[1] / 2 - centre_y
        self.matrix = matrix
        self.box = self._content_box(self.apply(mask.astype(np.uint8), nearest=True) > 0)

    def apply(self, image: np.ndarray, nearest=False) -> np.ndarray:
        import cv2

        return cv2.warpAffine(
            image, self.matrix, self.size,
            flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        )

    def crop(self, image: np.ndarray, nearest=False, pad=0.0) -> np.ndarray:
        """Rotate, then crop to the section with an optional margin."""
        y0, y1, x0, x1 = self.box
        if pad:
            grow = int(pad * (y1 - y0))
            y0, y1 = max(0, y0 - grow), min(self.size[1], y1 + grow)
            x0, x1 = max(0, x0 - grow), min(self.size[0], x1 + grow)
        return self.apply(image, nearest)[y0:y1, x0:x1]

    @staticmethod
    def _content_box(mask: np.ndarray, pad=6):
        ys, xs = np.where(mask)
        if ys.size == 0:
            return 0, mask.shape[0], 0, mask.shape[1]
        return (
            max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad),
            max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad),
        )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
class Block:
    """One stage: an image or a drawn panel, a short title, a one-line caption."""

    def __init__(self, label, title, caption="", image=None, aspect=None, draw=None):
        self.label = label
        self.title = title
        self.caption = caption
        self.image = image
        self.draw = draw
        self.aspect = aspect if aspect is not None else (
            image.shape[0] / image.shape[1] if image is not None else 1.0
        )

    def wrapped(self, width_in):
        if not self.caption:
            return ""
        measure = max(24, int(width_in * _CHARS_PER_INCH))
        return textwrap.fill(" ".join(self.caption.split()), measure)

    def height(self, width_in):
        text = self.wrapped(width_in)
        lines = (text.count("\n") + 1) if text else 0
        return TITLE_HEIGHT + width_in * self.aspect + (CAPTION_PAD + lines * CAPTION_LINE if lines else 0)


class Row:
    def __init__(self, blocks, arrows=None):
        self.blocks = blocks
        # One label per block, written on the arrow entering that block.
        self.arrows = arrows or []


class Sheet:
    """Stack rows from the top of the figure and connect them with arrows."""

    def __init__(self, width=FIGURE_WIDTH, margin=MARGIN, header=0.80, footer=0.30):
        self.width = width
        self.margin = margin
        self.header = header
        self.footer = footer
        self.rows: list[Row] = []

    def add(self, *blocks, arrows=None):
        self.rows.append(Row(list(blocks), arrows))

    def _column_width(self, count):
        usable = self.width - 2 * self.margin
        return (usable - (count - 1) * COLUMN_GAP) / count

    def height(self):
        total = self.header + self.footer + self.margin
        for position, row in enumerate(self.rows):
            column = self._column_width(len(row.blocks))
            total += max(block.height(column) for block in row.blocks)
            if position < len(self.rows) - 1:
                total += ARROW_GAP if self.rows[position + 1].arrows else BLOCK_GAP
        return total

    def render(self, figure):
        import matplotlib.patches as patches

        sheet_height = self.height()
        overlay = figure.add_axes([0, 0, 1, 1], zorder=5)
        overlay.axis("off")
        overlay.set_xlim(0, 1)
        overlay.set_ylim(0, 1)

        def fx(value):
            return value / self.width

        def fy(value):
            return value / sheet_height

        cursor = sheet_height - self.header
        previous_bottom = None

        for row in self.rows:
            column = self._column_width(len(row.blocks))
            tallest = max(block.height(column) for block in row.blocks)
            centres = []

            for position, block in enumerate(row.blocks):
                left = self.margin + position * (column + COLUMN_GAP)
                self._place(figure, block, left, cursor, column, fx, fy)
                centres.append(left + column / 2)

            if previous_bottom is not None and row.arrows:
                for position, label in enumerate(row.arrows):
                    if not label:
                        continue
                    start_x = centres[position] if len(row.arrows) > 1 else centres[0]
                    self._arrow(
                        overlay, patches, fx, fy,
                        previous_bottom, cursor, start_x, centres[position], label,
                    )

            previous_bottom = cursor - tallest
            cursor = previous_bottom - (ARROW_GAP if True else BLOCK_GAP)

        return sheet_height

    @staticmethod
    def _arrow(overlay, patches, fx, fy, top_y, bottom_y, start_x, end_x, label):
        """Draw one flow arrow and write the operation beside it."""
        arrow = patches.FancyArrowPatch(
            (fx(start_x), fy(top_y - 0.10)),
            (fx(end_x), fy(bottom_y + 0.06)),
            transform=overlay.transData,
            arrowstyle="-|>", mutation_scale=20,
            linewidth=2.0, color=FLOW,
            connectionstyle="arc3,rad=0.0" if abs(start_x - end_x) < 0.05 else "arc3,rad=0.12",
            shrinkA=0, shrinkB=0, zorder=6,
        )
        overlay.add_patch(arrow)
        overlay.text(
            fx((start_x + end_x) / 2) + 0.006, fy((top_y + bottom_y) / 2),
            label, fontsize=ARROW_SIZE, color=FLOW, fontweight="bold",
            ha="left", va="center", zorder=7,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.92},
        )

    def _place(self, figure, block, left, top, width, fx, fy):
        figure.text(
            fx(left), fy(top - TITLE_HEIGHT + 0.08),
            f"{block.label}   {block.title}",
            fontsize=TITLE_SIZE, fontweight="bold", color=INK, va="baseline", ha="left",
        )

        image_top = top - TITLE_HEIGHT
        image_height = width * block.aspect
        axes = figure.add_axes(
            [fx(left), fy(image_top - image_height), fx(width), fy(image_height)]
        )
        if block.image is not None:
            axes.imshow(block.image, aspect="auto", interpolation="antialiased")
            axes.set_xticks([])
            axes.set_yticks([])
            for spine in axes.spines.values():
                spine.set_edgecolor("#c9c9c9")
                spine.set_linewidth(0.8)
        else:
            axes.axis("off")
        if block.draw is not None:
            block.draw(axes)

        caption = block.wrapped(width)
        if caption:
            figure.text(
                fx(left), fy(image_top - image_height - CAPTION_PAD),
                caption, fontsize=CAPTION_SIZE, color=SUBDUED,
                va="top", ha="left", linespacing=1.5,
            )


# ---------------------------------------------------------------------------
# Section preparation
# ---------------------------------------------------------------------------
def _source_paths(key: str):
    """Raw section, leaflet image and bundle annotation for one key."""
    raw = DIR_EXAMPLES / key / f"{key}.png"
    leaflet = DIR_LEAFLET / f"{key}_LM.png"
    if not leaflet.exists():
        alternative = DIR_EXAMPLES / key / f"{key}_LM.png"
        leaflet = alternative if alternative.exists() else DIR_EXAMPLES / key / f"{key}LM.png"
    return (raw if raw.exists() else None), leaflet, DIR_BUNDLE / f"{key}_BU.png"


def _lumen_share(leaflet_full, bundle_mask, leaflet):
    """Fraction luminale du faisceau, mesuree comme en production.

    Passait par ``LUMEN_SCALE / WORKING_SCALE``, un rapport qui vaut un : la
    figure annoncait donc une valeur mesuree a l'echelle de travail alors que le
    reste du panneau, et la production, mesurent sur l'image d'acquisition.
    Le chiffre affiche differait de celui du tableau.
    """
    from .quantify import lumen_at_full_resolution

    if not bundle_mask.any():
        return 0.0
    fine = lumen_at_full_resolution(leaflet_full, bundle_mask, leaflet)
    denominator = float(fine["bundle"].sum())
    return float(fine["lumen"].sum()) / denominator if denominator else 0.0


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def pipeline_figure(key: str, output=None, model_name=None, rescale=CELLPOSE_RESCALE, dpi=220):
    """Render the pipeline flow diagram for one section."""
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    raw_path, leaflet_path, bundle_path = _source_paths(key)
    if not leaflet_path.exists() or not bundle_path.exists():
        raise SystemExit(f"Section {key}: leaflet or bundle image not found.")

    # Deux images distinctes : `raw` sert d'illustration du panneau A et peut
    # venir d'un autre fichier, tandis que `leaflet_full` est celle dont derivent
    # tous les masques. Mesurer le lumen sur la premiere alignerait des masques
    # sur une image qui n'est pas la leur.
    leaflet_full = imaging.read_rgb(leaflet_path)
    raw = imaging.read_rgb(raw_path) if raw_path else leaflet_full
    full_height, full_width = raw.shape[:2]

    scaled = imaging.clean_white_background(
        imaging.resize_rgb(leaflet_full, WORKING_SCALE)
    )
    leaflet = imaging.nonblack(scaled)
    align = Alignment(leaflet)

    bundle_rgb = cv2.resize(
        imaging.read_rgb(bundle_path), (scaled.shape[1], scaled.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    annotated = imaging.connected_instances(imaging.nonblack(bundle_rgb) & leaflet, MIN_BUNDLE_AREA)

    segmented, source_note = annotated, "from the annotation"
    if model_name:
        from . import instances as instance_route

        model = instance_route.load_model(instance_route.checkpoint_path(model_name))
        segmented = instance_route.predict(model, scaled, rescale)
        segmented[~leaflet] = 0
        source_note = "predicted"

    bundle_mask = segmented > 0
    to_full = 1.0 / (WORKING_SCALE ** 2)
    n_bundles = len([i for i in np.unique(segmented) if i > 0])
    leaflet_px, bundle_px = int(leaflet.sum()), int(bundle_mask.sum())
    lumen_share = _lumen_share(leaflet_full, bundle_mask, leaflet)
    diameters = [
        2 * np.sqrt(int((annotated == i).sum()) / np.pi) for i in np.unique(annotated) if i > 0
    ]
    median_diameter = float(np.median(diameters)) if diameters else float("nan")

    sheet = Sheet()

    # -- A ------------------------------------------------------------------
    raw_small = cv2.resize(raw, (scaled.shape[1], scaled.shape[0]), interpolation=cv2.INTER_AREA)
    sheet.add(Block(
        "A", "Section as imaged",
        "FASGA: lignin red-magenta, cellulose blue. Pale surround is mounting medium.",
        image=align.crop(raw_small, pad=0.16),
    ))

    # -- B ------------------------------------------------------------------
    footprint = align.crop(scaled).copy()
    imaging.draw_outline(
        footprint, align.crop(leaflet.astype(np.uint8), nearest=True), (255, 255, 255), 3
    )
    sheet.add(
        Block(
            "B", f"Leaflet footprint  ·  {100 * leaflet.mean():.1f} % of frame",
            "Denominator of every ratio.",
            image=footprint,
        ),
        arrows=["blacken background\ntouching a border"],
    )

    # -- C ------------------------------------------------------------------
    instance_view = align.crop(colour_instances(segmented, scaled), nearest=True)
    if model_name:
        imaging.draw_outline(
            instance_view, align.crop((annotated > 0).astype(np.uint8), nearest=True),
            (255, 255, 255), 3,
        )
    # The centroid of each object, from which depth and spacing are measured.
    cropped_instances = align.crop(segmented.astype(np.int32), nearest=True)
    centroids = []
    for index in (i for i in np.unique(cropped_instances) if i > 0):
        ys, xs = np.nonzero(cropped_instances == index)
        if ys.size:
            centroids.append((xs.mean(), ys.mean()))

    def draw_centroids(axes):
        if centroids:
            axes.scatter([c[0] for c in centroids], [c[1] for c in centroids],
                         s=26, c="white", edgecolor="#111111", linewidth=1.1, zorder=6)

    sheet.add(
        Block(
            "C", f"Bundle instances  ·  {n_bundles} objects  ({source_note})",
            "One colour per object, its centroid marked; touching bundles stay apart."
            + (" White outlines: annotation." if model_name else ""),
            image=instance_view, draw=draw_centroids,
        ),
        arrows=[f"instances.predict  ·  Cellpose-SAM\nat scale {rescale}"],
    )

    # -- D, E, F ------------------------------------------------------------
    y0, y1, x0, x1 = _square(_bundle_box(annotated, margin=0.75, pick="median"), scaled.shape[:2])
    scale_crop = scaled[y0:y1, x0:x1]

    # What decides whether the model can see a bundle is not the crop but the
    # OBJECT DIAMETER it was pretrained on: 7.5 to 120 px. Presented unscaled,
    # a bundle of this size sits above that ceiling; rescaling by 0.35 divides
    # its apparent diameter and brings it inside.
    areas = sorted(int((annotated == i).sum()) for i in np.unique(annotated) if i > 0)
    typical_area = areas[len(areas) // 2]
    typical_diameter = 2 * np.sqrt(typical_area / np.pi)
    biggest_diameter = 2 * np.sqrt(areas[-1] / np.pi)

    def draw_windows(axes):
        centre_y, centre_x = scale_crop.shape[0] / 2, scale_crop.shape[1] / 2
        rings = (
            (PRETRAINING_DIAMETERS[1], "#d55e00",
             f"ceiling at scale 1.0 — {PRETRAINING_DIAMETERS[1]:.0f} px"),
            (PRETRAINING_DIAMETERS[1] / rescale, "#009e73",
             f"ceiling at scale {rescale} — {PRETRAINING_DIAMETERS[1] / rescale:.0f} px"),
        )
        style = {"fontsize": 8.4, "fontweight": "bold", "ha": "center",
                 "bbox": {"boxstyle": "round,pad=0.22", "facecolor": "white",
                          "edgecolor": "none", "alpha": 0.9}}
        # Labels stacked in a corner rather than against each ring: the larger
        # ring runs past the crop, so a label tied to it lands outside the panel.
        for position, (diameter, colour, label) in enumerate(rings):
            axes.add_patch(patches.Circle((centre_x, centre_y), diameter / 2,
                                          linewidth=2.6, edgecolor=colour,
                                          facecolor="none", linestyle="--"))
            axes.text(scale_crop.shape[1] * 0.03,
                      scale_crop.shape[0] * (0.93 - 0.075 * position),
                      label, color=colour, **{**style, "ha": "left"})
        axes.annotate(
            f"a typical bundle:\n{typical_diameter:.0f} px across",
            xy=(centre_x + typical_diameter * 0.35, centre_y + typical_diameter * 0.35),
            xytext=(scale_crop.shape[1] * 0.97, scale_crop.shape[0] * 0.06),
            fontsize=8.6, fontweight="bold", color="#8a2d00", ha="right", va="center",
            arrowprops={"arrowstyle": "-|>", "color": "#8a2d00", "linewidth": 1.6,
                        "shrinkA": 4, "shrinkB": 2},
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.92},
        )

    y0, y1, x0, x1 = _square(_bundle_box(segmented), scaled.shape[:2])
    lumen_crop = scaled[y0:y1, x0:x1].copy()
    inside = bundle_mask[y0:y1, x0:x1]
    lumen = imaging.lumen_mask(lumen_crop, inside, imaging.nonblack(lumen_crop))
    lumen_view = lumen_crop.copy()
    lumen_view[lumen] = (0, 90, 255)
    imaging.draw_outline(lumen_view, inside.astype(np.uint8), (0, 0, 0), 3)
    lumen_pct = 100 * lumen.sum() / max(inside.sum(), 1)

    def draw_lumen_pointer(axes):
        ys, xs = np.where(lumen)
        if ys.size == 0:
            return
        target_y, target_x = int(np.median(ys)), int(np.median(xs))
        axes.annotate(
            "lumen", xy=(target_x, target_y),
            xytext=(lumen_view.shape[1] * 0.06, lumen_view.shape[0] * 0.12),
            fontsize=9, fontweight="bold", color="#0a3fa8", ha="left",
            arrowprops={"arrowstyle": "-|>", "color": "#0a3fa8", "linewidth": 1.6},
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.9},
        )

    # -- F: where the tissue sits -------------------------------------------
    # Position matters as much as amount: a pixel contributes to bending as the
    # SQUARE of its distance from the leaflet mid-plane, so this field is what
    # the geometric traits weigh.
    from . import traits as trait_route
    from .quantify import lumen_at_full_resolution, to_working_scale

    # Meme mesure que la production : le lumen sur l'image d'origine, la
    # geometrie a l'echelle de travail. Une figure qui montrerait d'autres
    # chiffres que le CSV serait pire qu'inutile.
    fine = lumen_at_full_resolution(leaflet_full, bundle_mask, leaflet)
    full_lumen = to_working_scale(fine["lumen"], bundle_mask.shape)
    measured = trait_route.section_traits(leaflet, segmented, full_lumen, fine=fine)
    field, rotated_leaflet, _, mid_line = trait_route.mid_plane(leaflet, bundle_mask)

    # A stretch of the lamina rather than the whole strip: at a third of the
    # sheet's width the full section collapses to an unreadable band, whereas a
    # window three times as wide as the leaflet is thick shows the gradient.
    y0f, y1f, x0f, x1f = Alignment._content_box(rotated_leaflet)
    thickness = y1f - y0f
    centre = (x0f + x1f) // 2
    x0f = max(x0f, centre - int(1.5 * thickness))
    x1f = min(x1f, centre + int(1.5 * thickness))
    depth_view = np.where(rotated_leaflet, field * trait_route.PIXEL_UM,
                          np.nan)[y0f:y1f, x0f:x1f]

    reliable_mid, _ = trait_route.midplane_reliable(rotated_leaflet)

    def draw_mid_plane(axes):
        axes.imshow(depth_view, cmap="magma", aspect="auto")
        if not reliable_mid:
            # Ne pas dessiner un repere que la table declare invalide : la
            # figure contredirait la mesure, et c'est la figure qu'on croit.
            message = ("Mid-plane unavailable: section cannot"
                       + chr(10) + "be represented as a single-valued strip")
            axes.text(0.5, 0.5, message,
                      transform=axes.transAxes, ha="center", va="center",
                      fontsize=9, fontweight="bold", color="white",
                      bbox={"boxstyle": "round,pad=0.4", "facecolor": "#8a1c1c",
                            "edgecolor": "none", "alpha": 0.92})
            axes.set_xlim(0, depth_view.shape[1])
            axes.set_ylim(depth_view.shape[0], 0)
            return
        columns = np.arange(mid_line.size)
        keep = (columns >= x0f) & (columns < x1f) & rotated_leaflet.any(axis=0)
        axes.plot(columns[keep] - x0f, mid_line[keep] - y0f, color="#00e5c0", lw=2.4)
        anchor = float(np.nanmean(mid_line[keep])) - y0f
        axes.annotate("mid-plane", xy=(depth_view.shape[1] * 0.5, anchor),
                      xytext=(0, -26), textcoords="offset points",
                      ha="center", va="center", fontsize=8.6, fontweight="bold",
                      color="#00b899", zorder=6,
                      arrowprops={"arrowstyle": "-|>", "color": "#00b899",
                                  "linewidth": 1.3, "shrinkB": 3},
                      bbox={"boxstyle": "round,pad=0.22", "facecolor": "white",
                            "edgecolor": "none", "alpha": 0.92})
        axes.set_xlim(0, depth_view.shape[1])
        axes.set_ylim(depth_view.shape[0], 0)

    sheet.add(
        Block("D", "Object size against the pretraining range",
              f"Median {typical_diameter:.0f} px, largest {biggest_diameter:.0f} px: "
              f"the midrib stays above the range even rescaled.",
              image=scale_crop, draw=draw_windows),
        Block("E", "Lumen",
              f"min(R,G,B) above {LUMEN_NORM_THRESHOLD:.0%} of the section's own range  ·  "
              f"{lumen_pct:.1f} % of the bundle.",
              image=lumen_view, draw=draw_lumen_pointer),
        Block("F", "Distance to the mid-plane",
              "Pale counts far more than dark: bending weighs the square of it.",
              aspect=depth_view.shape[0] / depth_view.shape[1], draw=draw_mid_plane),
        arrows=["the setting that\nmade it work",
                "imaging.lumen_mask\nrelative to each section",
                "traits.mid_plane\nfollows the lamina"],
    )

    # -- G: the row written out ---------------------------------------------
    readout = [
        ("bundles", f"{n_bundles}"),
        ("bundle / leaflet", f"{100 * bundle_px / max(leaflet_px, 1):.2f} %"),
        ("wall / bundle", f"{100 * float(measured['wall_over_bundle']):.1f} %"),
        ("thickness", f"{measured['leaflet_thickness_um']:.0f} µm"),
        ("bundle diameter", f"{measured['bundle_diameter_median_um']:.0f} µm"),
        ("depth below epidermis", f"{measured['bundle_depth_median_um']:.0f} µm"),
        ("stiffness share", f"{100 * float(measured['I_bundle_share_flat']):.1f} %"),
        ("vessels", f"{measured['n_vessels']}"),
    ]

    def draw_readout(axes):
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.add_patch(patches.FancyBboxPatch(
            (0.004, 0.10), 0.992, 0.80, boxstyle="round,pad=0.012",
            facecolor="#f6f6f6", edgecolor="#d5d5d5", linewidth=1.0,
        ))
        for position, (name, value) in enumerate(readout):
            x = 0.035 + position * 0.1205
            axes.text(x, 0.63, value, fontsize=13, family="monospace",
                      color=INK, va="center", fontweight="bold")
            axes.text(x, 0.34, name, fontsize=8.4, color=SUBDUED, va="center")

    sheet.add(
        Block("G", "One row of quantification.csv",
              "Eight of the seventy columns; areas also written in µm².",
              aspect=0.075, draw=draw_readout),
        arrows=["traits.section_traits"],
    )

    # -- Render -------------------------------------------------------------
    height = sheet.height()
    figure = plt.figure(figsize=(FIGURE_WIDTH, height), facecolor="white")
    sheet.render(figure)

    figure.text(
        MARGIN / FIGURE_WIDTH, 1 - 0.32 / height,
        "Faspy — quantifying vascular bundles in a FASGA-stained palm leaflet",
        fontsize=15.5, fontweight="bold", color=INK, va="baseline", ha="left",
    )
    figure.text(
        MARGIN / FIGURE_WIDTH, 1 - 0.56 / height,
        f"Section {key} · {full_width} × {full_height} px · every panel computed from the data",
        fontsize=9.5, color=SUBDUED, va="baseline", ha="left",
    )
    figure.text(
        1 - MARGIN / FIGURE_WIDTH, 0.17 / height,
        "github.com/paulcathelineau/Faspy",
        fontsize=8, color="#8a8a8a", va="baseline", ha="right",
    )

    output = output or (DIR_OUT / f"pipeline_{key}.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)
    print(f"Figure written -> {output}")
    return output


# ---------------------------------------------------------------------------
# Diameter distribution
# ---------------------------------------------------------------------------
# Cellpose-SAM was pretrained on images resized by a log-uniform factor of
# 0.25-4 around a 30 px mean object diameter and cropped to 256 px, which covers
# object diameters of roughly 7.5-120 px. Outside that band the model is working
# on sizes it never saw.
PRETRAINING_DIAMETERS = (7.5, 120.0)


def bundle_diameters(limit=0):
    """Equivalent diameter of every annotated bundle, in full-resolution pixels.

    Measured once at full resolution so the figure can show any working scale by
    simple multiplication: an equivalent diameter scales linearly with the image.
    """
    from .config import WORKING_SCALE
    from .datasets import load_pair, read_manifest

    rows = read_manifest()
    if limit:
        rows = rows[:limit]

    diameters, families = [], []
    for position, row in enumerate(rows, 1):
        _rgb, instances_map = load_pair(row)
        for index in np.unique(instances_map):
            if index == 0:
                continue
            area = int((instances_map == index).sum())
            # from working scale back to full resolution
            diameters.append(2 * np.sqrt(area / np.pi) / WORKING_SCALE)
            families.append(row.get("family", "?"))
        if position % 25 == 0 or position == len(rows):
            print(f"  {position}/{len(rows)} sections | {len(diameters)} bundles")
    return np.array(diameters), families


def diameter_figure(output=None, limit=0, scales=(1.0, 0.5, 0.5 * CELLPOSE_RESCALE), dpi=220):
    """Show the bundle size distribution against the range Cellpose-SAM knows.

    One panel per scale, sharing an axis, with the pretraining band shaded. It
    replaces a paragraph of argument: the working scale alone leaves most of the
    distribution outside the band, and the reduced scale brings it inside.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diameters, _families = bundle_diameters(limit)
    if diameters.size == 0:
        raise SystemExit("No annotated bundle found.")

    low, high = PRETRAINING_DIAMETERS
    figure, axes_list = plt.subplots(
        len(scales), 1, figsize=(9.5, 2.35 * len(scales) + 1.1), sharex=True,
        facecolor="white",
    )
    if len(scales) == 1:
        axes_list = [axes_list]

    edges = np.logspace(np.log10(3), np.log10(3000), 60)
    for axes, scale in zip(axes_list, scales):
        values = diameters * scale
        inside = ((values >= low) & (values <= high)).mean()

        axes.axvspan(low, high, color="#009e73", alpha=0.13, zorder=0)
        axes.hist(values, bins=edges, color="#4a6fa5", edgecolor="white", linewidth=0.4)
        axes.axvline(np.median(values), color="#d55e00", linewidth=2.0, zorder=3)

        axes.set_xscale("log")
        axes.set_ylabel("bundles")
        label = "full resolution" if scale == 1.0 else f"scale {scale:g}"
        axes.set_title(
            f"{label}  —  median {np.median(values):.0f} px, "
            f"{100 * inside:.0f} % inside the pretraining range",
            fontsize=10.5, fontweight="bold", loc="left", pad=5,
        )
        for spine in ("top", "right"):
            axes.spines[spine].set_visible(False)

    axes_list[0].text(
        np.sqrt(low * high), axes_list[0].get_ylim()[1] * 0.86,
        f"Cellpose-SAM pretraining range\n{low:g}–{high:g} px",
        ha="center", va="top", fontsize=8.5, color="#00674f", fontweight="bold",
    )
    axes_list[-1].set_xlabel("equivalent bundle diameter (px, log scale)")
    figure.suptitle(
        f"Bundle size against the range Cellpose-SAM was pretrained on  ·  "
        f"{len(diameters)} annotated bundles",
        fontsize=12.5, fontweight="bold", x=0.02, ha="left", y=0.985,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))

    output = output or (DIR_OUT / "bundle_diameters.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)
    print(f"Figure written -> {output}")
    return output


def _load_section(key: str, model_name=None, rescale=CELLPOSE_RESCALE):
    """Leaflet image, footprint and bundle instances at the working scale.

    Without a model the bundles come from the manual annotation, which lets the
    figure be rendered on a machine with no GPU.
    """
    import cv2

    _, leaflet_path, bundle_path = _source_paths(key)
    if not leaflet_path.exists():
        # A section key ends in a replicate number that is not always 1, so a
        # key typed from memory usually fails on that digit alone. Name the
        # replicates that do exist rather than leaving the caller to guess.
        palm = "_".join(key.split("_")[:2])
        siblings = sorted(path.name[: -len("_LM.png")]
                          for path in DIR_LEAFLET.glob(f"{palm}_*_LM.png"))
        hint = f" Sections held for {palm}: {', '.join(siblings)}." if siblings else ""
        raise SystemExit(f"Section {key}: leaflet image not found.{hint}")

    original = imaging.read_rgb(leaflet_path)
    scaled = imaging.clean_white_background(
        imaging.resize_rgb(original, WORKING_SCALE)
    )
    leaflet = imaging.nonblack(scaled)

    if model_name:
        from . import instances as instance_route

        model = instance_route.load_model(instance_route.checkpoint_path(model_name))
        segmented = instance_route.predict(model, scaled, rescale)
        segmented[~leaflet] = 0
        return scaled, leaflet, segmented, f"model {model_name}", original

    if not bundle_path.exists():
        raise SystemExit(f"Section {key}: no annotation, and no --model given.")
    bundle_rgb = cv2.resize(
        imaging.read_rgb(bundle_path), (scaled.shape[1], scaled.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    annotated = imaging.connected_instances(
        imaging.nonblack(bundle_rgb) & leaflet, MIN_BUNDLE_AREA
    )
    return scaled, leaflet, annotated, "annotation", original


def _scale_bar(axes, length_um, pixel_um, colour="white"):
    """Scale bar in the lower right of an image panel, in micrometres."""
    left, right = axes.get_xlim()
    bottom, top = axes.get_ylim()
    width = length_um / pixel_um
    x = right - 0.035 * (right - left) - width
    y = bottom - 0.09 * (bottom - top)
    axes.plot([x, x + width], [y, y], color=colour, lw=3.5,
              solid_capstyle="butt", zorder=7)
    axes.text(x + width / 2, y - 0.03 * (bottom - top), f"{length_um:.0f} um",
              color=colour, fontsize=8.5, fontweight="bold", ha="center",
              va="bottom", zorder=7)


def traits_figure(key: str, output=None, model_name=None, rescale=CELLPOSE_RESCALE, dpi=170):
    """Draw, on a real section, what every trait actually measures.

    A column of twenty numbers cannot be checked by eye. This figure puts them
    back on the image they came from -- the leaflet outline, the bundles and
    their centroids, the mid-plane, the lumina, the vessels -- so that a wrong
    trait shows up as a wrong picture. It is how the mid-plane estimator was
    caught reading 300 um on a section 205 um thick.
    """
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    from . import traits as trait_route
    from .config import LUMEN_MIN_DIAMETER_UM, VESSEL_MIN_DIAMETER_UM
    from .quantify import lumen_at_full_resolution, to_working_scale

    pixel_um = trait_route.PIXEL_UM
    scaled, leaflet, segmented, source, original = _load_section(key, model_name, rescale)
    fine = lumen_at_full_resolution(original, segmented > 0, leaflet)
    lumen = to_working_scale(fine["lumen"], leaflet.shape)
    measured = trait_route.section_traits(leaflet, segmented, lumen, fine=fine)

    # Two possible references for placing a bundle within the thickness: a
    # global straight line through the centroid, which mixes depth with the
    # curvature taken on at mounting, and the leaflet's mid-plane, which follows
    # the section and therefore reports depth alone. The second is what the
    # traits use; the first is drawn only for comparison.
    offset, rotated_leaflet, _, mid_line = trait_route.mid_plane(leaflet, segmented > 0)
    matrix, size = trait_route.straighten_matrix(leaflet)

    def straighten(image, nearest=False):
        return cv2.warpAffine(
            image, matrix, size,
            flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        )

    rotated = straighten(segmented.astype(np.uint16), True).astype(np.int32)
    bundles = []
    for index in (i for i in np.unique(segmented) if i > 0):
        rows, columns = np.nonzero(rotated == index)
        if not rows.size:
            continue
        bundles.append({
            "id": int(index),
            "area": int((segmented == index).sum()),
            "distance": float(offset[rows, columns].mean()),
            "y": rows.mean(),
            "x": columns.mean(),
        })
    if not bundles:
        raise SystemExit(f"Section {key}: no bundle to draw.")

    # Bundles are not coloured arbitrarily: the hue encodes their AREA, which
    # makes the spread of sizes readable at a glance. The scale is logarithmic
    # because one bundle can be twenty times the area of the rest, and a linear
    # ramp would then flatten every small one into the same pale tint. No blue
    # anywhere in the ramp, blue being reserved for the lumen.
    areas_um2 = np.array([b["area"] for b in bundles], dtype=np.float64) * pixel_um ** 2
    norm = mcolors.LogNorm(vmin=float(areas_um2.min()), vmax=float(areas_um2.max()))
    cmap = plt.get_cmap("YlOrRd")

    view = (straighten(scaled) * 0.55).astype(np.uint8)
    for bundle, area in zip(bundles, areas_um2):
        view[rotated == bundle["id"]] = np.array(cmap(norm(area))[:3]) * 255
    view[straighten(lumen.astype(np.uint8), True) > 0] = (0, 90, 255)

    footprint = straighten(leaflet.astype(np.uint8), True) > 0
    contours, _ = cv2.findContours(
        footprint.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(view, contours, -1, (255, 255, 255), 2)

    rows, columns = np.nonzero(footprint)
    top, bottom = max(0, rows.min() - 8), rows.max() + 8
    left, right = max(0, columns.min() - 8), columns.max() + 8
    view = view[top:bottom, left:right]
    axis_y, axis_x = rows.mean() - top, columns.mean() - left

    # Bounded to the columns the leaflet occupies: elsewhere the mid-line is
    # only an interpolation, and drawing it would run the axis outside the
    # section.
    span = np.nonzero(rotated_leaflet.any(axis=0))[0]
    inside = slice(span[0], span[-1] + 1)
    curve = (np.arange(mid_line.size)[inside] - left, mid_line[inside] - top)

    figure = plt.figure(figsize=(14, 12.6), facecolor="white")
    grid = figure.add_gridspec(
        4, 2, height_ratios=[1.1, 0.9, 1.5, 1.1], hspace=0.48, wspace=0.22,
        left=0.05, right=0.945, top=0.92, bottom=0.05,
    )
    label = dict(fontsize=8.5, fontweight="bold", va="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

    # -- A: the measured section -------------------------------------------
    axes = figure.add_subplot(grid[0, :])
    axes.imshow(view, aspect="auto")
    axes.set_ylim(view.shape[0], 0)

    # One dot per bundle, at its centroid: both the distance to the mid-plane
    # and the nearest-neighbour spacing are measured from this point.
    axes.scatter([b["x"] - left for b in bundles], [b["y"] - top for b in bundles],
                 s=26, c="white", edgecolor=INK, linewidth=1.1, zorder=6)
    axes.axhline(axis_y, color="#8a8a8a", lw=1.4, ls=":")
    axes.text(view.shape[1] * 0.01, axis_y, " global straight axis ",
              color="#6a6a6a", ha="left", **label)
    axes.plot(curve[0], curve[1], color="#d55e00", lw=2.2)
    anchor = curve[0].size // 5
    axes.annotate("mid-plane", (curve[0][anchor], curve[1][anchor]),
                  textcoords="offset points", xytext=(0, -16), ha="center",
                  color="#d55e00", **label)
    axes.plot(axis_x, axis_y, marker="P", ms=12, color=INK, mec="white",
              mew=1.5, zorder=6)
    axes.annotate("section centroid", (axis_x, axis_y), textcoords="offset points",
                  xytext=(12, 12), color=INK, **label)

    bar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                          fraction=0.016, pad=0.008)
    bar.set_label("bundle area (um2)", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    axes.set_xticks([])
    axes.set_yticks([])
    _scale_bar(axes, 200, pixel_um)
    bundle_share = 100 * (segmented > 0).sum() / max(leaflet.sum(), 1)
    axes.set_title(
        f"A   Measured section - {len(bundles)} bundles coloured by area, "
        f"white dots their centroids, lumen in blue",
        fontsize=TITLE_SIZE, fontweight="bold", loc="left",
    )
    axes.set_xlabel(
        f"thickness {measured['leaflet_thickness_um']:.0f} um  ·  "
        f"bundles {bundle_share:.1f} % of the leaflet  ·  "
        f"median spacing {measured['bundle_spacing_um']:.0f} um  ·  "
        f"depth below the epidermis {measured['bundle_depth_median_um']:.0f} um",
        fontsize=9,
    )

    # -- A': the field that weights the stiffness ---------------------------
    # A second moment cannot be read off a number. This map shows where it comes
    # from: every pixel is weighted by the SQUARE of its distance to the
    # mid-plane, so the pale zones count for enormously more than the dark ones,
    # and whether the bundles sit where it matters becomes visible at once.
    axes = figure.add_subplot(grid[1, :])
    field = offset[top:bottom, left:right]
    image = axes.imshow(
        np.where(footprint[top:bottom, left:right], field * pixel_um, np.nan),
        cmap="magma", aspect="auto",
    )
    contours, _ = cv2.findContours(
        (rotated[top:bottom, left:right] > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    for contour in contours:
        axes.plot(contour[:, 0, 0], contour[:, 0, 1], color="#00e5c0", lw=1.2)

    axes.set_ylim(field.shape[0], 0)
    axes.set_xticks([])
    axes.set_yticks([])
    bar = figure.colorbar(image, ax=axes, fraction=0.018, pad=0.008)
    bar.set_label("distance to the mid-plane (um)", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    _scale_bar(axes, 200, pixel_um)
    axes.set_title("A'   What weights the stiffness - bundle outlines in green",
                   fontsize=TITLE_SIZE, fontweight="bold", loc="left")
    axes.set_xlabel(
        "dark = on the mid-plane, contributes almost nothing  ·  "
        "pale = against the epidermis, contributes as the square of the distance",
        fontsize=9,
    )

    # -- B: lumina and vessels ----------------------------------------------
    axes = figure.add_subplot(grid[2, 0])
    largest = max(bundles, key=lambda b: b["area"])
    selected = segmented == largest["id"]
    rows, columns = np.nonzero(selected)
    pad = int(0.35 * max(np.ptp(rows), np.ptp(columns))) + 10
    y0, y1 = max(0, rows.min() - pad), min(scaled.shape[0], rows.max() + pad)
    x0, x1 = max(0, columns.min() - pad), min(scaled.shape[1], columns.max() + pad)

    crop = scaled[y0:y1, x0:x1].copy()
    crop_lumen = lumen[y0:y1, x0:x1]
    crop[crop_lumen] = (0, 90, 255)
    contours, _ = cv2.findContours(
        selected[y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(crop, contours, -1, (0, 0, 0), 2)
    axes.imshow(crop)

    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        crop_lumen.astype(np.uint8), 8
    )
    n_vessels = 0
    for i in range(1, count):
        diameter = 2 * np.sqrt(stats[i, cv2.CC_STAT_AREA] / np.pi) * pixel_um
        if diameter >= VESSEL_MIN_DIAMETER_UM:
            axes.add_patch(Circle(centroids[i], diameter / pixel_um / 2 + 3,
                                  fill=False, ec="#00c000", lw=2))
            n_vessels += 1

    axes.set_xticks([])
    axes.set_yticks([])
    _scale_bar(axes, 20, pixel_um)
    axes.set_title("B   Lumina and vessels", fontsize=TITLE_SIZE,
                   fontweight="bold", loc="left")
    axes.set_xlabel(
        f"green circles = lumina at least {VESSEL_MIN_DIAMETER_UM:.0f} um "
        f"({n_vessels} here)  ·  detection floor {LUMEN_MIN_DIAMETER_UM:.0f} um",
        fontsize=9,
    )

    # -- C: why position matters --------------------------------------------
    axes = figure.add_subplot(grid[2, 1])
    distance = np.array([b["distance"] for b in bundles]) * pixel_um
    contribution = areas_um2 * distance ** 2
    axes.scatter(distance, areas_um2,
                 s=40 + 260 * contribution / contribution.max(),
                 c="#4a6fa5", edgecolor="white", zorder=3)
    axes.set_xlabel("distance from the bundle centroid to the mid-plane (um)", fontsize=9)
    axes.set_ylabel("bundle area (um2)", fontsize=9)
    axes.grid(alpha=0.25)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.set_title("C   Why position matters", fontsize=TITLE_SIZE,
                   fontweight="bold", loc="left", pad=20)
    axes.text(0.0, 1.012,
              "dot size = contribution to the second moment, that is "
              "area x distance squared",
              transform=axes.transAxes, ha="left", va="bottom",
              fontsize=CAPTION_SIZE, color=SUBDUED)

    # -- D: the numbers ------------------------------------------------------
    axes = figure.add_subplot(grid[3, :])
    axes.axis("off")
    wall = float(measured["wall_over_bundle"])
    blocks = [
        ("Quantity", [
            ("bundles", f"{len(bundles)}"),
            ("bundles / leaflet", f"{bundle_share:.2f} %"),
            ("wall / bundle", f"{100 * wall:.1f} %"),
            ("lumen / bundle", f"{100 * (1 - wall):.1f} %"),
        ]),
        ("Geometry", [
            ("leaflet thickness", f"{measured['leaflet_thickness_um']:.0f} um"),
            ("I leaflet", f"{measured['I_leaflet_um4']:.3g} um4"),
            ("I bundles", f"{measured['I_bundle_um4']:.3g} um4"),
            ("bundle share, mid-plane",
             f"{100 * float(measured['I_bundle_share_flat']):.1f} %"),
        ]),
        ("Organisation", [
            ("median diameter", f"{measured['bundle_diameter_median_um']:.0f} um"),
            ("relative depth", f"{float(measured['bundle_depth_relative']):.2f}"),
            ("spacing", f"{measured['bundle_spacing_um']:.0f} um"),
            ("spread of sizes", f"{float(measured['bundle_area_cv']):.2f}"),
        ]),
        ("Lumina", [
            ("objects", f"{measured['n_lumen']}"),
            ("median diameter", f"{measured['lumen_diameter_median_um']:.1f} um"),
            ("90th percentile", f"{measured['lumen_diameter_p90_um']:.1f} um"),
            ("vessels", f"{measured['n_vessels']}"),
        ]),
    ]
    for column, (heading, entries) in enumerate(blocks):
        x = 0.02 + column * 0.25
        axes.text(x, 0.92, heading, fontsize=10.5, fontweight="bold",
                  color="#1f3b57", transform=axes.transAxes)
        for row, (name, value) in enumerate(entries):
            y = 0.72 - row * 0.19
            axes.text(x, y, name, fontsize=9.5, color=SUBDUED, transform=axes.transAxes)
            axes.text(x + 0.21, y, value, fontsize=9.5, fontweight="bold",
                      ha="right", transform=axes.transAxes)

    figure.suptitle(f"Anatomical traits measured - {key}   ({source})",
                    fontsize=14, fontweight="bold", x=0.05, ha="left", y=0.965)

    output = output or DIR_OUT / f"traits_{key}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)
    print(f"-> {output}")
    return output
