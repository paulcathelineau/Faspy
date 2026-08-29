"""Evaluation metrics, shared by the semantic and the instance route.

Both routes are judged on exactly the same numbers, which is what makes them
comparable. Pixel IoU is reported but is not the criterion: for small objects
with a diffuse border it moves very little while the quantities that matter to
the biology — how many bundles, and how much area they cover — move a lot.

The accumulator collects per-section results and turns them into a single
summary at the end, so a caller only ever writes a loop over its validation
sections.
"""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from .config import MATCH_IOU_THRESHOLD, N_CLASSES


def _mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


# ---------------------------------------------------------------------------
# Object matching
# ---------------------------------------------------------------------------
def match_instances(truth: np.ndarray, prediction: np.ndarray, iou_threshold=MATCH_IOU_THRESHOLD):
    """Greedily pair predicted objects with annotated ones.

    Each annotated object takes the unclaimed prediction it overlaps best, and
    counts as detected when that overlap reaches ``iou_threshold``. Returns
    ``(n_truth, n_predicted, true positives, false positives, false negatives)``.
    """
    truth_ids = [i for i in np.unique(truth) if i > 0]
    predicted_ids = [j for j in np.unique(prediction) if j > 0]

    claimed, true_positives = set(), 0
    for index in truth_ids:
        selected = truth == index
        area = int(selected.sum())
        best_iou, best_id = 0.0, -1
        for candidate in np.unique(prediction[selected]):
            if candidate == 0 or candidate in claimed:
                continue
            other = prediction == candidate
            intersection = int((selected & other).sum())
            union = area + int(other.sum()) - intersection
            iou = intersection / union if union else 0.0
            if iou > best_iou:
                best_iou, best_id = iou, candidate
        if best_iou >= iou_threshold:
            true_positives += 1
            claimed.add(best_id)

    return (
        len(truth_ids),
        len(predicted_ids),
        true_positives,
        len(predicted_ids) - len(claimed),
        len(truth_ids) - true_positives,
    )


def shape_agreement(truth: np.ndarray, prediction: np.ndarray):
    """Per matched object, return containment of the prediction and the area ratio.

    Containment near 1.0 means the prediction sits entirely inside the annotated
    object: a concentric, smaller object rather than a noisy boundary. A low
    spread of the area ratio means a constant scale error, which a single
    multiplicative factor can correct; a high spread means it cannot.
    """
    containment, ratio = [], []
    for index in np.unique(truth):
        if index == 0:
            continue
        selected = truth == index
        area = int(selected.sum())
        if area == 0:
            continue
        candidates = [j for j in np.unique(prediction[selected]) if j > 0]
        if not candidates:
            continue
        best = max(candidates, key=lambda j: int((selected & (prediction == j)).sum()))
        other = prediction == best
        predicted_area = int(other.sum())
        if predicted_area > 0:
            containment.append(int((selected & other).sum()) / predicted_area)
            ratio.append(predicted_area / area)
    return containment, ratio


def detection_by_size(truth: np.ndarray, prediction: np.ndarray, iou_threshold=MATCH_IOU_THRESHOLD):
    """Per annotated object, return ``(area, detected)``.

    Reported by area quartile. If recall collapses on the largest quartile, the
    area deficit comes from missing whole bundles, not from tight outlines, and
    the two call for opposite fixes.
    """
    out = []
    for index in np.unique(truth):
        if index == 0:
            continue
        selected = truth == index
        area = int(selected.sum())
        if area == 0:
            continue
        best = 0.0
        for candidate in np.unique(prediction[selected]):
            if candidate == 0:
                continue
            other = prediction == candidate
            intersection = int((selected & other).sum())
            union = area + int(other.sum()) - intersection
            best = max(best, intersection / union if union else 0.0)
        out.append((area, best >= iou_threshold))
    return out


# ---------------------------------------------------------------------------
# Semantic confusion matrix
# ---------------------------------------------------------------------------
def confusion_matrix(truth: np.ndarray, prediction: np.ndarray, ignore_index=255):
    keep = truth != ignore_index
    counts = np.bincount(
        N_CLASSES * truth[keep].astype(np.int64) + prediction[keep],
        minlength=N_CLASSES ** 2,
    )
    return counts.reshape(N_CLASSES, N_CLASSES)


def metrics_from_confusion(matrix: np.ndarray) -> dict:
    """Per-class IoU, Dice, precision and recall, plus mIoU and pixel accuracy."""
    matrix = matrix.astype(np.float64)
    out = {}
    for index in range(N_CLASSES):
        true_positives = matrix[index, index]
        false_positives = matrix[:, index].sum() - true_positives
        false_negatives = matrix[index, :].sum() - true_positives
        denominator = true_positives + false_positives + false_negatives
        out[index] = {
            "IoU": true_positives / denominator if denominator else float("nan"),
            "Dice": (
                2 * true_positives / (2 * true_positives + false_positives + false_negatives)
                if (2 * true_positives + false_positives + false_negatives)
                else float("nan")
            ),
            "precision": (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives)
                else float("nan")
            ),
            "recall": (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives)
                else float("nan")
            ),
            "support": matrix[index, :].sum(),
        }
    out["pixel_accuracy"] = np.diag(matrix).sum() / matrix.sum() if matrix.sum() else float("nan")
    out["mIoU"] = float(np.nanmean([out[c]["IoU"] for c in range(N_CLASSES)]))
    return out


# ---------------------------------------------------------------------------
# Quantification accumulator
# ---------------------------------------------------------------------------
class Quantification:
    """Collect per-section results and summarise them once at the end."""

    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.truth_area = 0
        self.predicted_area = 0
        self.iou = []
        self.area_signed = []
        self.count_absolute = []
        self.count_relative = []
        self.count_signed = []
        self.containment = []
        self.area_ratio = []
        self.by_size = []
        self.by_family = defaultdict(lambda: {"truth_area": 0, "predicted_area": 0})

    def add(self, truth: np.ndarray, prediction: np.ndarray, family: str = "?", shapes=True):
        """Accumulate one section, given two instance maps."""
        truth_mask, predicted_mask = truth > 0, prediction > 0
        intersection = int((truth_mask & predicted_mask).sum())
        union = int((truth_mask | predicted_mask).sum())
        self.iou.append(intersection / union if union else 1.0)

        truth_area = int(truth_mask.sum())
        predicted_area = int(predicted_mask.sum())
        self.truth_area += truth_area
        self.predicted_area += predicted_area
        self.by_family[family]["truth_area"] += truth_area
        self.by_family[family]["predicted_area"] += predicted_area
        if truth_area > 0:
            self.area_signed.append((predicted_area - truth_area) / truth_area)

        n_truth, n_predicted, tp, fp, fn = match_instances(truth, prediction)
        self.true_positives += tp
        self.false_positives += fp
        self.false_negatives += fn
        self.count_absolute.append(abs(n_predicted - n_truth))
        if n_truth > 0:
            self.count_relative.append(abs(n_predicted - n_truth) / n_truth)
            self.count_signed.append((n_predicted - n_truth) / n_truth)

        if shapes:
            contained, ratio = shape_agreement(truth, prediction)
            self.containment += contained
            self.area_ratio += ratio
            self.by_size += detection_by_size(truth, prediction)

    # -- derived quantities -------------------------------------------------
    def _quartiles(self) -> dict:
        if not self.by_size:
            return {}
        areas = np.array([area for area, _ in self.by_size], dtype=float)
        detected = np.array([hit for _, hit in self.by_size], dtype=bool)
        cuts = np.percentile(areas, [25, 50, 75])
        bounds = [(0, cuts[0]), (cuts[0], cuts[1]), (cuts[1], cuts[2]), (cuts[2], np.inf)]
        total = areas.sum()

        out = {}
        for quartile, (low, high) in enumerate(bounds, 1):
            selected = (areas >= low) & (areas < high)
            out[f"recall_Q{quartile}"] = (
                float(detected[selected].mean()) if selected.sum() else float("nan")
            )
            out[f"missed_area_Q{quartile}"] = (
                float(areas[selected & ~detected].sum() / total) if total else float("nan")
            )
        return out

    def summary(self) -> dict:
        precision = (
            self.true_positives / (self.true_positives + self.false_positives)
            if (self.true_positives + self.false_positives)
            else float("nan")
        )
        recall = (
            self.true_positives / (self.true_positives + self.false_negatives)
            if (self.true_positives + self.false_negatives)
            else float("nan")
        )
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")

        # Aggregate factor: total annotated area over total predicted area. It is
        # the figure reported in the paper. The median of per-section ratios is
        # given alongside because the aggregate is dominated by the largest
        # sections and is a poorer estimate of the typical bias.
        aggregate = self.truth_area / self.predicted_area if self.predicted_area else float("nan")
        per_section = [1.0 / (1.0 + s) for s in self.area_signed if (1.0 + s) > 1e-6]
        median = float(np.median(per_section)) if per_section else float("nan")

        def calibrated(factor):
            return _mean([abs(factor * (1 + s) - 1) for s in self.area_signed])

        ratio_cv = (
            float(np.std(self.area_ratio) / np.mean(self.area_ratio))
            if self.area_ratio and np.mean(self.area_ratio) > 0
            else float("nan")
        )

        # Average precision as the Cellpose papers define it, which is not the
        # COCO area-under-the-curve quantity: AP = TP / (TP + FP + FN) at a
        # fixed IoU threshold. It is a deterministic transform of the F1 above,
        # AP = F1 / (2 - F1), and is reported because it is what the instance
        # segmentation literature quotes.
        denominator = self.true_positives + self.false_positives + self.false_negatives
        average_precision = self.true_positives / denominator if denominator else float("nan")

        return {
            "bundle_IoU": _mean(self.iou),
            "object_precision": precision,
            "object_recall": recall,
            "object_F1": f1,
            "object_AP": average_precision,
            "count_MAE": _mean(self.count_absolute),
            "count_error": _mean(self.count_relative),
            "count_bias": _mean(self.count_signed),
            "area_bias": _mean(self.area_signed),
            "area_error": _mean([abs(s) for s in self.area_signed]),
            "calibration_factor": aggregate,
            "area_error_calibrated": calibrated(aggregate),
            "calibration_factor_median": median,
            "area_error_calibrated_median": calibrated(median),
            "containment_median": (
                float(np.median(self.containment)) if self.containment else float("nan")
            ),
            "area_ratio_median": (
                float(np.median(self.area_ratio)) if self.area_ratio else float("nan")
            ),
            "area_ratio_cv": ratio_cv,
            **self._quartiles(),
        }

    def family_calibration(self) -> list[tuple[str, float, float]]:
        out = []
        for family, areas in sorted(self.by_family.items()):
            factor = (
                areas["truth_area"] / areas["predicted_area"]
                if areas["predicted_area"]
                else float("nan")
            )
            bias = (
                areas["predicted_area"] / areas["truth_area"] - 1
                if areas["truth_area"]
                else float("nan")
            )
            out.append((family, factor, bias))
        return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_metrics_csv(path, summary: dict, families=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for name, value in summary.items():
            writer.writerow([name, round(value, 4) if isinstance(value, float) else value])
        if families:
            writer.writerow([])
            writer.writerow(["family", "calibration_factor", "area_bias"])
            for family, factor, bias in families:
                writer.writerow([family, round(factor, 4), round(bias, 4)])


def format_summary(summary: dict) -> str:
    """One compact block, used by every command that evaluates something."""
    lines = [
        f"  AP@0.5 {summary['object_AP']:.3f} | object F1 {summary['object_F1']:.3f} "
        f"(precision {summary['object_precision']:.3f}, recall {summary['object_recall']:.3f}) "
        f"| bundle IoU {summary['bundle_IoU']:.3f}",
        f"  count error {100 * summary['count_error']:.1f} % "
        f"(bias {100 * summary['count_bias']:+.1f} %, MAE {summary['count_MAE']:.2f} bundles)",
        f"  area bias {100 * summary['area_bias']:+.1f} % | "
        f"calibrated error {100 * summary['area_error_calibrated']:.1f} % "
        f"(factor x{summary['calibration_factor']:.3f})",
    ]
    if not np.isnan(summary.get("containment_median", float("nan"))):
        lines.append(
            f"  outline: containment {summary['containment_median']:.3f} | "
            f"area ratio {summary['area_ratio_median']:.3f} "
            f"(CV {summary['area_ratio_cv']:.3f})"
        )
    if "recall_Q1" in summary:
        recall = " ".join(f"Q{q} {summary[f'recall_Q{q}']:.2f}" for q in (1, 2, 3, 4))
        missed = " ".join(f"Q{q} {100 * summary[f'missed_area_Q{q}']:.0f} %" for q in (1, 2, 3, 4))
        lines.append(f"  recall by size (Q1 smallest to Q4 largest): {recall}")
        lines.append(f"  area lost to missed objects:                {missed}")
    return "\n".join(lines)
