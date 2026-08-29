"""Cross-validation for both routes, reporting one common set of metrics.

Folds are stratified by individual and drawn by palm, so no individual appears in both halves. Every fold trains
from scratch on the other folds and is scored on sections the model has never
seen. After the folds, a final model is trained on the whole annotated set and
saved for production use.
"""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from . import imaging, instances, metrics, semantic
from .config import (
    CELLPOSE_EPOCHS,
    CELLPOSE_RESCALE,
    CELLPROB_THRESHOLD,
    CLASSES,
    DIR_EVAL_INSTANCE,
    DIR_EVAL_SEMANTIC,
    DIR_UNET_MODELS,
    EPOCHS,
    FLOW_THRESHOLD,
    INFER_STRIDE,
    MIN_BUNDLE_AREA,
    N_CLASSES,
    N_FOLDS,
    PATCH,
    SEED,
    UNET_CHECKPOINT,
    WORKING_SCALE,
)
from .datasets import (
    load_pair,
    read_manifest,
    resolve,
    stratified_kfold,
    train_validation_split,
)


# ---------------------------------------------------------------------------
# Instance route
# ---------------------------------------------------------------------------
def cross_validate_instances(
    n_folds=N_FOLDS,
    epochs=CELLPOSE_EPOCHS,
    rescale=CELLPOSE_RESCALE,
    cellprob=CELLPROB_THRESHOLD,
    flow=FLOW_THRESHOLD,
    train_final=True,
):
    rows = read_manifest()
    folds = stratified_kfold(rows, n_folds, SEED)
    gpu = instances.gpu_available()

    print(
        f"Cellpose-SAM | GPU={gpu} | {n_folds} folds | {len(rows)} sections "
        f"| rescale={rescale} | min area={MIN_BUNDLE_AREA} px"
    )
    if rescale == 1.0:
        print(
            "  Warning: at rescale 1.0 the largest bundles exceed the 256 px "
            "training crop and are never learnt. Use 0.35."
        )

    accumulator = metrics.Quantification()
    for index in range(n_folds):
        training, validation = train_validation_split(rows, folds, index)
        print(f"\n=== Fold {index + 1}/{n_folds}: train {len(training)} / validate {len(validation)} ===")
        model = instances.train(training, f"cpsam_fold{index}", epochs, rescale, gpu)
        for row in validation:
            rgb, truth = load_pair(row)
            predicted = instances.predict(model, rgb, rescale, cellprob, flow)
            accumulator.add(truth, predicted, row.get("family", "?"))
        print(f"  cumulative bundle IoU {np.mean(accumulator.iou):.3f}")

    summary = accumulator.summary()
    metrics.write_metrics_csv(
        DIR_EVAL_INSTANCE / "instance_metrics.csv", summary, accumulator.family_calibration()
    )
    _write_instance_summary(summary, rows, folds, rescale, epochs)
    print(f"\nResults -> {DIR_EVAL_INSTANCE}")
    print(metrics.format_summary(summary))

    if train_final:
        print("\n=== Final model on every annotated section ===")
        instances.train(rows, "cpsam_final", epochs, rescale, gpu)
        print(f"Final model: {instances.checkpoint_path('cpsam_final')}")
    return summary


def zero_shot_baseline(
    rescale=CELLPOSE_RESCALE,
    cellprob=CELLPROB_THRESHOLD,
    flow=FLOW_THRESHOLD,
    limit=0,
):
    """Score the published Cellpose-SAM checkpoint without any fine-tuning.

    This is the reference the fine-tuned model has to beat, and it answers the
    first question a reader asks: was the training necessary at all? No fold
    structure is needed. The pretrained model has never seen any of these
    sections, so the whole annotated set is valid test data.

    Run it at more than one scale. Comparing zero-shot at 1.0 with zero-shot at
    0.35 separates what rescaling contributes from what fine-tuning contributes,
    which the fine-tuned numbers alone cannot do.
    """
    rows = read_manifest()
    if limit:
        rows = rows[:limit]

    print(
        f"Zero-shot Cellpose-SAM | no fine-tuning | {len(rows)} sections "
        f"| rescale={rescale} | cellprob={cellprob} | flow={flow}"
    )
    model = instances.load_model(None)   # published checkpoint

    accumulator = metrics.Quantification()
    for position, row in enumerate(rows, 1):
        rgb, truth = load_pair(row)
        predicted = instances.predict(model, rgb, rescale, cellprob, flow)
        accumulator.add(truth, predicted, row.get("family", "?"))
        if position % 20 == 0 or position == len(rows):
            print(f"  {position}/{len(rows)} sections")

    summary = accumulator.summary()
    summary["rescale"] = rescale
    tag = str(rescale).replace(".", "")
    metrics.write_metrics_csv(
        DIR_EVAL_INSTANCE / f"zero_shot_{tag}.csv", summary, accumulator.family_calibration()
    )
    print(f"\nZero-shot, rescale {rescale} -> {DIR_EVAL_INSTANCE / f'zero_shot_{tag}.csv'}")
    print(metrics.format_summary(summary))
    return summary


def _write_instance_summary(summary, rows, folds, rescale, epochs):
    families = defaultdict(int)
    for row in rows:
        families[row.get("family", "?")] += 1

    lines = [
        "# Instance segmentation of vascular bundles (Cellpose-SAM)\n",
        "## Dataset",
        f"- Annotated sections: **{len(rows)}** ({dict(sorted(families.items()))})",
        f"- Working scale {WORKING_SCALE}, further reduced by {rescale} for this route",
        f"  (effective scale {WORKING_SCALE * rescale:.3f} of the original)",
        f"- Minimum bundle area: {MIN_BUNDLE_AREA} px at the working scale\n",
        "## Protocol",
        f"- {len(folds)}-fold cross-validation, stratified by individual",
        f"- Validation sections per fold: {[len(fold) for fold in folds]}",
        f"- {epochs} epochs per fold, one model trained from the pretrained "
        f"checkpoint each time\n",
        "## Results",
        f"- **AP at IoU 0.5**: {summary['object_AP']:.3f} "
        f"(TP / (TP + FP + FN), as defined in the Cellpose papers)",
        f"- **Object F1**: {summary['object_F1']:.3f} "
        f"(precision {summary['object_precision']:.3f}, recall {summary['object_recall']:.3f})",
        f"- **Count error**: {100 * summary['count_error']:.1f} % "
        f"(bias {100 * summary['count_bias']:+.1f} %, MAE {summary['count_MAE']:.2f} bundles)",
        f"- **Area error after calibration**: {100 * summary['area_error_calibrated']:.1f} % "
        f"(raw bias {100 * summary['area_bias']:+.1f} %, factor "
        f"x{summary['calibration_factor']:.3f})",
        f"- Bundle IoU: {summary['bundle_IoU']:.3f}",
    ]
    if "recall_Q1" in summary:
        lines += [
            "\n## Recall by bundle size",
            "| | Q1 smallest | Q2 | Q3 | Q4 largest |",
            "|---|---|---|---|---|",
            "| Recall | " + " | ".join(f"{summary[f'recall_Q{q}']:.2f}" for q in (1, 2, 3, 4)) + " |",
            "| Share of area lost | "
            + " | ".join(f"{100 * summary[f'missed_area_Q{q}']:.0f} %" for q in (1, 2, 3, 4))
            + " |",
        ]
    DIR_EVAL_INSTANCE.mkdir(parents=True, exist_ok=True)
    (DIR_EVAL_INSTANCE / "instance_summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Semantic route
# ---------------------------------------------------------------------------
def cross_validate_semantic(n_folds=N_FOLDS, epochs=EPOCHS, train_final=True):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = read_manifest()
    folds = stratified_kfold(rows, n_folds, SEED)
    print(f"U-Net | device={device} | {n_folds} folds | {len(rows)} sections")

    per_fold = []
    total = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    per_family = defaultdict(lambda: np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64))
    accumulator = metrics.Quantification()

    for index in range(n_folds):
        training, validation = train_validation_split(rows, folds, index)
        print(f"\n=== Fold {index + 1}/{n_folds}: train {len(training)} / validate {len(validation)} ===")
        model = semantic.train(training, device, epochs)
        model.eval()

        fold_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        for row in validation:
            rgb = imaging.read_bgr_as_rgb(resolve(row, "input"))
            truth = imaging.read_grey(resolve(row, "mask"))
            predicted = semantic.postprocess(semantic.predict(model, rgb, device))

            matrix = metrics.confusion_matrix(truth, predicted)
            fold_matrix += matrix
            total += matrix
            per_family[row.get("family", "?")] += matrix

            from .config import BUNDLE_CLASS

            accumulator.add(
                imaging.connected_instances(truth == BUNDLE_CLASS, MIN_BUNDLE_AREA),
                imaging.connected_instances(predicted == BUNDLE_CLASS, MIN_BUNDLE_AREA),
                row.get("family", "?"),
                shapes=False,
            )

        fold_metrics = metrics.metrics_from_confusion(fold_matrix)
        per_fold.append(fold_metrics)
        print(
            f"  mIoU {fold_metrics['mIoU']:.3f} | "
            + " ".join(f"{CLASSES[c]} {fold_metrics[c]['IoU']:.3f}" for c in range(N_CLASSES))
        )

    summary = accumulator.summary()
    _write_semantic_reports(rows, folds, per_fold, total, per_family, summary, epochs)
    print(f"\nResults -> {DIR_EVAL_SEMANTIC}")
    print(metrics.format_summary(summary))

    if train_final:
        print("\n=== Final model on every annotated section ===")
        final = semantic.train(rows, device, epochs)
        DIR_UNET_MODELS.mkdir(parents=True, exist_ok=True)
        torch.save({"model": final.state_dict(), "epochs": epochs}, UNET_CHECKPOINT)
        print(f"Final model: {UNET_CHECKPOINT}")
    return summary


def _mean_std(per_fold, metric, class_index=None):
    values = [
        (fold[metric] if class_index is None else fold[class_index][metric]) for fold in per_fold
    ]
    values = [value for value in values if value == value]
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def _write_semantic_reports(rows, folds, per_fold, total, per_family, summary, epochs):
    DIR_EVAL_SEMANTIC.mkdir(parents=True, exist_ok=True)

    with open(DIR_EVAL_SEMANTIC / "semantic_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["class", "IoU_mean", "IoU_std", "Dice_mean", "Dice_std",
             "precision_mean", "precision_std", "recall_mean", "recall_std"]
        )
        for index in range(N_CLASSES):
            row = [CLASSES[index]]
            for name in ("IoU", "Dice", "precision", "recall"):
                row += [round(value, 4) for value in _mean_std(per_fold, name, index)]
            writer.writerow(row)
        writer.writerow([])
        for name in ("mIoU", "pixel_accuracy"):
            mean, std = _mean_std(per_fold, name)
            writer.writerow([name, round(mean, 4), "std", round(std, 4)])

    with open(DIR_EVAL_SEMANTIC / "semantic_per_family.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "class", "IoU", "Dice", "precision", "recall", "support_px"])
        for family, matrix in sorted(per_family.items()):
            family_metrics = metrics.metrics_from_confusion(matrix)
            for index in range(N_CLASSES):
                writer.writerow(
                    [family, CLASSES[index]]
                    + [round(family_metrics[index][name], 4)
                       for name in ("IoU", "Dice", "precision", "recall")]
                    + [int(family_metrics[index]["support"])]
                )

    normalised = total / total.sum(axis=1, keepdims=True).clip(min=1)
    with open(DIR_EVAL_SEMANTIC / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + [f"predicted_{CLASSES[c]}" for c in range(N_CLASSES)])
        for index in range(N_CLASSES):
            writer.writerow(
                [f"true_{CLASSES[index]}"]
                + [round(normalised[index, j], 4) for j in range(N_CLASSES)]
            )
    _save_confusion_figure(normalised)
    metrics.write_metrics_csv(DIR_EVAL_SEMANTIC / "quantification_metrics.csv", summary)

    mean_iou, std_iou = _mean_std(per_fold, "mIoU")
    mean_accuracy, std_accuracy = _mean_std(per_fold, "pixel_accuracy")
    lines = [
        "# Semantic segmentation (U-Net)\n",
        "## Protocol",
        f"- {len(folds)}-fold cross-validation, stratified by individual",
        f"- {len(rows)} annotated sections, working scale {WORKING_SCALE}",
        f"- Sliding window {PATCH} px, stride {INFER_STRIDE} px",
        f"- {epochs} epochs per fold, Adam, weighted cross-entropy plus Dice\n",
        "## Per-class results (mean over folds)",
        "| Class | IoU | Dice | Precision | Recall |",
        "|---|---|---|---|---|",
    ]
    for index in range(N_CLASSES):
        cells = []
        for name in ("IoU", "Dice", "precision", "recall"):
            mean, std = _mean_std(per_fold, name, index)
            cells.append(f"{mean:.3f} ± {std:.3f}")
        lines.append(f"| {CLASSES[index]} | " + " | ".join(cells) + " |")
    lines += [
        f"\n- **mIoU**: {mean_iou:.3f} ± {std_iou:.3f}",
        f"- **Pixel accuracy**: {mean_accuracy:.3f} ± {std_accuracy:.3f}",
        "\n## Bundle quantification",
        "Reported for comparison with the instance route. Pixel IoU is not the "
        "criterion here; counting and area are.\n",
        f"- Object F1: {summary['object_F1']:.3f}",
        f"- Count error: {100 * summary['count_error']:.1f} %",
        f"- Area error after calibration: {100 * summary['area_error_calibrated']:.1f} %",
    ]
    (DIR_EVAL_SEMANTIC / "semantic_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _save_confusion_figure(normalised):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figure, axes = plt.subplots(figsize=(5, 4))
    image = axes.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    axes.set_xticks(range(N_CLASSES))
    axes.set_yticks(range(N_CLASSES))
    axes.set_xticklabels([CLASSES[c] for c in range(N_CLASSES)])
    axes.set_yticklabels([CLASSES[c] for c in range(N_CLASSES)])
    axes.set_xlabel("Predicted")
    axes.set_ylabel("Annotated")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            axes.text(
                j, i, f"{normalised[i, j]:.2f}", ha="center", va="center",
                color="white" if normalised[i, j] > 0.5 else "black",
            )
    figure.colorbar(image)
    figure.tight_layout()
    figure.savefig(DIR_EVAL_SEMANTIC / "confusion_matrix.png", dpi=150)
    plt.close(figure)
