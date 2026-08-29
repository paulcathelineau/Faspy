"""Diagnostics: data quality checks and the audits that guide the next decision.

These commands do not train anything. Each answers one question that changes
what to do next, and each writes a file that can be inspected by eye.
"""
from __future__ import annotations

import csv
import re

import numpy as np
from PIL import Image

from . import imaging, instances, metrics
from .config import (
    CELLPOSE_RESCALE,
    CELLPROB_THRESHOLD,
    DIR_BUNDLE,
    DIR_EVAL_INSTANCE,
    DIR_LEAFLET,
    FLOW_THRESHOLD,
    MIN_BUNDLE_AREA,
    N_FOLDS,
    SEED,
)
from .datasets import load_pair, read_manifest, resolve, stratified_kfold


# ---------------------------------------------------------------------------
# File integrity
# ---------------------------------------------------------------------------
def check_images() -> int:
    """Report source PNGs that cannot be fully decoded.

    Truncated images are tolerated everywhere else in the pipeline, which means
    they fail silently. This is the one place that must not tolerate them, so
    the truncation guard is deliberately left at its default here.
    """
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = False
    Image.MAX_IMAGE_PIXELS = None

    total_bad = 0
    for directory in (DIR_LEAFLET, DIR_BUNDLE):
        if not directory.exists():
            continue
        paths = sorted(directory.glob("*.png"))
        bad = []
        for path in paths:
            try:
                with Image.open(path) as image:
                    image.load()
            except Exception as error:                      # noqa: BLE001
                bad.append((path.name, f"{type(error).__name__}: {error}"))
        print(f"{directory.name:10}: {len(paths)} file(s), {len(bad)} unreadable")
        for name, error in bad:
            print(f"    ! {name} -> {error}")
        total_bad += len(bad)

    print("-" * 50)
    print(f"{total_bad} file(s) to re-export." if total_bad else "Every file decodes cleanly.")
    return total_bad


def check_annotations(low_percentile=5.0) -> None:
    """Flag annotations that are implausible against the rest of the set.

    The decisive signal is the share of the leaflet occupied by bundles. Across
    the set that share is tightly distributed, so a section far below the bulk is
    usually one where bundles were left undrawn rather than one where they are
    genuinely scarce. It is a flag, not a filter: bundle density does vary between
    species, so anything reported here must be looked at before being excluded.

    A section with undrawn bundles is doubly harmful. It teaches the model that
    bundles are background, and it inflates the count of detected objects that no
    annotation covers, which is the very statistic used to judge the reference.
    """
    rows = read_manifest(include_excluded=True)
    print(f"{len(rows)} annotated section(s).")

    measured, suspicious = [], []
    for position, row in enumerate(rows, 1):
        mask = imaging.read_grey(resolve(row, "mask"))
        if mask is None:
            continue
        leaflet = int((mask >= 1).sum())
        bundle = int((mask == 2).sum())
        if leaflet == 0:
            suspicious.append((row["key"], "empty leaflet mask"))
            continue

        share = 100 * bundle / leaflet
        measured.append((row["key"], share))
        if share > 50:
            suspicious.append((row["key"], f"bundles cover {share:.0f} % of the section"))
        elif bundle == 0:
            suspicious.append((row["key"], "no bundle annotated"))
        if position % 50 == 0 or position == len(rows):
            print(f"  {position}/{len(rows)}")

    if measured:
        shares = np.array([value for _, value in measured])
        cut = float(np.percentile(shares, low_percentile))
        print(
            f"\nBundle share of the leaflet: median {np.median(shares):.2f} %, "
            f"{low_percentile:g}th percentile {cut:.2f} %"
        )
        print(f"\nLowest sections:")
        for key, share in sorted(measured, key=lambda item: item[1])[:10]:
            marker = "  <-- below the percentile" if share <= cut else ""
            print(f"  {key:22} {share:5.2f} %{marker}")
        print(
            "\nA low share can be biological, so inspect these before excluding any "
            "of them."
        )

    if suspicious:
        print(f"\n{len(suspicious)} implausible annotation(s):")
        for key, reason in suspicious:
            print(f"  {key:22} {reason}")
        print("\nAdd confirmed cases to EXCLUDED_KEYS in config.py.")


# ---------------------------------------------------------------------------
# Ground-truth completeness
# ---------------------------------------------------------------------------
def orphan_census(
    model_name=None,
    folds=N_FOLDS,
    rescale=CELLPOSE_RESCALE,
    cellprob=CELLPROB_THRESHOLD,
    flow=FLOW_THRESHOLD,
    overlap_fraction=0.25,
    limit=0,
    crops=True,
):
    """Find predicted bundles that overlap no annotated object, and show them.

    This decides the next move. If the omissions are concentrated on a handful
    of sections, dropping those sections is viable; if they are spread across
    the set, the labels have to be completed instead.

    By default each section is predicted by the fold model that never saw it.
    Passing ``model_name`` uses one model for everything, which is simpler but
    optimistic, since most sections were in its training set: usable to *find*
    omissions, not to measure how many there are.

    ``limit`` truncates the manifest in order, and the manifest is sorted by key,
    so a small limit samples one individual only. Use it for a smoke test, never
    for a per-family conclusion.
    """
    rows = read_manifest()
    if limit:
        rows = rows[:limit]
        print("  Note: --limit takes the first sections in key order, so the "
              "sample is not representative across individuals.\n")

    if model_name:
        assignment = {row["key"]: instances.checkpoint_path(model_name) for row in rows}
        mode = f"single model ({model_name})"
    else:
        assignment = {}
        for index, fold in enumerate(stratified_kfold(read_manifest(), folds, SEED)):
            for row in fold:
                assignment[row["key"]] = instances.checkpoint_path(f"cpsam_fold{index}")
        mode = f"out-of-fold ({folds} models)"

    gpu = instances.gpu_available()
    print(f"Orphan census | {len(rows)} sections | rescale {rescale} | GPU={gpu}")
    print(f"Mode: {mode}\n")

    output_dir = DIR_EVAL_INSTANCE / "orphans"
    output_dir.mkdir(parents=True, exist_ok=True)

    cache, records, n_predictions = {}, [], 0
    for position, row in enumerate(rows, 1):
        checkpoint = assignment.get(row["key"])
        if checkpoint is None:
            continue
        if checkpoint not in cache:
            cache[checkpoint] = instances.load_model(checkpoint, gpu)

        rgb, truth = load_pair(row)
        predicted = instances.predict(cache[checkpoint], rgb, rescale, cellprob, flow)
        n_predictions += len([i for i in np.unique(predicted) if i > 0])

        depth = imaging.depth_map(rgb)
        annotated = truth > 0
        for index in np.unique(predicted):
            if index == 0:
                continue
            selected = predicted == index
            area = int(selected.sum())
            if area < MIN_BUNDLE_AREA:
                continue
            if int((selected & annotated).sum()) >= overlap_fraction * area:
                continue

            ys, xs = np.where(selected)
            cy = int(np.clip(round(ys.mean()), 0, truth.shape[0] - 1))
            cx = int(np.clip(round(xs.mean()), 0, truth.shape[1] - 1))
            radius = float(np.sqrt(area / np.pi))
            name = f"{row['key']}_obj{int(index):03d}.png"

            if crops:
                _write_crop(output_dir / name, rgb, annotated, selected, ys, xs)

            records.append(
                {
                    "key": row["key"],
                    "family": row.get("family", "?"),
                    "object_id": int(index),
                    "area": area,
                    "depth": round(float(depth[cy, cx]), 1),
                    "depth_over_radius": round(float(depth[cy, cx]) / radius, 2) if radius else 0.0,
                    "crop": name,
                    "decision": "",
                }
            )

        if position % 10 == 0 or position == len(rows):
            print(f"  {position}/{len(rows)} sections | {len(records)} orphans")

    if not records:
        print("\nNo orphan found: the annotation looks complete.")
        return []

    csv_path = DIR_EVAL_INSTANCE / "orphan_census.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    _report_orphans(records, rows, n_predictions, csv_path, output_dir if crops else None)
    return records


def _write_crop(path, rgb, annotated, selected, ys, xs):
    margin = int(0.35 * max(np.ptp(ys), np.ptp(xs))) + 10
    y0, y1 = max(0, ys.min() - margin), min(rgb.shape[0], ys.max() + margin)
    x0, x1 = max(0, xs.min() - margin), min(rgb.shape[1], xs.max() + margin)

    import cv2

    crop = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR).copy()
    thickness = max(2, int(0.004 * max(crop.shape[:2])))
    imaging.draw_outline(crop, annotated[y0:y1, x0:x1], (255, 255, 255), thickness)
    imaging.draw_outline(crop, selected[y0:y1, x0:x1], (0, 0, 255), thickness + 1)
    cv2.imwrite(str(path), crop)


def _report_orphans(records, rows, n_predictions, csv_path, crop_dir):
    per_section: dict[str, int] = {}
    per_family: dict[str, int] = {}
    for record in records:
        per_section[record["key"]] = per_section.get(record["key"], 0) + 1
        per_family[record["family"]] = per_family.get(record["family"], 0) + 1

    ordered = sorted(per_section.items(), key=lambda item: -item[1])
    cumulative, top = 0, 0
    for _, count in ordered:
        cumulative += count
        top += 1
        if cumulative >= 0.8 * len(records):
            break

    print("\n" + "=" * 62)
    print(f"Orphans: {len(records)} of {n_predictions} predictions "
          f"({100 * len(records) / n_predictions:.1f} %)")
    print(f"Sections affected: {len(per_section)} / {len(rows)} "
          f"({100 * len(per_section) / len(rows):.0f} %)")
    print(f"80 % of orphans fall in {top} section(s) "
          f"({100 * top / len(rows):.0f} % of the set)")

    if top <= 0.15 * len(rows):
        print("\n>>> CONCENTRATED: excluding these sections from training is viable.")
    else:
        print("\n>>> DIFFUSE: exclusion would cost too many sections; complete the labels.")

    print("\nMost affected sections:")
    for key, count in ordered[:10]:
        print(f"  {key:24s} {count:3d}")
    print(f"\nBy individual: {dict(sorted(per_family.items()))}")
    print(f"\nCSV -> {csv_path}")
    if crop_dir:
        print(f"Crops -> {crop_dir}  (white = annotated, red = detected but not annotated)")
    print("\nFill in `decision`: yes = real bundle, no = false detection.")


# ---------------------------------------------------------------------------
# Threshold and depth calibration
# ---------------------------------------------------------------------------
def sweep_thresholds(
    fold=0,
    folds=N_FOLDS,
    model_name=None,
    cellprob_values=(0.0, -1.0, -2.0, -3.0),
    flow_values=(0.4,),
    rescale=CELLPOSE_RESCALE,
    limit=0,
):
    """Score decoding thresholds on one fold's validation sections.

    No retraining: a lower ``cellprob_threshold`` grows the masks, a higher
    ``flow_threshold`` tolerates more irregular outlines. Because the model is
    the one that held this fold out, the sections are genuinely unseen.
    """
    rows = read_manifest()
    validation = stratified_kfold(rows, folds, SEED)[fold]
    if limit:
        validation = validation[:limit]

    checkpoint = instances.checkpoint_path(model_name or f"cpsam_fold{fold}")
    model = instances.load_model(checkpoint, instances.gpu_available())
    total = len(cellprob_values) * len(flow_values)
    print(f"Threshold sweep | {checkpoint} | fold {fold}/{folds} | "
          f"{len(validation)} sections | {total} pass(es)\n")

    results = []
    for flow in flow_values:
        for cellprob in cellprob_values:
            accumulator = metrics.Quantification()
            for row in validation:
                rgb, truth = load_pair(row)
                predicted = instances.predict(model, rgb, rescale, cellprob, flow)
                accumulator.add(truth, predicted, row.get("family", "?"))
            summary = accumulator.summary()
            summary["cellprob_threshold"] = cellprob
            summary["flow_threshold"] = flow
            summary["rescale"] = rescale
            results.append(summary)
            print(f"cellprob={cellprob:+.1f} flow={flow:.2f} rescale={rescale:.2f}")
            print(metrics.format_summary(summary))
            print()

    DIR_EVAL_INSTANCE.mkdir(parents=True, exist_ok=True)
    path = DIR_EVAL_INSTANCE / "threshold_sweep.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()}
            )

    best = min(results, key=lambda result: abs(result["area_bias"]))
    print(f"Written -> {path}")
    print(f"Smallest area bias: cellprob={best['cellprob_threshold']:+.1f} "
          f"flow={best['flow_threshold']:.2f} -> {100 * best['area_bias']:+.1f} % "
          f"| F1 {best['object_F1']:.3f}")
    return results


def calibrate_depth():
    """Distribution of annotated bundle depth, used to set a trichome filter.

    A trichome is an epidermal outgrowth, so its centroid lies within about one
    radius of the cuticle; a vascular bundle is buried in the mesophyll. Any
    threshold must be read off the annotated objects, never chosen by hand: the
    first percentile keeps 99 % of real bundles by construction.
    """
    rows = read_manifest()
    depths, ratios, records = [], [], []

    for position, row in enumerate(rows, 1):
        rgb, truth = load_pair(row)
        depth = imaging.depth_map(rgb)
        for index, area, radius, value in imaging.instance_geometry(truth, depth):
            ratio = value / radius if radius else 0.0
            depths.append(value)
            ratios.append(ratio)
            records.append(
                {
                    "key": row["key"],
                    "object_id": index,
                    "area": area,
                    "depth": round(value, 1),
                    "depth_over_radius": round(ratio, 2),
                }
            )
        if position % 25 == 0 or position == len(rows):
            print(f"  {position}/{len(rows)} sections | {len(records)} bundles")

    DIR_EVAL_INSTANCE.mkdir(parents=True, exist_ok=True)
    path = DIR_EVAL_INSTANCE / "bundle_depth.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    percentiles = [0.5, 1, 5, 50]
    print(f"\n{len(records)} annotated bundles")
    print("  depth (px)      : " + "  ".join(
        f"p{p}={np.percentile(depths, p):.0f}" for p in percentiles))
    print("  depth / radius  : " + "  ".join(
        f"p{p}={np.percentile(ratios, p):.2f}" for p in percentiles))
    print(f"\nRecommended threshold (1st percentile): depth "
          f"{np.percentile(depths, 1):.0f} px, ratio {np.percentile(ratios, 1):.2f}")
    print(f"Written -> {path}")


# ---------------------------------------------------------------------------
# Validation against manual measurements
# ---------------------------------------------------------------------------
def _section_id(name):
    """Split a sample name into site, palm and section, ignoring any suffix."""
    match = re.match(r"([A-Z0-9]+)_(\d{4})_(\d+)", str(name))
    return (match.group(1), match.group(2), match.group(3)) if match else None


def validate_lumen(reference_csv, exclude=(), sweep=True):
    """Compare pipeline areas with manual measurements, and calibrate the lumen rule.

    The reference file must carry a sample name and the columns ``Leaflet area``,
    ``Bundle area`` and ``lumen area``, measured at full resolution. Sections are
    matched on site, palm and section number, so a sample measured on a different
    section of the same palm is not paired.

    Leaflet and bundle areas are computed from the annotation, which checks the
    area pipeline itself: any disagreement there is a mask or scaling problem,
    not a segmentation one. The lumen comparison is what matters, since the
    photometric rule has no other reference. ``sweep`` reports the brightness
    threshold that removes the bias, which is how the shipped value was chosen.
    """
    import numpy as np

    from .config import BUNDLE_CLASS, LUMEN_NORM_THRESHOLD
    from .datasets import resolve

    with open(reference_csv, encoding="utf-8-sig") as handle:
        sample = handle.readline()
        delimiter = ";" if sample.count(";") > sample.count(",") else ","
        handle.seek(0)
        rows = [row for row in csv.DictReader(handle, delimiter=delimiter)
                if (row.get("Leaflet area") or "").strip()]

    key_column = next(iter(rows[0]))
    manual = {_section_id(row[key_column]): row for row in rows if _section_id(row[key_column])}
    ours = {_section_id(row["key"]): row for row in read_manifest() if _section_id(row["key"])}
    matched = [k for k in sorted(set(manual) & set(ours)) if ours[k]["key"] not in set(exclude)]

    print(f"Reference: {len(manual)} measured sections | manifest: {len(ours)} | "
          f"matched on section: {len(matched)}")
    if not matched:
        return []

    records, brightness = [], []
    for key in matched:
        row = ours[key]
        # Meme resolution que les mesures ImageJ auxquelles on se compare, et que
        # la production depuis le passage a la pleine resolution.
        rgb = imaging.clean_white_background(
            imaging.read_rgb(DIR_LEAFLET / f"{row['key']}_LM.png"))
        leaflet = imaging.nonblack(rgb)
        bundle = imaging.nonblack(imaging.read_rgb(DIR_BUNDLE / f"{row['key']}_BU.png"))
        if bundle.shape != leaflet.shape:
            bundle = cv2.resize(bundle.astype(np.uint8),
                                (leaflet.shape[1], leaflet.shape[0]),
                                interpolation=cv2.INTER_NEAREST) > 0
        bundle &= leaflet
        mask = None
        darkest = rgb.min(axis=2)
        low, high = np.percentile(darkest[leaflet], (1, 99))
        normalised = (darkest.astype(np.float32) - low) / max(float(high - low), 1.0)
        lumen = imaging.lumen_mask(rgb, bundle, leaflet)
        records.append({
            "key": row["key"],
            "leaflet": float(leaflet.sum()),
            "bundle": float(bundle.sum()),
            "lumen": float(lumen.sum()),
            "leaflet_ref": float(manual[key]["Leaflet area"]),
            "bundle_ref": float(manual[key]["Bundle area"]),
            "lumen_ref": float(manual[key]["lumen area"]),
        })
        brightness.append((normalised[bundle], float(manual[key]["lumen area"])))

    print(f"\n{'section':20}{'leaflet':>12}{'bundle':>12}{'lumen':>12}   (relative to reference)")
    for record in records:
        print(f"{record['key']:20}"
              + "".join(f"{100 * (record[n] - record[n + '_ref']) / record[n + '_ref']:>11.1f}%"
                        for n in ("leaflet", "bundle", "lumen")))

    print()
    for name in ("leaflet", "bundle", "lumen"):
        ours_v = np.array([r[name] for r in records])
        ref_v = np.array([r[name + "_ref"] for r in records])
        relative = (ours_v - ref_v) / ref_v
        print(f"  {name:9}: median bias {100 * np.median(relative):+6.1f} % | "
              f"median absolute error {100 * np.median(np.abs(relative)):5.1f} % | "
              f"r = {np.corrcoef(ours_v, ref_v)[0, 1]:.3f}")

    if sweep:
        print(f"\nNormalised threshold sweep (current value {LUMEN_NORM_THRESHOLD}):")
        best = None
        for threshold in [t / 100 for t in range(70, 95, 2)]:
            relative = [(int((values > threshold).sum()) - reference) / reference
                        for values, reference in brightness]
            bias, error = 100 * np.median(relative), 100 * np.median(np.abs(relative))
            if best is None or error < best[1]:
                best = (threshold, error, bias)
            print(f"  {threshold:>5.2f}: bias {bias:+7.1f} %   absolute error {error:6.1f} %")
        print(f"  -> lowest error at {best[0]:.2f} (bias {best[2]:+.1f} %, error {best[1]:.1f} %)")

    path = DIR_EVAL_INSTANCE / "manual_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nWritten -> {path}")
    return records


def compare_lumen_masks(lumen_dir=None, dark_threshold=60, exclude=()):
    """Compare the photometric lumen rule with manually drawn lumen masks, pixel by pixel.

    Area agreement only shows that two methods produce the same total; this shows
    that they select the same pixels, which is a far stronger statement. The
    reference masks follow the ImageJ convention of dark objects on a white
    ground and are confined to the annotated bundles before comparison, since
    the photometric rule only ever applies inside a bundle.
    """
    import cv2
    import numpy as np

    from .config import BUNDLE_CLASS, DIR_DIC, LUMEN_NORM_THRESHOLD
    from .datasets import _harmonised_key, _normalise_source_name, resolve

    lumen_dir = lumen_dir or (DIR_DIC / "Lumen")
    if not lumen_dir.exists():
        print(f"No manual lumen masks at {lumen_dir}.")
        return []

    index = {}
    for path in sorted(lumen_dir.glob("*.png")):
        key = _harmonised_key(_normalise_source_name(str(path)))
        if key not in index or not path.name.endswith("_(RGB).png"):
            index[key] = path

    rows = {row["key"]: row for row in read_manifest()}
    matched = [k for k in sorted(set(index) & set(rows)) if k not in set(exclude)]
    print(f"Manual lumen masks: {len(index)} | matched to the manifest: {len(matched)}\n")
    if not matched:
        return []

    print(f"{'section':20}{'IoU':>8}{'precision':>11}{'recall':>9}")
    records = []
    for key in matched:
        row = rows[key]
        # A pleine resolution, du cote de la reference. La version precedente
        # ramenait le masque manuel a l'echelle de travail pour le comparer :
        # elle degradait la verite terrain afin de la confronter a une mesure
        # elle-meme degradee, ce qui flattait les deux.
        rgb = imaging.read_rgb(DIR_LEAFLET / f"{key}_LM.png")
        rgb = imaging.clean_white_background(rgb)
        leaflet = imaging.nonblack(rgb)
        bundle = imaging.nonblack(imaging.read_rgb(DIR_BUNDLE / f"{key}_BU.png"))
        if bundle.shape != leaflet.shape:
            bundle = cv2.resize(bundle.astype(np.uint8),
                                (leaflet.shape[1], leaflet.shape[0]),
                                interpolation=cv2.INTER_NEAREST) > 0
        bundle &= leaflet

        reference = imaging.read_rgb(index[key]).astype(np.int32).sum(2) < dark_threshold
        if reference.shape != leaflet.shape:
            reference = cv2.resize(reference.astype(np.uint8),
                                   (leaflet.shape[1], leaflet.shape[0]),
                                   interpolation=cv2.INTER_NEAREST) > 0
        reference &= bundle

        predicted = imaging.lumen_mask(rgb, bundle, leaflet)
        intersection = int((predicted & reference).sum())
        union = int((predicted | reference).sum())
        record = {
            "key": key,
            "IoU": intersection / union if union else 1.0,
            "precision": intersection / max(int(predicted.sum()), 1),
            "recall": intersection / max(int(reference.sum()), 1),
        }
        records.append(record)
        print(f"{key:20}{record['IoU']:>8.3f}{record['precision']:>11.3f}{record['recall']:>9.3f}")

    for name in ("IoU", "precision", "recall"):
        values = np.array([r[name] for r in records])
        print(f"\n  median {name}: {np.median(values):.3f}", end="")
    print(f"\n  at a normalised threshold of {LUMEN_NORM_THRESHOLD}")

    path = DIR_EVAL_INSTANCE / "lumen_mask_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nWritten -> {path}")
    return records
