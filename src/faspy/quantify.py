"""Production quantification of every leaflet section.

Self-contained: the leaflet footprint comes from the leaflet image itself, the
bundles from the final Cellpose model, and the lumen from a photometric rule
applied inside the detected bundles.

Lumen is measured rather than segmented. A learnt lumen class only ever worked
on the DIC subset, whose staining and exposure differ from the native set, so it
generalised to nothing.
"""
from __future__ import annotations

import csv
import math

import cv2
import numpy as np

from . import imaging, instances, traits
from .config import (
    CELLPOSE_RESCALE,
    CELLPROB_THRESHOLD,
    DIR_LEAFLET,
    DIR_OUT,
    EXCLUDED_SECTIONS,
    FLOW_THRESHOLD,
    PIXEL_AREA_UM2,
    min_bundle_area_px,
    pixel_area_for,
    pixel_size_for,
    WORKING_SCALE,
)

RESULTS_CSV = DIR_OUT / "quantification.csv"
FIELDS = [
    "key",
    "n_bundles",
    "leaflet_area_px",
    "bundle_area_px",
    "lumen_area_px",
    "leaflet_area_um2",
    "bundle_area_um2",
    "lumen_area_um2",
    "bundle_over_leaflet",
    "lumen_over_leaflet",
] + traits.TRAIT_FIELDS


def lumen_at_full_resolution(rgb: np.ndarray, bundle: np.ndarray,
                             leaflet: np.ndarray,
                             pixel_um: float) -> dict:
    """Lumen measured on the original image rather than on the working scale.

    The rule itself is unchanged and stays photometric and relative: see
    :func:`faspy.imaging.lumen_mask`. What changes is where it is applied. This
    helper used to resize by ``LUMEN_SCALE / WORKING_SCALE``, a ratio that
    equals one under the present configuration — the docstring promised a finer
    scale and no rescaling took place. Cavities are small enough for that to
    matter: the median lumen spans 9 px at the working scale and 18 px here, so
    its area is sampled four times as finely and the smallest ones stop brushing
    the detection floor.

    ``rgb`` is the section at acquisition resolution; the masks arrive at the
    working scale and are lifted to meet it. Returns the full-resolution masks
    together with their pixel size, ready for :func:`faspy.traits.section_traits`.
    """
    height, width = rgb.shape[:2]

    def lift(mask):
        return cv2.resize(mask.astype(np.uint8), (width, height),
                          interpolation=cv2.INTER_NEAREST) > 0

    fine_bundle, fine_leaflet = lift(bundle), lift(leaflet)
    if not fine_bundle.any() or not fine_leaflet.any():
        empty = np.zeros((height, width), dtype=bool)
        return {"lumen": empty, "bundle": fine_bundle, "leaflet": fine_leaflet,
                "pixel_um": pixel_um}

    return {
        "lumen": imaging.lumen_mask(rgb, fine_bundle, fine_leaflet),
        "bundle": fine_bundle,
        "leaflet": fine_leaflet,
        "pixel_um": pixel_um,
    }


def to_working_scale(mask: np.ndarray, shape) -> np.ndarray:
    """Bring a full-resolution mask back for the geometric traits."""
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]),
                      interpolation=cv2.INTER_NEAREST) > 0


def _warn_if_stale(checkpoint) -> None:
    """Refuse to run silently against a model older than the fold checkpoints.

    The final model is written only when cross-validation finishes, so between
    two campaigns the file on disk belongs to the previous one.
    """
    folds = sorted(checkpoint.parent.glob("cpsam_fold*"))
    if not folds or not checkpoint.exists():
        return
    newest = max(path.stat().st_mtime for path in folds)
    if checkpoint.stat().st_mtime < newest:
        print(
            "  Warning: the final model predates the fold checkpoints in the same "
            "directory. It is almost certainly from an earlier campaign. Re-run "
            "the cross-validation, or pass an explicit --model."
        )


def run(model_name="cpsam_final", rescale=CELLPOSE_RESCALE, limit=0) -> None:
    checkpoint = instances.checkpoint_path(model_name)
    _warn_if_stale(checkpoint)

    gpu = instances.gpu_available()
    model = instances.load_model(checkpoint, gpu)
    print(f"Cellpose model loaded (GPU={gpu}): {checkpoint}")

    paths = sorted(DIR_LEAFLET.glob("*_LM.png"))

    # Sections whose key ends in the duplicate suffix and whose twin is also
    # present are byte-identical copies left on disk by an earlier converter.
    # They were removed from the manifest but not from the image directory, and
    # a duplicated row would count the same palm twice in any later analysis.
    keys = {path.name[: -len("_LM.png")] for path in paths}
    duplicates = {k for k in keys if k.endswith("_v2") and k[:-3] in keys}
    if duplicates:
        paths = [p for p in paths if p.name[: -len("_LM.png")] not in duplicates]
        print(f"{len(duplicates)} duplicate section(s) skipped.")

    # Unusable images are dropped here rather than left to produce a row that
    # looks ordinary and that nothing downstream can tell apart.
    dropped = sorted(k for k in keys if k in EXCLUDED_SECTIONS)
    if dropped:
        paths = [p for p in paths if p.name[: -len("_LM.png")] not in EXCLUDED_SECTIONS]
        print(f"{len(dropped)} section(s) excluded from production: "
              f"{', '.join(dropped)} (see EXCLUDED_SECTIONS in config.py).")

    # Une passe limitee est un essai, pas un resultat : elle ecrit ailleurs.
    # Sans cela, `quantify --limit 1` remplace le CSV de production par une
    # seule ligne, et rien ne le signale.
    destination = RESULTS_CSV
    if limit:
        paths = paths[:limit]
        destination = RESULTS_CSV.with_name(f"{RESULTS_CSV.stem}_limit{limit}.csv")
        print(f"Trial run: {limit} section(s), written to {destination.name} so that "
              f"{RESULTS_CSV.name} is left intact.")
    print(f"{len(paths)} section(s) to quantify.")

    to_full_resolution = 1.0 / (WORKING_SCALE ** 2)
    destination.parent.mkdir(parents=True, exist_ok=True)

    failures = []
    with open(destination, "w", newline="", encoding="utf-8") as handle:
        # Une cellule vide, pas la chaine "nan" : les consommateurs en aval
        # comptent "nan" comme une valeur renseignee et l'integrent aux medianes
        # et aux variances. Le vide est le seul marqueur qu'ils traitent tous.
        writer = csv.DictWriter(handle, fieldnames=FIELDS, restval="")
        writer.writeheader()

        for position, path in enumerate(paths, 1):
            key = path.name[: -len("_LM.png")]
            try:
                rgb = imaging.read_rgb(path)
                scaled = imaging.clean_white_background(
                    imaging.resize_rgb(rgb, WORKING_SCALE)
                )
                leaflet = imaging.nonblack(scaled)

                predicted = instances.predict(
                    model, scaled, rescale, CELLPROB_THRESHOLD, FLOW_THRESHOLD,
                    min_size=min_bundle_area_px(key, WORKING_SCALE),
                )
                predicted[~leaflet] = 0        # bundles cannot lie outside the section
                bundle = predicted > 0

                # Lumen on the original image; geometry stays at the working
                # scale, where the objects are a hundred times larger.
                # Deux systemes d'acquisition, deux tailles de pixel.
                pixel_um = pixel_size_for(key)
                pixel_area = pixel_area_for(key)
                fine = lumen_at_full_resolution(rgb, bundle, leaflet, pixel_um)
                lumen = to_working_scale(fine["lumen"], bundle.shape)

                leaflet_px = int(leaflet.sum())
                bundle_px = int(bundle.sum())

                full = [leaflet_px * to_full_resolution,
                        bundle_px * to_full_resolution,
                        int(fine["lumen"].sum())]     # already full-resolution
                row = {
                        "key": key,
                        "n_bundles": len([i for i in np.unique(predicted) if i > 0]),
                        "leaflet_area_px": int(full[0]),
                        "bundle_area_px": int(full[1]),
                        "lumen_area_px": int(full[2]),
                        "leaflet_area_um2": round(full[0] * pixel_area, 1),
                        "bundle_area_um2": round(full[1] * pixel_area, 1),
                        "lumen_area_um2": round(full[2] * pixel_area, 1),
                        "bundle_over_leaflet": (
                            round(bundle_px / leaflet_px, 5) if leaflet_px else 0.0
                        ),
                        "lumen_over_leaflet": (
                            round(full[2] / full[0], 5) if full[0] else 0.0
                        ),
                        **traits.section_traits(leaflet, predicted, lumen, fine=fine,
                                               pixel_um=pixel_um / WORKING_SCALE),
                }
                writer.writerow({
                    k: ("" if isinstance(v, float) and not math.isfinite(v) else v)
                    for k, v in row.items()
                })
            except Exception as error:                      # noqa: BLE001
                failures.append((key, f"{type(error).__name__}: {error}"))
                print(f"  ! {key} skipped ({type(error).__name__}: {error})")

            if position % 20 == 0 or position == len(paths):
                print(f"  {position}/{len(paths)}")

    written = len(paths) - len(failures)
    print(f"{written} of {len(paths)} section(s) written -> {destination}")

    # Une quantification partielle qui annonce sa reussite est le pire des cas :
    # le CSV existe, il a l'air normal, et il manque des lignes. On le dit, et on
    # sort en erreur pour qu'un enchainement de scripts s'arrete ici.
    if failures:
        print()
        print(f"  INCOMPLETE: {len(failures)} section(s) failed and are absent "
              f"from the file. Do not use it as a production table.")
        for key, reason in failures:
            print(f"    {key}: {reason}")
        raise SystemExit(1)
