"""Dataset assembly: source conversion, label maps, the manifest, and sampling.

The pipeline never learns from the source PNGs directly. This module turns them
into a single derived dataset under ``seg_work``:

* one three-class label map per annotated section, at the working scale;
* one rescaled leaflet image per section, used as model input;
* ``manifest.csv``, the authoritative list of usable sections.

It also owns the cross-validation split and the patch sampler, so that every
consumer of the dataset sees exactly the same partition.
"""
from __future__ import annotations

import csv
import glob
import os
import random
import re
import shutil
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import imaging
from .config import (
    BUNDLE_CENTRED_FRACTION,
    BUNDLE_CLASS,
    DIR_BUNDLE,
    DIR_DIC,
    DIR_EXAMPLES,
    DIR_INPUTS,
    DIR_LEAFLET,
    DIR_MASKS,
    DIR_OUT,
    EXCLUDED_KEYS,
    MANIFEST,
    PATCH,
    PATCHES_PER_IMAGE,
    SEED,
    WORKDIR,
    WORKING_SCALE,
)

MANIFEST_FIELDS = ["key", "family", "input", "mask"]
_FAMILIES = ("GALB", "MC87", "NOUR")


# ---------------------------------------------------------------------------
# Manifest access
# ---------------------------------------------------------------------------
def family_of(key: str) -> str:
    """Site a section was collected at, read from its key prefix."""
    for family in _FAMILIES:
        if key.startswith(family):
            return family
    return "OTHER"


def palm_of(key: str) -> str:
    """Individual a section was cut from: the site and the palm number.

    A key reads SITE_PALM_replicate, so dropping the replicate leaves the
    individual. Seven keys carry an extra segment, MC87_0259_1_1 or
    MC87_1178_2_bis, which the palm number survives unchanged. Falls back to the
    whole key when the pattern does not match, so an unparsable key becomes its
    own group rather than silently joining another palm's.
    """
    match = re.match(r"^([A-Z0-9]+)_(\d{4})", key)
    return f"{match.group(1)}-{match.group(2)}" if match else key


def read_manifest(include_excluded: bool = False) -> list[dict]:
    """Sections of the annotated set, minus those whose annotation is unusable.

    The exclusion used to be applied only when the manifest was rebuilt, so a
    key added to EXCLUDED_KEYS after the last build kept taking part in training
    and validation until someone remembered to run ``prepare`` again. Three keys
    were in exactly that state. Filtering on read makes the config the single
    authority, whatever the age of the manifest file.

    ``include_excluded`` returns everything, for the diagnostics that exist
    precisely to inspect the rejected sections.
    """
    with open(MANIFEST, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if include_excluded:
        return rows

    kept = [row for row in rows if row["key"] not in EXCLUDED_KEYS]
    dropped = len(rows) - len(kept)
    if dropped:
        print(f"[manifest] {len(kept)} section(s); {dropped} excluded by config "
              f"({', '.join(sorted(EXCLUDED_KEYS & {r['key'] for r in rows}))})")
    return kept


def resolve(row: dict, column: str):
    """Absolute path of a manifest entry; stored paths are root-relative."""
    return WORKDIR.parent / row[column]


def load_pair(row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rgb, instance map)`` for one manifest entry.

    Instances are the connected components of the bundle class. No separate
    instance annotation exists or is needed.
    """
    rgb = imaging.read_bgr_as_rgb(resolve(row, "input"))
    mask = imaging.read_grey(resolve(row, "mask"))
    from .config import min_bundle_area_px, WORKING_SCALE

    # Seuil physique : le meme nombre de pixels ne decrit pas le meme objet
    # selon la chaine d'acquisition (voir min_bundle_area_px dans config).
    instances = imaging.connected_instances(
        mask == BUNDLE_CLASS, min_bundle_area_px(row['key'], WORKING_SCALE))
    return rgb, instances


# ---------------------------------------------------------------------------
# Source conversion: the DIC set
# ---------------------------------------------------------------------------
_DIC_FAMILY = {"M": "MC87", "N": "NOUR", "G": "GALB"}
_DIC_WHITE = 230
_DIC_BLACK = 200


def _harmonised_key(source_key: str) -> str:
    """``1033_M259_1_1_7um_700_2000_400`` becomes ``MC87_0259_1_1``.

    The individual letter selects the family, the digits are padded to four, and
    parsing stops at the magnification block.
    """
    tokens = source_key.split("_")
    individual = tokens[1] if len(tokens) > 1 else "?"
    family = _DIC_FAMILY.get(individual[0], individual[0])
    digits = re.sub(r"\D", "", individual)
    name = f"{family}_{int(digits):04d}" if digits else f"{family}_0000"
    name += f"_{tokens[2]}" if len(tokens) > 2 else "_X"

    for token in tokens[3:]:
        lowered = token.lower()
        if re.match(r"^7[uµ]?m$", lowered) or re.match(r"^x?\d{3,4}", lowered) or "x" in lowered:
            break
        if token.isdigit() and len(token) <= 2:
            name += f"_{token}"
        elif "bis" in lowered:
            name += "_bis"
    return name


def _normalise_source_name(path: str) -> str:
    name = os.path.basename(path)
    name = re.sub(r"^Mask of Result of ", "", name)
    name = re.sub(r"\.png_\(RGB\)\.png$", ".png", name)
    name = re.sub(r"-1\.png$", ".png", name)
    return re.sub(r"\.png$", "", name)


def _index_dic_subdir(subdir: str) -> dict[str, str]:
    index: dict[str, str] = {}
    for path in glob.glob(os.path.join(str(DIR_DIC / subdir), "*.png")):
        key = _normalise_source_name(path)
        if key not in index:
            index[key] = path
        elif index[key].endswith("_(RGB).png") and not path.endswith("_(RGB).png"):
            index[key] = path   # prefer the binary mask over the RGB rendering
    return index


def _outside_of(white: np.ndarray) -> np.ndarray:
    """White area reachable from a corner, i.e. outside the section."""
    height, width = white.shape
    flood = white.astype(np.uint8).copy()
    scratch = np.zeros((height + 2, width + 2), np.uint8)
    for x, y in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        if flood[y, x] == 1:
            cv2.floodFill(flood, scratch, (x, y), 2)
    return flood == 2


def convert_dic() -> None:
    """Convert the DIC set into the leaflet/bundle convention used everywhere.

    DIC stores the leaflet on white and the bundles as black blobs on white.
    Converted sections keep the individual naming of the native set, because
    they come from the same MC87 and NOUR individuals.

    A converted section whose harmonised key already exists is *skipped*, not
    renamed. Earlier versions appended a ``_v2`` suffix, which silently created
    byte-identical twins of native sections; because folds are drawn by key, one
    twin could train while the other validated. That leak inflated every
    cross-validated figure produced before it was found.
    """
    native = {
        os.path.basename(path)[: -len("_LM.png")]
        for path in glob.glob(os.path.join(str(DIR_LEAFLET), "*_LM.png"))
    }

    leaflets = _index_dic_subdir("Leaflet mask")
    bundles = _index_dic_subdir("Bundles")
    keys = sorted(set(leaflets) & set(bundles))
    print(f"[dic] {len(keys)} source section(s) with both a leaflet and bundles.")

    written, skipped, mapping = 0, [], []
    for source in keys:
        key = _harmonised_key(source)
        if key in native:
            skipped.append((source, key))
            continue
        native.add(key)
        mapping.append((source, key))

        leaflet_rgb = imaging.read_rgb(leaflets[source])
        height, width = leaflet_rgb.shape[:2]
        white = (
            (leaflet_rgb[:, :, 0] > _DIC_WHITE)
            & (leaflet_rgb[:, :, 1] > _DIC_WHITE)
            & (leaflet_rgb[:, :, 2] > _DIC_WHITE)
        )
        outside = _outside_of(white)

        leaflet = leaflet_rgb.copy()
        leaflet[outside] = 0
        imaging.write_rgb(DIR_LEAFLET / f"{key}_LM.png", leaflet)

        bundle_rgb = imaging.read_rgb(bundles[source])
        if bundle_rgb.shape[:2] != (height, width):
            bundle_rgb = cv2.resize(
                bundle_rgb, (width, height), interpolation=cv2.INTER_NEAREST
            )
        bundle = (bundle_rgb.astype(np.int32).sum(2) < _DIC_BLACK) & ~outside
        out = np.zeros((height, width), np.uint8)
        out[bundle] = 255
        cv2.imwrite(str(DIR_BUNDLE / f"{key}_BU.png"), out)
        written += 1

    if skipped:
        print(f"[dic] {len(skipped)} source section(s) skipped: key already native.")
        for source, key in skipped[:10]:
            print(f"       {source} -> {key}")

    DIR_OUT.mkdir(parents=True, exist_ok=True)
    with open(DIR_OUT / "dic_name_mapping.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dic_source", "harmonised_key"])
        writer.writerows(mapping)
    print(f"[dic] {written} section(s) converted.")


def integrate_examples() -> None:
    """Copy the fully annotated reference sections into the main directories."""
    if not DIR_EXAMPLES.exists():
        print(f"[examples] {DIR_EXAMPLES} not found, skipped.")
        return

    targets = {"LM": DIR_LEAFLET, "BU": DIR_BUNDLE}
    copied = 0
    for subdir in sorted(DIR_EXAMPLES.iterdir()):
        if not subdir.is_dir():
            continue
        key = subdir.name
        for suffix, destination in targets.items():
            # Both the regular and the malformed naming appear in this set.
            source = next(
                (
                    candidate
                    for candidate in (subdir / f"{key}_{suffix}.png", subdir / f"{key}{suffix}.png")
                    if candidate.exists()
                ),
                None,
            )
            if source is None:
                continue
            target = destination / f"{key}_{suffix}.png"
            if not target.exists():
                shutil.copy2(source, target)
                copied += 1
    print(f"[examples] {copied} file(s) copied.")


# ---------------------------------------------------------------------------
# Label maps and manifest
# ---------------------------------------------------------------------------
def build_label_map(key: str):
    """Build ``(input rgb, label map)`` at the working scale, or ``None``.

    Priority is background, then leaflet, then bundle. The leaflet footprint
    comes from the leaflet image itself, so a bundle can never be labelled
    outside the section.
    """
    leaflet_path = DIR_LEAFLET / f"{key}_LM.png"
    bundle_path = DIR_BUNDLE / f"{key}_BU.png"
    if not leaflet_path.exists() or not bundle_path.exists():
        return None

    leaflet = imaging.read_rgb(leaflet_path)
    height, width = leaflet.shape[:2]
    size = (int(round(width * WORKING_SCALE)), int(round(height * WORKING_SCALE)))

    scaled = cv2.resize(leaflet, size, interpolation=cv2.INTER_AREA)
    scaled = imaging.clean_white_background(scaled)

    label = np.zeros(scaled.shape[:2], dtype=np.uint8)
    label[imaging.nonblack(scaled)] = 1

    bundle = cv2.resize(imaging.read_rgb(bundle_path), size, interpolation=cv2.INTER_AREA)
    label[imaging.nonblack(bundle) & (label == 1)] = BUNDLE_CLASS
    return scaled, label


def _annotated_keys() -> list[str]:
    keys = (path.name[: -len("_BU.png")] for path in DIR_BUNDLE.glob("*_BU.png"))
    return sorted(key for key in keys if key not in EXCLUDED_KEYS)


def _duplicate_keys(rows: list[dict]) -> dict[str, str]:
    """Map each duplicate key to the key it duplicates, comparing file contents.

    Sections are grouped by the shape and checksum of their label map first, so
    only genuine candidates are compared pixel by pixel.
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        mask = imaging.read_grey(resolve(row, "mask"))
        if mask is None:
            continue
        buckets[(mask.shape, int(mask.sum()))].append(row)

    duplicates: dict[str, str] = {}
    for candidates in buckets.values():
        if len(candidates) < 2:
            continue
        for position, row in enumerate(candidates):
            if row["key"] in duplicates:
                continue
            first = imaging.read_grey(resolve(row, "mask"))
            for other in candidates[position + 1:]:
                if other["key"] in duplicates:
                    continue
                second = imaging.read_grey(resolve(other, "mask"))
                if first.shape == second.shape and np.array_equal(first, second):
                    duplicates[other["key"]] = row["key"]
    return duplicates


def build_dataset() -> None:
    """Write the derived dataset and the manifest, dropping duplicate sections."""
    DIR_INPUTS.mkdir(parents=True, exist_ok=True)
    DIR_MASKS.mkdir(parents=True, exist_ok=True)

    keys = _annotated_keys()
    print(f"[build] {len(keys)} annotated section(s).")

    rows = []
    for position, key in enumerate(keys, 1):
        built = build_label_map(key)
        if built is None:
            print(f"  ! {key}: leaflet or bundle image missing, skipped.")
            continue
        scaled, label = built
        input_path = DIR_INPUTS / f"{key}.png"
        mask_path = DIR_MASKS / f"{key}.png"
        imaging.write_rgb(input_path, scaled)
        cv2.imwrite(str(mask_path), label)
        rows.append(
            {
                "key": key,
                "family": family_of(key),
                "input": str(input_path.relative_to(WORKDIR.parent)),
                "mask": str(mask_path.relative_to(WORKDIR.parent)),
            }
        )
        if position % 20 == 0 or position == len(keys):
            print(f"  {position}/{len(keys)}")

    duplicates = _duplicate_keys(rows)
    if duplicates:
        print(f"\n[build] {len(duplicates)} duplicate section(s) removed:")
        for duplicate, original in sorted(duplicates.items()):
            print(f"  {duplicate}  ==  {original}")
        rows = [row for row in rows if row["key"] not in duplicates]

    with open(MANIFEST, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = defaultdict(int)
    for row in rows:
        counts[row["family"]] += 1
    print(f"\n[build] {len(rows)} unique section(s) -> {MANIFEST}")
    print(f"[build] per family: {dict(sorted(counts.items()))}")


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def stratified_kfold(rows: list[dict], n_folds: int, seed: int = SEED) -> list[list[dict]]:
    """Split by PALM, stratified by family, so folds stay comparable and clean.

    Sections of one individual differ far more from another individual's than
    from each other, so an unstratified split would make folds measure different
    things.

    The unit assigned to a fold is the palm, not the section. Splitting by
    section looks equivalent while almost every palm contributes exactly one, and
    is not: a palm with two sections then has one in training and the other in
    validation, and the model is scored on an individual it has already seen.
    That is what happened to NOUR-0508 under the earlier section-wise split, its
    two sections falling in folds 1 and 3.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[palm_of(row["key"])].append(row)

    by_family: dict[str, list[list[dict]]] = defaultdict(list)
    for members in grouped.values():
        head = members[0]
        family = head.get("family") or family_of(head["key"])
        by_family[family].append(sorted(members, key=lambda row: row["key"]))

    folds: list[list[dict]] = [[] for _ in range(n_folds)]
    rng = random.Random(seed)
    offset = 0
    for family in sorted(by_family):
        palms = sorted(by_family[family], key=lambda group: group[0]["key"])
        rng.shuffle(palms)
        for position, group in enumerate(palms):
            folds[(offset + position) % n_folds].extend(group)
        # Carry the position on to the next family, otherwise every family
        # restarts at fold 0 and the first folds end up systematically larger.
        offset = (offset + len(palms)) % n_folds
    return folds


def train_validation_split(rows, folds, index):
    """Return ``(training rows, validation rows)`` for one fold."""
    validation = folds[index]
    held_out = {row["key"] for row in validation}
    return [row for row in rows if row["key"] not in held_out], validation


# ---------------------------------------------------------------------------
# Patch sampling
# ---------------------------------------------------------------------------
class PatchDataset(Dataset):
    """Random patches drawn from sections held in memory.

    A fixed share of patches is centred on a bundle. Bundles occupy a few per
    cent of the pixels, so uniform sampling would spend almost every patch on
    background.
    """

    def __init__(self, rows, patches_per_image=PATCHES_PER_IMAGE, augment=True):
        self.patch = PATCH
        self.per_image = patches_per_image
        self.augment = augment
        self.samples = []
        for row in rows:
            image = imaging.read_bgr_as_rgb(resolve(row, "input"))
            mask = imaging.read_grey(resolve(row, "mask"))
            self.samples.append(
                {
                    "image": image,
                    "mask": mask,
                    "bundle": np.argwhere(mask == BUNDLE_CLASS),
                }
            )

    def __len__(self):
        return len(self.samples) * self.per_image

    def _crop(self, sample):
        image, mask, bundle = sample["image"], sample["mask"], sample["bundle"]
        patch = self.patch
        height, width = mask.shape
        if height < patch or width < patch:
            pad_y, pad_x = max(0, patch - height), max(0, patch - width)
            image = cv2.copyMakeBorder(image, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT)
            mask = cv2.copyMakeBorder(mask, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT)
            height, width = mask.shape

        if len(bundle) and random.random() < BUNDLE_CENTRED_FRACTION:
            cy, cx = bundle[random.randrange(len(bundle))]
            y = int(np.clip(cy - patch // 2, 0, height - patch))
            x = int(np.clip(cx - patch // 2, 0, width - patch))
        else:
            y = random.randint(0, height - patch)
            x = random.randint(0, width - patch)
        return image[y:y + patch, x:x + patch].copy(), mask[y:y + patch, x:x + patch].copy()

    def __getitem__(self, index):
        image, mask = self._crop(self.samples[index // self.per_image])
        if self.augment:
            # Flips and right-angle rotations only. Hue jitter is never applied:
            # in FASGA the colour carries the biological signal.
            if random.random() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                image, mask = image[::-1], mask[::-1]
            turns = random.randint(0, 3)
            if turns:
                image, mask = np.rot90(image, turns), np.rot90(mask, turns)

        image = torch.from_numpy(np.ascontiguousarray(image).transpose(2, 0, 1)).float() / 255.0
        return image, torch.from_numpy(np.ascontiguousarray(mask)).long()
