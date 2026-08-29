"""Central configuration: paths, class scheme, thresholds and hyper-parameters.

Every tunable value used anywhere in the pipeline is defined here. No other
module hard-codes a path or a threshold.

The dataset root is resolved from the ``FASGA_ROOT`` environment variable when
it is set, so the same checkout runs unchanged on a workstation and on a
training server.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# config.py sits at <repo>/src/faspy/config.py, so the dataset root defaults to
# the directory containing the repository, where IMG_LM and IMG_BU live.
_DEFAULT_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("FASGA_ROOT", _DEFAULT_ROOT))

DIR_LEAFLET = ROOT / "IMG_LM"    # FASGA-stained leaflet section on black; model input
DIR_BUNDLE = ROOT / "IMG_BU"     # vascular bundle annotation on black
DIR_EXAMPLES = ROOT / "LU+VA+BULU"   # fully annotated reference sections
DIR_DIC = ROOT / "DIC"           # second histological set, ImageJ export conventions

WORKDIR = ROOT / "seg_work"      # derived data at the working scale
DIR_INPUTS = WORKDIR / "inputs"
DIR_MASKS = WORKDIR / "masks"
MANIFEST = WORKDIR / "manifest.csv"

DIR_UNET_MODELS = ROOT / "seg_models"
DIR_CELLPOSE_MODELS = ROOT / "cellpose_models"
UNET_CHECKPOINT = DIR_UNET_MODELS / "unet_fasga.pt"

DIR_OUT = ROOT / "Out"
DIR_EVAL_SEMANTIC = DIR_OUT / "eval"
DIR_EVAL_INSTANCE = DIR_OUT / "eval_cellpose"

# ---------------------------------------------------------------------------
# Class scheme
# ---------------------------------------------------------------------------
# Three mutually exclusive classes, applied in increasing order of priority.
# Vascular tissue and lumen were dropped as learnt classes; see README, section
# "Design decisions". Lumen is now measured photometrically at quantification
# time rather than segmented.
CLASSES = {0: "background", 1: "leaflet", 2: "bundle"}
N_CLASSES = len(CLASSES)
BUNDLE_CLASS = 2

# Overlay colours, BGR for OpenCV.
CLASS_COLOURS = {
    0: (0, 0, 0),
    1: (0, 200, 0),
    2: (0, 0, 255),
}

# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------
# Source sections are roughly 8000 x 12000 px. All learning and evaluation runs
# at WORKING_SCALE; the instance route applies a second reduction on top of it
# (CELLPOSE_RESCALE), so the effective scale there is the product of the two.
WORKING_SCALE = 0.5

NONBLACK_THRESHOLD = 15    # summed RGB above which a pixel belongs to the section
WHITE_BG_THRESHOLD = 230   # R, G and B all above this: stray white background

# Keys excluded from training because their annotation is unusable. The leaflet
# image itself stays a valid inference target.
# Excluded from training and evaluation because their bundle annotation is
# unusable. The leaflet image of each remains a valid inference target, so these
# sections are still quantified in production; they simply no longer teach the
# model, or count against it.
# Sections dropped from the PRODUCTION quantification (the 434-section set).
# Distinct from EXCLUDED_KEYS, which only concerns the annotated set.
#
# NOUR_0261_1: the PNG was cut off mid-write. It holds 314 valid IDAT chunks, a
# garbled fragment, and no end marker. Tissue goes from 2,106 pixels on row 2454
# to zero on row 2455 -- a clean cut through the leaflet, where a healthy image
# tapers off. Because readers tolerate truncated PNGs and pad the remainder with
# black, and the pipeline reads black as background, the section would be
# measured truncated without raising anything. Drop it until the source mosaic
# is re-exported.
EXCLUDED_SECTIONS = {"NOUR_0261_1"}

EXCLUDED_KEYS = {
    "NOUR_2014_1",   # bundle mask covers the whole leaflet
    "MC87_0177_2",   # one bundle drawn out of many; 2.08 % bundle share, the
                     # lowest of the set, against a site median of 13.7 %
    "NOUR_2000_2",   # bundles left undrawn; area 56 % below the manual measurement
    "MC87_0383_4",   # bundles left undrawn; lumen IoU 0.18 against manual masks
}

# Smallest object accepted as a bundle, defined at full resolution so that it
# stays physically meaningful when WORKING_SCALE changes.
MIN_BUNDLE_AREA_FULLRES = 6000
MIN_BUNDLE_AREA = max(1, int(MIN_BUNDLE_AREA_FULLRES * WORKING_SCALE ** 2))

# ---------------------------------------------------------------------------
# Semantic route (U-Net)
# ---------------------------------------------------------------------------
PATCH = 256
INFER_STRIDE = 192
BASE_CHANNELS = 32
IGNORE_INDEX = 255

CLASS_WEIGHTS = [0.5, 1.0, 2.5]   # background, leaflet, bundle
BATCH_SIZE = 8
EPOCHS = 35            # the cross-validation setting that produced the published figures
LEARNING_RATE = 1e-3
PATCHES_PER_IMAGE = 40
BUNDLE_CENTRED_FRACTION = 0.6     # share of patches centred on a bundle
POSTPROCESS_BUNDLE = True         # morphological closing plus area filter

# ---------------------------------------------------------------------------
# Instance route (Cellpose-SAM)
# ---------------------------------------------------------------------------
# Cellpose-SAM trains on fixed 256 px crops. At WORKING_SCALE alone the median
# bundle diameter is 168 px, so the largest bundles do not fit and are never
# learnt. Shrinking the data to 0.35 brings the median to 56 px, inside the
# window the network was designed for. Training and inference must always use
# the same value.
CELLPOSE_RESCALE = 0.35
CELLPROB_THRESHOLD = 0.0
FLOW_THRESHOLD = 0.4
CELLPOSE_EPOCHS = 100
CELLPOSE_LEARNING_RATE = 1e-5
CELLPOSE_WEIGHT_DECAY = 0.1

# ---------------------------------------------------------------------------
# Quantification
# ---------------------------------------------------------------------------
# Lumen is measured as bright pixels inside a bundle, at its own scale. The
# threshold is expressed as a fraction of each image's own dynamic range rather
# than as an absolute level, because an absolute level does not transfer between
# acquisition pipelines: converted sections here are brighter than natively
# annotated ones (median darkest channel 172 against 156 inside bundles), and a
# level calibrated on one under-detects the other by a factor of five.
#
# Calibrated against manual ImageJ measurements of lumen area on 17 sections:
#   absolute 180    bias +51.0 %  error 51.1 %  IoU 0.645  | ratio between
#   absolute 205    bias  -1.1 %  error 13.6 %  IoU 0.794  | pipelines 4.93
#   normalised 0.82 bias  -2.2 %  error 12.3 %  IoU 0.811  | ratio 0.98
LUMEN_NORM_THRESHOLD = 0.82
LUMEN_NORM_PERCENTILES = (1, 99)
LUMEN_SCALE = 0.5

# ---------------------------------------------------------------------------
# Spatial calibration
# ---------------------------------------------------------------------------
# Measured from a stage graticule following the laboratory's ImageJ protocol:
# 674.4149 px/mm at a 2.5x objective, scaling linearly with magnification.
# Sections were imaged with a 20x objective, hence 5395.3192 px/mm.
#
# Independent check: the camera spans 22.3 mm across 6000 photosites, so a
# photosite is 3.7167 um, which at 20x projects to 0.18584 um at the specimen,
# within 0.26 % of the calibrated value.
#
# The visual magnification of the microscope is not the magnification at the
# sensor: the eyepieces are not in the camera path, so only the objective and
# the camera adapter enter.
PIXELS_PER_MM = 5395.3192
PIXEL_SIZE_UM = 1000.0 / PIXELS_PER_MM        # at acquisition resolution
PIXEL_AREA_UM2 = PIXEL_SIZE_UM ** 2

# ---------------------------------------------------------------------------
# Anatomical traits
# ---------------------------------------------------------------------------
# A lumen wider than this equivalent diameter is counted as a vessel. Metaxylem
# vessels typically span 20-100 um while fibre lumina stay well below; the
# vessel count proved stable anywhere between 15 and 25 um on these sections.
VESSEL_MIN_DIAMETER_UM = 15.0

# Smallest object counted as a lumen at all. Without a floor the lumen map
# fragments into thousands of isolated pixels and the median diameter falls to
# 1.4 um, which is no longer a cell cavity. At 2 um the median returns to 2.9 um,
# the order of a fibre lumen.
LUMEN_MIN_DIAMETER_UM = 2.0

# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------
N_FOLDS = 5
SEED = 42
MATCH_IOU_THRESHOLD = 0.5   # object counts as detected above this IoU


# ---------------------------------------------------------------------------
# Second acquisition system: the AMAP sections
# ---------------------------------------------------------------------------
# The 24 sections prepared at AMAP were imaged on a Keyence VHX-7000 at a single
# magnification, not on the Canon/Olympus setup the calibration above describes.
# Their scale bar, burnt into the source images, reads 100 um for 696-697 px on
# every one of the 18 images where it could be measured, a spread of 0.15 pct.
# That is 0.14347 um per pixel, against 0.18535 for the native set.
#
# Applying the native value to them overstates lengths by 29.2 pct and areas by
# 66.9 pct. Nothing catches it downstream: the manual ImageJ reference is in
# PIXELS, so the area check that agrees to 0.0 pct tests the masking chain and
# never the conversion factor.
#
# The list is written out rather than derived from the DIC directory, so that
# the pipeline calibrates correctly on a machine that does not hold it. Measured
# values are archived in Out/eval_cellpose/calibration_keyence.csv.
PIXEL_SIZE_UM_KEYENCE = 0.14347

KEYENCE_SECTIONS = {
    "MC87_0115_3",
    "MC87_0117_3",
    "MC87_0177_2",
    "MC87_0259_1_1",
    "MC87_0282_1",
    "MC87_0301_3",
    "MC87_0312_1",
    "MC87_0333_3",
    "MC87_0364_4_3",
    "MC87_0383_4",
    "MC87_1019_2",
    "MC87_1040_9_3",
    "MC87_1044_2",
    "MC87_1089_2",
    "MC87_1178_2_bis",
    "NOUR_0012_1_2",
    "NOUR_0019_2",
    "NOUR_0084_3",
    "NOUR_0434_5",
    "NOUR_0500_1_3",
    "NOUR_0508_3",
    "NOUR_2000_2",
    "NOUR_2003_6_1",
    "NOUR_2009_4",
}


def pixel_size_for(key):
    """Pixel size in um at acquisition resolution, for one section key."""
    return PIXEL_SIZE_UM_KEYENCE if key in KEYENCE_SECTIONS else PIXEL_SIZE_UM


def pixel_area_for(key):
    """Pixel area in um2 at acquisition resolution, for one section key."""
    return pixel_size_for(key) ** 2


# Seuil de taille minimale d'un faisceau, exprime PHYSIQUEMENT.
#
# Il etait fixe a 6 000 px a pleine resolution, ce qui vaut 206 um2 sur la
# chaine native mais seulement 124 um2 sur le Keyence, dont le pixel est plus
# petit. Le meme nombre de pixels ne decrit donc pas le meme objet selon la
# chaine, et le seuil ne "preserve son sens physique" que s'il est defini en
# um2 puis converti coupe par coupe. L'effet mesure sur les annotations est
# faible -- un objet sur 1 126 -- mais il porte sur le comptage lui-meme.
MIN_BUNDLE_AREA_UM2 = MIN_BUNDLE_AREA_FULLRES * PIXEL_SIZE_UM ** 2


# Le jeu de donnees distribue a ete produit avec un seuil en PIXELS identique
# pour les deux chaines : 6 000 px a pleine resolution, soit 206 um2 en natif
# mais 123.5 um2 sur le Keyence. L'ecart a ete mesure et il est negligeable --
# UN objet sur 1 126 dans les annotations -- mais il est reel.
#
# Passer ce drapeau a True active le seuil physique, correct, au prix d'une
# divergence avec les mesures publiees. Il reste a False pour que le code
# reproduise exactement le fichier distribue ; le manuscrit decrit le seuil tel
# qu'il a ete applique, et non tel qu'il aurait du l'etre.
MIN_BUNDLE_AREA_PHYSICAL = False


def min_bundle_area_px(key=None, scale=1.0):
    """Seuil en pixels pour une coupe donnee, a l'echelle demandee.

    ``scale`` vaut 1.0 a la resolution d'acquisition et WORKING_SCALE a
    l'echelle de travail. ``key`` a None retombe sur la calibration native.

    Avec MIN_BUNDLE_AREA_PHYSICAL a False, renvoie le seuil uniforme qui a
    produit les mesures publiees, quelle que soit la coupe.
    """
    if not MIN_BUNDLE_AREA_PHYSICAL:
        return max(1, int(round(MIN_BUNDLE_AREA_FULLRES * scale ** 2)))
    pixel = PIXEL_SIZE_UM if key is None else pixel_size_for(key)
    return max(1, int(round(MIN_BUNDLE_AREA_UM2 / pixel ** 2 * scale ** 2)))


# Faut-il remplacer par NaN les traits derives du plan median quand le controle
# echoue ? NON par defaut, et c'est deliberé : le critere n'est pas arrete (le
# rapport d'aspect rejette 34 % des coupes, ce qui est trop), et une passe de
# production coute une heure de serveur. On ECRIT donc les mesures brutes --
# midplane_span_px, midplane_dt2_px, midplane_ratio, midplane_aspect -- et le
# seuil s'applique ensuite sur le CSV, autant de fois qu'on veut, sans relancer
# quoi que ce soit. Passer a True une fois le critere valide par lecture.
MIDPLANE_INVALIDATE = False
