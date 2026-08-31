"""Anatomical traits derived from the segmented bundles.

Counting bundles and summing their area answers *how much* conducting tissue a
leaflet carries. The biomechanical question also asks *where* that tissue sits,
which those quantities cannot express. This module adds three families of
descriptors, all derived from instances the model has already produced. No
additional model, no additional training.

``wall``          bundle area minus lumen area, and its shares.

``geometry``      the second moment of area about the leaflet's mid-plane.
                  Flexural stiffness does not follow the amount of material but
                  its position: a element's contribution grows with the SQUARE
                  of its distance from the neutral axis, so a peripheral bundle
                  counts for far more than a central one of equal area. This is
                  the trait that ties anatomy directly to mechanics.

``organisation``  depth beneath the epidermis, spacing between neighbours,
                  dispersion of sizes, and the lumen resolved into objects --
                  the large ones being metaxylem vessels, the small ones fibre
                  lumina.

Lengths come out in micrometres, areas in um^2 and second moments in um^4, via
the calibration held in :mod:`faspy.config`.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import (
    LUMEN_MIN_DIAMETER_UM,
    MIDPLANE_INVALIDATE,
    PIXEL_SIZE_UM,
    VESSEL_MIN_DIAMETER_UM,
    WORKING_SCALE,
)

#: Pixel size at the WORKING scale, where every mask in this module lives.
PIXEL_UM = PIXEL_SIZE_UM / WORKING_SCALE


def straighten_matrix(mask: np.ndarray):
    """Rotation laying the strip horizontal, and the resulting canvas size."""
    points = cv2.findNonZero(mask.astype(np.uint8))
    (cx, cy), (width, height), angle = cv2.minAreaRect(points)
    if width < height:
        angle += 90.0

    rows, columns = mask.shape
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    size = (int(rows * sin + columns * cos), int(rows * cos + columns * sin))
    matrix[0, 2] += size[0] / 2 - cx
    matrix[1, 2] += size[1] / 2 - cy
    return matrix, size


# Taille de l'ouverture morphologique appliquee AVANT toute geometrie, en
# micrometres pour rester independante de la chaine d'acquisition.
# Fixee par balayage contre une lecture visuelle a l'aveugle de 50 coupes.
# 3 um est la PLUS PETITE ouverture qui atteigne le plein effet : elle repare
# les 23 lignes jugees invalides sans en degrader aucune des 26 jugees valides,
# et 4, 6, 8 et 10 um ne font pas mieux tout en retirant davantage de tissu
# (1.19 % d'aire a 3 um contre 2.21 % a 10 um). Le temoin sans nettoyage n'en
# reparait que 6 sur 23.
GEOMETRY_OPEN_UM = 3.0


def geometry_mask(leaflet: np.ndarray, pixel_um: float | None = None) -> np.ndarray:
    """Masque de foliole nettoye, pour la GEOMETRIE seulement.

    Les mosaiques portent de fines lignes de panneautage qui ne sont pas noires :
    nonblack() les integre au masque, ou elles deplacent les bornes haute et
    basse de chaque colonne, donc leur milieu. Leur surface est negligeable --
    beaucoup de coupes touchees ont plus de 99 % de leur aire dans la composante
    principale, parfois une seule composante -- mais leur effet sur la ligne
    mediane ne l'est pas.

    Mesure sur les 50 coupes lues a l'aveugle : la part de la ligne mediane
    tombant hors du tissu passe d'une mediane de 0.250 a 0.005 sur les 24 coupes
    jugees invalides, sans degrader les 26 jugees valides (0 rejet a un seuil de
    0.10). Garder seulement la composante principale ne suffisait pas : le bruit
    est souvent connecte au tissu.

    Ce masque ne sert QU'A la geometrie. Les surfaces, fractions et comptages
    restent calcules sur le masque d'origine, sans quoi l'ouverture modifierait
    des mesures qu'elle n'a pas vocation a corriger.
    """
    # Mettre GEOMETRY_OPEN_UM a zero reproduit la geometrie HISTORIQUE, sans
    # filtre : c'est ainsi que l'ancien sections.csv reste reproductible.
    if GEOMETRY_OPEN_UM <= 0:
        return leaflet
    pixel_um = PIXEL_UM if pixel_um is None else pixel_um
    side = max(3, int(round(GEOMETRY_OPEN_UM / pixel_um))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
    opened = cv2.morphologyEx(leaflet.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    if not opened.any():
        return leaflet
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        opened.astype(np.uint8), 8)
    if count > 2:
        opened = labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    return opened


def midplane_reliable(rotated: np.ndarray, min_aspect: float = 3.0):
    """Le plan median est-il exploitable sur cette coupe redressee ?

    `straighten_matrix` ORIENTE la coupe, elle ne la deplie pas : c'est une
    rotation rigide vers l'axe de `cv2.minAreaRect`. La ligne mediane n'est donc
    definie que si la coupe redressee est une BANDE, c'est-a-dire une fonction a
    une seule valeur de x. Deux mecanismes distincts font echouer cette
    condition, tous deux constates sur ce jeu :

    - des composantes parasites faussent l'orientation. Sur NOUR_0543_2, 56
      composantes pour une principale a 99.4 % : minAreaRect sur l'ensemble
      donne un rapport d'aspect de 1.19 a -0.2 degre, sur la principale seule
      1.98 a -50.3 degres. Cinquante degres d'ecart.
    - une bande unique mais assez courbee remplit une boite quasi carree. Sur
      NOUR_0473_1 et NOUR_1003_1, UNE seule composante et pourtant un rapport
      de 1.02 et 1.05 : une bande courbee ne peut pas etre serree par un
      rectangle, sa propre courbure lui fait occuper une aire carree.

    Nettoyer le masque est donc necessaire mais NON suffisant : meme reduite a
    sa composante principale, NOUR_0543_2 reste a 1.98, sous le seuil.

    Aucun cas de double traversee verticale -- une colonne coupant deux bras
    separes par un vide -- n'a ete confirme ici : la part de colonnes ambigues
    reste entre 0.002 et 0.04 partout, y compris sur les coupes rejetees. Le
    critere correspondant est conserve parce qu'il coute peu et couvre une forme
    de pli que ce jeu ne contient pas.

    Le correctif de fond est un repere curviligne : abscisse le long de la ligne
    mediane ordonnee, puis distance suivant la normale locale. Ce controle n'est
    qu'un garde-fou, pas une geometrie correcte.

    Renvoie (ok, diagnostic).
    """
    ys, xs = np.nonzero(rotated)
    if ys.size == 0:
        return False, {"aspect": 0.0, "span_median": 0.0}
    width = xs.max() - xs.min() + 1
    height = ys.max() - ys.min() + 1
    aspect = width / height if height else 0.0

    spans, ambiguous, occupied = [], 0, 0
    for column in range(rotated.shape[1]):
        rows = np.nonzero(rotated[:, column])[0]
        if not rows.size:
            continue
        occupied += 1
        spans.append(rows[-1] - rows[0] + 1)
        # Ambiguite verticale : deux segments SUBSTANTIELS separes par un grand
        # vide. C'est la violation directe de l'hypothese de mid_plane, alors
        # que le rapport d'aspect n'en est qu'un indice global. Un simple
        # decrochement de bord ne compte pas.
        pieces = np.split(rows, np.where(np.diff(rows) != 1)[0] + 1)
        thick = [q for q in pieces if len(q) >= 20]
        if len(thick) >= 2:
            gap = thick[1][0] - thick[0][-1]
            if gap >= 0.5 * max(len(thick[0]), len(thick[1])):
                ambiguous += 1
    span_median = float(np.median(spans)) if spans else 0.0
    ambiguous_share = ambiguous / occupied if occupied else 1.0

    # Le rapport d'aspect suffit a separer les deux populations : 6 a 9 quand le
    # redressement reussit, 1.0 a 1.4 quand il echoue. La borne basse sur
    # l'epaisseur ne rattrape que le cas degenere ou la rotation a place la
    # coupe en travers et ou les colonnes ne coupent qu'un lisere de tissu.
    # Epaisseur INDEPENDANTE, tiree de la transformee de distance : invariante
    # par rotation, sans reference biologique ni calibration. Pour une bande
    # d'epaisseur constante, 2*DT_p95 vaut environ 95 % de l'epaisseur reelle.
    import cv2 as _cv2
    dt = _cv2.distanceTransform(rotated.astype(np.uint8), _cv2.DIST_L2,
                                _cv2.DIST_MASK_PRECISE)
    inside = dt[rotated]
    dt2 = 2.0 * float(np.percentile(inside, 95)) if inside.size else 0.0
    ratio = span_median / dt2 if dt2 else float("nan")

    ok = (aspect >= min_aspect and span_median >= 10.0
          and ambiguous_share <= 0.25)
    return ok, {"aspect": round(float(aspect), 2),
                "span_median": round(span_median, 1),
                "dt2_px": round(dt2, 1),
                "ratio": round(float(ratio), 3) if np.isfinite(ratio) else float("nan"),
                "ambiguous_share": round(float(ambiguous_share), 3)}


def mid_plane(leaflet: np.ndarray, *masks: np.ndarray, pixel_um=None,
              already_clean: bool = False):
    """The leaflet's mid-plane, and every pixel's distance from it.

    The mid-plane is taken column by column in the straightened frame, as the
    midpoint between the upper and the lower face. Distance to it is the
    Euclidean distance to that line once rasterised, not the former
    |y - mid(x)| / sqrt(1 + slope^2), which measured distance to the LOCAL
    TANGENT rather than to the nearest point of the curve. It therefore follows the
    section's curvature, which makes the distance to it a genuine position
    THROUGH THE THICKNESS: a bundle keeps a constant normal distance from the
    mid-plane -- which is NOT its depth below the epidermis, a different
    quantity read from the distance transform -- whether the section came to rest flat or curved on the slide. A single
    straight axis would instead count curvature as though it were depth.

    Decomposing the variance of the curvature index over the 148 admissible
    sections, AFTER the geometric correction, attributes 6.9 % of it to species
    and 1.0 % to site; the remaining 93.1 % lies between sections of one species
    at one site. The comparison with two other traits measured the same way on
    the same sections is what makes this conclusive: species explains 31.7 % of
    the bundles' share of the second moment and 37.2 % of leaflet thickness.
    The method separates species perfectly well when there is something to
    separate. Curvature in these images is therefore dominated by how the
    section was mounted, not by the plant's architecture.

    An earlier implementation used the ridge of the distance transform. It broke
    into a staircase and vanished at the leaflet's tapered tips, where the
    "distance to the axis" became a distance ALONG the blade and peaked at
    300 um on a section 205 um thick.

    Distance is the Euclidean distance to the rasterised mid-line, not a
    vertical offset and no longer a distance to the local tangent,
    so that steeply inclined stretches are not overstated.

    Returns ``(offset, leaflet, [masks], mid_line)``, all in the straightened
    frame.
    """
    # Nettoyage prealable : voir geometry_mask. Le masque d'origine reste
    # celui des surfaces ; seule la geometrie utilise celui-ci.
    # section_traits nettoie deja : un second passage serait idempotent mais
    # l'API doit garantir un nettoyage UNIQUE, pas s'en remettre a une
    # propriete mathematique.
    if not already_clean:
        leaflet = geometry_mask(leaflet, pixel_um)
    matrix, size = straighten_matrix(leaflet)

    def rotate(mask):
        return cv2.warpAffine(
            mask.astype(np.uint8), matrix, size, flags=cv2.INTER_NEAREST
        ) > 0

    rotated = rotate(leaflet)
    rows = np.arange(rotated.shape[0], dtype=np.float64)[:, None]
    occupied = rotated.any(axis=0)
    top = np.where(rotated, rows, np.inf).min(axis=0)
    bottom = np.where(rotated, rows, -np.inf).max(axis=0)
    middle = np.where(
        occupied,
        (np.where(occupied, top, 0.0) + np.where(occupied, bottom, 0.0)) / 2.0,
        np.nan,
    )

    if not occupied.any():
        zeros = np.zeros(rotated.shape, dtype=np.float64)
        return zeros, rotated, [rotate(m) for m in masks], middle

    columns = np.arange(rotated.shape[1])
    middle = np.interp(columns, columns[occupied], middle[occupied])
    window = max(3, int(occupied.sum()) // 30) | 1          # smooth the outline
    padded = np.r_[
        np.full(window // 2, middle[0]), middle, np.full(window // 2, middle[-1])
    ]
    middle = np.convolve(padded, np.ones(window) / window, mode="valid")

    # Distance au polyligne median RASTERISE, et non plus l'approximation
    # |y - mid(x)| / sqrt(1 + pente^2), qui mesure la distance a la TANGENTE
    # locale et non au point le plus proche de la courbe. cv2.polylines relie
    # les sommets consecutifs : sans cela une ligne a forte pente laisse des
    # trous entre colonnes et la transformee de distance surestime localement.
    #
    # Mesure de l'ecart sur les sorties finales, coupes a plan median valide :
    # 0.3 a 0.5 % sur I_leaflet_flat_um4 et 0.01 sur curvature_index, mais
    # jusqu'a +5.1 % sur I_bundle_flat_um4 et environ +5 % en relatif sur
    # I_bundle_share_flat (NOUR_2009_4 : 0.1360 -> 0.1426). L'erreur s'annule
    # en moyenne sur la foliole entiere et NON sur les faisceaux, qui sont
    # localises : c'est la part des faisceaux qui est la plus sensible.
    # L'approximation n'etait donc pas la source des valeurs aberrantes, et le
    # calcul exact ne les corrige pas non plus -- NOUR_0543_2 passe de 0.09 a
    # 0.17, toujours impossible. Il est adopte parce qu'il ne coute rien et
    # supprime une approximation dont on n'a plus a discuter.
    #
    # Ce n'est PAS la distance au polyligne continu : c'est la distance a sa
    # rasterisation, donc juste a environ un pixel pres. DIST_MASK_PRECISE
    # calcule un vrai euclidien plutot que l'approximation par masque 5x5.
    columns_index = np.arange(middle.size)
    valid = occupied & np.isfinite(middle)
    line = np.zeros(rotated.shape, dtype=np.uint8)
    if valid.sum() > 1:
        points = np.column_stack([
            columns_index[valid],
            np.clip(np.round(middle[valid]), 0, rotated.shape[0] - 1),
        ]).astype(np.int32)
        cv2.polylines(line, [points], False, 1, 1)
    offset = cv2.distanceTransform(1 - line, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return offset, rotated, [rotate(m) for m in masks], middle


#: Physical size of the morphological kernel. Fixing it in pixels silently
#: changes the operation with the resolution: a 3x3 kernel spans 1.1 um at the
#: working scale, where it erodes cavities only 5 px across, and 0.6 um at
#: acquisition resolution. Declaring it in micrometres keeps it one operation.
MORPHOLOGY_UM = 0.7


def _lumen_objects(mask: np.ndarray, min_diameter_um: float,
                   pixel_um: float | None = None) -> np.ndarray:
    """Connected components of the lumen, after morphological cleaning.

    An opening drops isolated pixels, a closing rejoins cavities split by a
    shadow, and the size floor removes whatever remains below the scale of a
    cell cavity. Without the floor the map fragments into thousands of single
    pixels and the median diameter falls to 1.4 um, which is noise rather than
    anatomy.

    ``pixel_um`` allows a resolution other than the working scale, in practice
    acquisition resolution, where the median lumen spans 18 px rather than 9 and
    its area is therefore sampled four times as finely.
    """
    pixel_um = PIXEL_UM if pixel_um is None else pixel_um
    side = max(3, int(round(MORPHOLOGY_UM / pixel_um))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    floor_px = max(1, int(np.pi * (min_diameter_um / pixel_um / 2) ** 2))
    count, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    return np.array(
        [
            stats[i, cv2.CC_STAT_AREA]
            for i in range(1, count)
            if stats[i, cv2.CC_STAT_AREA] >= floor_px
        ],
        dtype=np.float64,
    )


def strip_length(middle, occupied):
    """Longueur d'ARC de la ligne mediane, en pixels du repere redresse.

    On somme sqrt(1 + pente^2) colonne par colonne plutot que de compter les
    colonnes : une portion inclinee est plus longue que sa projection. Sur une
    bande parfaitement horizontale les deux coincident, et l'ecart grandit avec
    la courbure -- ce qui est precisement le cas a corriger.

    Renvoie None si la ligne mediane n'est pas exploitable ; l'appelant retombe
    alors sur la corde du rectangle englobant, moins juste mais toujours
    definie.

    RESIDU CONNU, ET POURQUOI ON N'Y TOUCHE PAS. Mesure sur 120 coupes contre
    l'epaisseur lue colonne par colonne dans le repere redresse : la corde
    donnait +2.1 % en mediane et +6.0 % sur les coupes les plus courbees ;
    l'arc donne -2.5 % et -0.8 %. La CORRELATION de l'erreur avec la courbure
    tombe de +0.253 a +0.139, et c'est le seul critere solide ici, car il ne
    demande pas de connaitre la vraie valeur -- il demande seulement que
    l'epaisseur ne depende pas du montage de la lame.

    L'erreur absolue, elle, ne bouge pas : 4.7 % contre 5.0 %. Ces deux
    valeurs sont indiscernables parce que la reference n'est PAS un etalon :
    une hauteur de colonne vaut l'epaisseur reelle divisee par cos(pente), donc
    elle surestime, et elle surestime davantage la ou la coupe est courbee.
    Aucune mesure manuelle d'epaisseur n'existe dans ce jeu pour trancher
    l'exactitude absolue.

    Le residu vient du tremblement de la ligne mediane : sommer sqrt(1+pente^2)
    sur une ligne qui tremble gonfle sa longueur. Un lissage supplementaire le
    reduit -- fenetre 401, ecart -1.5 % et correlation +0.106 -- mais le gain
    est faible et la fenetre serait reglee contre une reference biaisee. C'est
    la facon classique d'obtenir un chiffre qui parait meilleur sans etre plus
    juste, donc on s'en abstient.
    """
    if middle is None or not np.any(occupied):
        return None
    line = np.asarray(middle, dtype=np.float64)[occupied]
    if line.size < 2 or not np.all(np.isfinite(line)):
        return None
    return float(np.sum(np.sqrt(1.0 + np.diff(line) ** 2))) + 1.0


def _principal_axes(mask: np.ndarray):
    """Centroid and principal axes of the section.

    The MAJOR axis runs along the strip -- the leaflet's width seen in section;
    the MINOR axis crosses its thickness. Bending a leaflet happens about the
    major axis, so that is the neutral axis.
    """
    ys, xs = np.nonzero(mask)
    if ys.size < 3:
        return None

    cy, cx = ys.mean(), xs.mean()
    _, vectors = np.linalg.eigh(np.cov(np.vstack([ys - cy, xs - cx])))
    return (cy, cx), vectors[:, 0]          # smallest eigenvalue: the thickness


def _second_moment(mask: np.ndarray, centre, minor) -> float:
    """Second moment of area about a straight neutral axis, in px^4.

    Each pixel counts as one unit of area at its signed distance from the axis,
    measured along the thickness direction.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0.0

    cy, cx = centre
    distance = (ys - cy) * minor[0] + (xs - cx) * minor[1]
    return float(np.sum(distance.astype(np.float64) ** 2))


def _median_nearest_neighbour(points) -> float | None:
    """Median distance to the nearest neighbour, in px; None below two points."""
    if len(points) < 2:
        return None

    array = np.asarray(points, dtype=np.float64)
    squared = ((array[:, None, :] - array[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(squared, np.inf)
    return float(np.median(np.sqrt(squared.min(axis=1))))


def section_traits(leaflet: np.ndarray, instances: np.ndarray, lumen: np.ndarray,
                   fine: dict | None = None,
                   pixel_um: float | None = None) -> dict:
    """Traits of one section, from masks at the working scale.

    ``leaflet``    boolean footprint of the leaflet
    ``instances``  bundle instance map, 0 being background
    ``lumen``      boolean lumen mask, already confined to the bundles
    ``fine``       optional ``{lumen, bundle, leaflet, pixel_um}`` at acquisition
                   resolution

    Lumina are small: the median spans 9 px at the working scale against 18 at
    acquisition resolution, and the smallest brush the detection floor. Every
    quantity that depends on them — non-luminal area, object count, diameters,
    vessels — is therefore measured on the original image whenever the caller can
    supply it. Geometry stays at the working scale: it concerns objects a hundred
    times larger and would gain nothing.

    Values are returned already converted to um, um^2 or um^4.
    """
    # Resolution reelle de CETTE coupe : le second systeme d'acquisition
    # n'a pas la meme taille de pixel (voir KEYENCE_SECTIONS dans config).
    pixel_um = PIXEL_UM if pixel_um is None else pixel_um
    out: dict = {}
    pixel_area = pixel_um ** 2
    bundle = instances > 0

    # --- wall --------------------------------------------------------------
    # Descriptive rather than histological: every bundle pixel not recognised as
    # a cavity lands here, which includes genuine walls but also cell contents,
    # stained conducting tissue and shadows. Calling it wall would assert an
    # identity that has not been validated; the column names are kept for
    # continuity with published tables.
    if fine is not None:
        fine_pixel = fine["pixel_um"]
        bundle_px, lumen_px = int(fine["bundle"].sum()), int(fine["lumen"].sum())
        leaflet_px, fine_area = int(fine["leaflet"].sum()), fine_pixel ** 2
    else:
        fine_pixel, fine_area = pixel_um, pixel_area
        bundle_px, lumen_px = int(bundle.sum()), int(lumen.sum())
        leaflet_px = int(leaflet.sum())

    wall_px = bundle_px - lumen_px
    out["wall_area_um2"] = round(wall_px * fine_area, 1)
    out["wall_over_bundle"] = round(wall_px / max(bundle_px, 1), 5)
    out["wall_over_leaflet"] = round(wall_px / max(leaflet_px, 1), 5)
    out["ground_area_um2"] = round(int(leaflet.sum() - bundle.sum()) * pixel_area, 1)

    # A L'ECHELLE DE LA FOLIOLE, la composition a TROIS parties et non deux :
    #
    #     C = grands lumens >= seuil   (definis par la TAILLE, pas l'identite)
    #     R = reste du faisceau        (petits lumens et materiel non luminal)
    #     G = tissu fondamental        (hors faisceaux)
    #
    # Trois parties EXCLUSIVES, donc deux coordonnees independantes :
    #
    #     log(C/R)      allocation a l'interieur du faisceau, ecrite plus bas
    #                   sous log_conduit_over_bundle_rest ;
    #     log((C+R)/G)  part de la foliole investie dans le vasculaire, ici.
    #
    # Ensemble elles determinent entierement la composition, alors que
    # ratio_BU_LM et vessel_area_frac_BU, etant des parts d'un tout, sont
    # contraints et ne peuvent pas entrer tels quels dans un modele.
    #
    # Pour qui veut des coordonnees ORTHONORMEES (ilr) : ilr1 = log(C/B)/sqrt(2)
    # et ilr2 = sqrt(2/3)*log(sqrt(C*B)/G). Les log-rapports bruts sont
    # conserves ici parce qu'ils se lisent directement.
    #
    # CE TRAIT NE DEPEND PAS DES LUMENS et se calcule donc hors du bloc qui les
    # resout : une coupe sans lumen detecte a malgre tout une part vasculaire.
    # Tout est compte dans les MEMES pixels, le rapport est sans dimension.
    _vasc = float(bundle_px)
    _ground = float(leaflet_px) - _vasc
    out["log_vascular_over_ground"] = (
        round(float(np.log(_vasc / _ground)), 4)
        if _vasc > 0 and _ground > 0 else float("nan"))

    # --- geometry ----------------------------------------------------------
    # UN SEUL masque nettoye pour toute la geometrie. L'appliquer au seul plan
    # median melangerait un numerateur bruite et un denominateur nettoye dans
    # curvature_index, et laisserait l'epaisseur et la profondeur sur le masque
    # d'origine. Les surfaces et fractions publiees restent, elles, calculees
    # sur le masque d'origine : voir geometry_mask.
    geometry_leaflet = geometry_mask(leaflet, pixel_um)
    geometry_bundle = bundle & geometry_leaflet
    out["geometry_area_removed"] = round(
        1.0 - geometry_leaflet.sum() / max(int(leaflet.sum()), 1), 5)

    axes = _principal_axes(geometry_leaflet)
    if axes is not None:
        centre, minor = axes
        straight_leaflet = _second_moment(geometry_leaflet, centre, minor) * pixel_um ** 4
        straight_bundle = _second_moment(geometry_bundle, centre, minor) * pixel_um ** 4

        offset, rotated_leaflet, (rotated_bundle,), middle_line = mid_plane(
            geometry_leaflet, geometry_bundle, pixel_um=pixel_um,
            already_clean=True)
        # Le plan median n'a de sens que si la coupe redressee est une bande.
        # Sinon les traits qui en derivent sont ecrits a NaN plutot qu'a une
        # valeur fausse que rien en aval ne distinguerait.
        reliable, _diag = midplane_reliable(rotated_leaflet)
        out["midplane_reliable"] = int(reliable)
        # Mesures brutes : elles permettent de rejouer n'importe quel seuil sur
        # le CSV sans refaire la passe de production.
        out["midplane_span_px"] = _diag["span_median"]
        out["midplane_dt2_px"] = _diag.get("dt2_px", float("nan"))
        out["midplane_ratio"] = _diag.get("ratio", float("nan"))
        out["midplane_aspect"] = _diag["aspect"]
        flat_leaflet = float(np.sum(offset[rotated_leaflet] ** 2)) * pixel_um ** 4
        flat_bundle = float(np.sum(offset[rotated_bundle] ** 2)) * pixel_um ** 4

        out["I_leaflet_um4"] = round(straight_leaflet, 1)
        out["I_bundle_um4"] = round(straight_bundle, 1)
        out["I_leaflet_flat_um4"] = round(flat_leaflet, 1)
        out["I_bundle_flat_um4"] = round(flat_bundle, 1)

        # Share of the geometric stiffness carried by the bundles. Compare it
        # with bundle_over_leaflet: a larger share means the bundles sit towards
        # the faces and work harder than their area alone would suggest.
        out["I_bundle_share"] = round(
            straight_bundle / straight_leaflet, 5) if straight_leaflet else 0.0
        out["I_bundle_share_flat"] = round(
            flat_bundle / flat_leaflet, 5) if flat_leaflet else 0.0

        # What the curvature adds. One means a flat section.
        #
        # Do NOT use this as an explanatory variable: see :func:`mid_plane`, it
        # measures the mounting far more than the plant. Median 1.70, p90 4.74 across the 148 admissible sections after the geometric correction
        # across the annotated sections, so curvature typically inflates the
        # second moment by 60 % and by up to a factor of four. Any biological
        # analysis should rest on I_bundle_share_flat, which excludes it.
        out["curvature_index"] = round(
            straight_leaflet / flat_leaflet, 3) if flat_leaflet else 0.0

        # Mean thickness = area divided by the length of the strip.
        #
        # The length is the ARC of the mid-plane, not the chord of the bounding
        # rectangle. A curved strip is longer than the straight line joining its
        # ends, so the chord understated the length and the thickness came out
        # too large -- by 2.1 % at the median and 16.1 % at the ninth decile
        # over 120 sections, and the error followed the curvature acquired on
        # the slide rather than anything in the plant.
        ys, xs = np.nonzero(geometry_leaflet)
        points = np.column_stack([xs, ys]).astype(np.int32)
        _, (width, height), _ = cv2.minAreaRect(points)
        arc = strip_length(middle_line, rotated_leaflet.any(axis=0))
        if arc is not None and arc > 1.0:
            out["leaflet_thickness_um"] = round(
                geometry_leaflet.sum() / arc * pixel_um, 2)
        else:
            out["leaflet_thickness_um"] = round(
            geometry_leaflet.sum() / max(max(width, height), 1) * pixel_um, 2
        )
    else:
        for field in GEOMETRY_FIELDS:
            out[field] = 0.0

    # --- organisation ------------------------------------------------------
    areas, centroids = [], []
    for index in (i for i in np.unique(instances) if i > 0):
        selected = instances == index
        ys, xs = np.nonzero(selected)
        areas.append(int(selected.sum()))
        centroids.append((ys.mean(), xs.mean()))
    areas = np.array(areas, dtype=np.float64)

    if areas.size:
        diameters = 2 * np.sqrt(areas / np.pi) * pixel_um
        out["bundle_area_mean_um2"] = round(float(areas.mean()) * pixel_area, 1)
        out["bundle_area_max_um2"] = round(float(areas.max()) * pixel_area, 1)
        out["bundle_area_cv"] = round(float(areas.std() / areas.mean()), 4)
        out["bundle_diameter_median_um"] = round(float(np.median(diameters)), 2)

        # Depth of each centroid beneath the epidermis, by distance transform,
        # which is immune to how the section was laid on the slide.
        depth = cv2.distanceTransform(geometry_leaflet.astype(np.uint8),
                                      cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        # Les centroides doivent etre recalcules SUR le masque de geometrie :
        # si le nettoyage retire la region qui contient un centroide d'origine,
        # sa profondeur devient artificiellement nulle et tire la mediane vers
        # le bas. Un faisceau entierement retire est compte et signale, pas
        # silencieusement lu a zero.
        depths, dropped = [], 0
        for index in (i for i in np.unique(instances) if i > 0):
            piece = (instances == index) & geometry_leaflet
            if not piece.any():
                dropped += 1
                continue
            ys_, xs_ = np.nonzero(piece)
            depths.append(depth[int(round(ys_.mean())), int(round(xs_.mean()))])
        out["bundles_outside_geometry"] = dropped
        # Aucun faisceau ne survit au nettoyage : ecrire zero serait une fausse
        # mesure, meme accompagnee du compteur. NaN dit l'absence.
        median_depth = float(np.median(depths)) if depths else float("nan")
        out["bundle_depth_median_um"] = round(median_depth * pixel_um, 2)

        # Relative to the half-thickness: 1 at the core, 0 against the epidermis.
        half = out.get("leaflet_thickness_um", 0.0) / 2 or 1.0
        out["bundle_depth_relative"] = round(median_depth * pixel_um / half, 4)

        spacing = _median_nearest_neighbour(centroids)
        out["bundle_spacing_um"] = round(spacing * pixel_um, 2) if spacing else 0.0
    else:
        for field in ORGANISATION_FIELDS:
            out[field] = 0.0

    # --- lumen resolved into objects ---------------------------------------
    if fine is not None:
        lumen_areas = _lumen_objects(fine["lumen"], LUMEN_MIN_DIAMETER_UM, fine_pixel)
    else:
        lumen_areas = _lumen_objects(lumen, LUMEN_MIN_DIAMETER_UM, pixel_um)
    if lumen_areas.size:
        diameters = 2 * np.sqrt(lumen_areas / np.pi) * fine_pixel
        vessels = diameters >= VESSEL_MIN_DIAMETER_UM
        out["n_lumen"] = int(lumen_areas.size)
        out["lumen_diameter_median_um"] = round(float(np.median(diameters)), 2)
        # The 90th percentile is steadier than the maximum, which rests on a
        # single object.
        out["lumen_diameter_p90_um"] = round(float(np.percentile(diameters, 90)), 2)
        out["lumen_diameter_max_um"] = round(float(diameters.max()), 2)

        # Capacite hydraulique implicite de la geometrie des conduits.
        #
        # Hagen-Poiseuille donne K proportionnel a d^4 pour un conduit
        # cylindrique : un lumen de 30 um vaut seize fois un de 15 um. Sommer
        # les d^4 donne donc une grandeur dominee par les grands conduits, et
        # c'est ce qui la rend robuste la ou n_vessels ne l'est pas. Entre 8 et
        # 20 um de seuil, le comptage chute d'un facteur 17 sur les 2486
        # faisceaux de production, de 6.09 a 0.35 par faisceau, tandis que la
        # somme des puissances quatre, si on lui applique le MEME plancher, ne
        # perd que 11.6 % -- 6.9 % sur les 1151 faisceaux annotes. Elle est
        # donc environ 8 a 14 fois moins sensible au seuil que le comptage,
        # soit un ordre de grandeur. Publiee, elle n'emploie aucun seuil. Les
        # petits objets n'y pesent rien. Aucun seuil de vaisseau n'entre donc dans ce calcul :
        # tous les lumens y contribuent, ponderes par leur importance
        # hydraulique.
        #
        # Ce n'est PAS une conductance mesuree. Les resistances des ponctuations
        # et des perforations, la longueur des conduits, les embolies et surtout
        # le trajet extra-xylemien font que la conductance reelle d'une feuille
        # est plusieurs fois inferieure a ce que l'anatomie suggere : Martre &
        # Durand (2001) mesurent chez Festuca une conductance axiale environ
        # trois fois plus faible que l'estimation anatomique. A presenter comme
        # une capacite hydraulique theorique, jamais comme une conductance.
        out["conduit_sum_d4_um4"] = round(float(np.sum(diameters ** 4)), 1)
        # Diametre pondere hydrauliquement : le diametre qu'auraient n conduits
        # identiques offrant la meme somme de d^4.
        out["conduit_dh_um"] = round(
            float((np.sum(diameters ** 4) / diameters.size) ** 0.25), 2)
        out["n_vessels"] = int(vessels.sum())
        out["vessel_area_over_bundle"] = round(
            float(lumen_areas[vessels].sum()) / max(float(bundle_px), 1.0), 5
        )

        # Composition a TROIS parties exclusives, et deux coordonnees.
        #
        #     C  grands lumens, >= VESSEL_MIN_DIAMETER_UM
        #     R  reste du faisceau : petits lumens ET materiel non luminal
        #     G  tissu fondamental, hors faisceaux
        #
        # Une composition a D parties demande D-1 coordonnees. Une version
        # precedente definissait QUATRE compartiments et n'en donnait que deux,
        # et son denominateur -- la paroi seule -- ignorait les petits lumens.
        # Trois parties exclusives reglent les deux defauts.
        #
        # Les fractions brutes ne peuvent pas entrer dans un modele : elles sont
        # contraintes par leur somme, si bien que correler deux parts d'un meme
        # tout mesure en partie cette contrainte. Le log-rapport y echappe.
        #
        # NaN quand C vaut zero. Pas de pseudocompte silencieux : un
        # remplacement de zeros doit rester sous l'aire du plus petit conduit
        # detectable, pi*(seuil/2)^2, et rester un choix explicite de l'analyste.
        #
        # C EST DEFINI PAR LA TAILLE. "Grands lumens", pas "vaisseaux" : le seuil
        # ne separe pas vaisseaux et fibres, et le phloeme n'est pas detecte.
        #
        # Les fractions sont contraintes par leur somme -- ce qui est cavite
        # n'est pas paroi -- si bien que correler deux parts d'un meme tout
        # mesure en partie la contrainte plutot que la biologie. Le log-rapport
        # y echappe : il varie librement et une difference y a le meme sens quel
        # que soit le niveau de depart. C'est la forme sous laquelle ce compromis
        # doit entrer dans un modele.
        #
        # Numerateur et denominateur sont comptes dans les MEMES pixels, donc le
        # rapport est sans dimension et la calibration s'annule.
        #
        # NaN quand aucun conduit n'atteint le seuil, plutot qu'une valeur
        # inventee -- c'est le cas de 77 % des faisceaux annotes a 11 um, bien
        # moins souvent d'une coupe entiere, qui en porte plusieurs. Un
        # remplacement de zeros en aval doit rester SOUS l'aire du plus petit
        # conduit detectable, pi*(seuil/2)^2 soit 95 um2 a 11 um, sans quoi il
        # affirmerait avoir vu ce qui n'a pas ete vu.
        _cond = float(lumen_areas[vessels].sum())
        _rest = float(bundle_px) - _cond
        out["log_conduit_over_bundle_rest"] = (
            round(float(np.log(_cond / _rest)), 4)
            if _cond > 0 and _rest > 0 else float("nan"))
    else:
        for field in LUMEN_FIELDS:
            out[field] = 0
        # Zero conduit n'est pas un log-rapport nul, c'est une absence : sans
        # conduit le rapport n'existe pas. Ecrire 0 le ferait passer pour un
        # faisceau ou conduits et paroi s'equilibrent.
        out["log_conduit_over_bundle_rest"] = float("nan")

    # Traits qui reposent sur le plan median : sans repere fiable, une valeur
    # est pire qu'une absence, car rien en aval ne la distingue.
    if MIDPLANE_INVALIDATE and out.get("midplane_reliable") == 0:
        for field in ("I_leaflet_flat_um4", "I_bundle_flat_um4",
                      "I_bundle_share_flat", "curvature_index",
                      "leaflet_thickness_um", "bundle_depth_relative"):
            if field in out:
                out[field] = float("nan")

    return out


WALL_FIELDS = [
    "wall_area_um2", "wall_over_bundle", "wall_over_leaflet", "ground_area_um2",
]
GEOMETRY_FIELDS = [
    "I_leaflet_um4", "I_bundle_um4", "I_bundle_share",
    "geometry_area_removed", "bundles_outside_geometry",
    "midplane_reliable", "midplane_span_px", "midplane_dt2_px",
    "midplane_ratio", "midplane_aspect",
    "I_leaflet_flat_um4", "I_bundle_flat_um4", "I_bundle_share_flat",
    "curvature_index", "leaflet_thickness_um",
]
ORGANISATION_FIELDS = [
    "bundle_area_mean_um2", "bundle_area_max_um2", "bundle_area_cv",
    "bundle_diameter_median_um", "bundle_depth_median_um",
    "bundle_depth_relative", "bundle_spacing_um",
]
LUMEN_FIELDS = [
    "n_lumen", "lumen_diameter_median_um", "lumen_diameter_p90_um",
    "lumen_diameter_max_um", "n_vessels", "vessel_area_over_bundle",
    "conduit_sum_d4_um4", "conduit_dh_um",
    "log_conduit_over_bundle_rest",
    "log_vascular_over_ground",
]

#: Every trait column, in the order they are written to the results file.
TRAIT_FIELDS = WALL_FIELDS + GEOMETRY_FIELDS + ORGANISATION_FIELDS + LUMEN_FIELDS
