"""
Pixel positions of the LOKI detector banks from their 8 corner positions.

This is the geometry core used to produce the pixel data of the NeXus/JSON
instrument description (see nexus_json.py). The tube grid is built with
simple vector steps between the corners, and the straw layout inside the tubes
is taken from the LOKI Geant4 geometry (G4GeoLoki, https://github.com/ess-dg/dgcode_projects/tree/main/LOKI/G4GeoLoki),
which it reproduces exactly (see the tests).

Input: the 8 corners of a bank, in the REFERENCE_BANKS_DATA format [mm]:
  'A': [front layer top tube, back layer top tube, front layer bottom tube,
        back layer bottom tube] at the FIRST pixel end of the straws,
  'B': the same 4 tubes at the LAST pixel end.
  The points are on the tube axis (= centre straw axis) at the straw ends.
  "top/bottom" = lowest/highest local tube id (left/right for vertical banks).

Model of a bank:
  - 4 layers x R rows of tubes (R = tubes per layer = 2 * packs), tube centre
        tube(layer, row) = A[0] + layer * layer_step + row * row_step
    with layer_step = (A[1] - A[0]) / 3 and row_step = (A[2] - A[0]) / (R - 1)
    (a parallelogram grid: the layers are shifted along the rows, because the
    packs are tilted by 13.45 degrees);
  - 7 straws per tube in a fixed hexagonal pattern around the tube axis
    (straw 3 is the centre straw), see STRAW_POSITIONS_PACK_FRAME;
  - N pixels per straw along the straw, pixel k centred at (k + 0.5) * L / N
    from the first pixel end (L = straw length = |B[0] - A[0]|).

Straw pattern mirroring: in banks 1, 2, 5 and 6 the packs are mounted upside
down, so these banks are MIRROR images of the banks 3, 4, 7, 8 (and 0), not
rotated copies. In a frame of (row direction, depth direction) the straw
pattern and the layer step of such a bank are mirrored along the row
direction. Whether a bank is mirrored is derived from its corners: the layer
step points towards higher tube ids in mirrored banks, towards lower ones in
the others. It must agree with MIRRORED_BANK_IDS (checked).

Only A[0], A[1], A[2] and B[0] define the model. The corners are checked
(ValueError otherwise):
  - they are finite numbers;
  - the other corners are consistent with the model (A[3] = A[1] + A[2] - A[0],
    B[i] - A[i] = B[0] - A[0]) within CORNER_TOLERANCE;
  - the tube pitch in a layer (|row_step|) and between the layers
    (|layer_step|) are the nominal 28.40 / 27.00 mm within PITCH_TOLERANCE
    (corners of another bank, of the wrong size, or with A[1] / A[2] swapped
    fail this);
  - the straw length (|B[0] - A[0]|) is the nominal straw length of the bank
    within CORNER_TOLERANCE (the corners are the straw ends after the trimming of
    the end electronics: e.g. untrimmed tube ends or pixel-centre corners fail
    this);
  - the frame (along, row direction, layer direction) is right-handed, as for
    every correctly ordered bank: swapped 'A' / 'B' (first / last pixel ends)
    make it left-handed;
  - the straw pattern (mirrored or not, see above) matches MIRRORED_BANK_IDS.
    For corners from calibration.py this always holds (the calibration builds
    the layer shift from the same list); it guards hand-entered corners.

Bank position (the bank translation, origin of the local offsets), selectable;
the absolute pixel positions are the same for both:
  FIRST_PIXEL_END ("first_pixel_end", default): the corner A[0] itself, i.e.
      the FIRST PIXEL END of the centre straw of tube 0 (front layer, top tube,
      or leftmost tube in vertical banks), on the tube axis. It is not a pixel
      position: the first pixel centre of that straw is half a pixel further
      along it. This is the same convention as in the current loki.json.
  CENTRE_OF_MASS ("centre_of_mass"): the mean of all pixel centres of the bank
      (equal weights), i.e. the centre of the tube grid, halfway along the
      straws. It does not depend on the number of pixels per straw.

Pixel reference point (what a pixel offset points to), selectable:
  PIXEL_CENTRE ("pixel_centre", default): the centre of the pixel,
      at (k + 0.5) * L / N along the straw;
  PIXEL_BASE   ("pixel_base"): the centre of the pixel's first face (the face
      towards the first pixel end), at k * L / N along the straw.
The pixel shape in the NeXus output (nexus_json.py) is set accordingly.

Pixel numbering (as in Geant4 and the ICD). NOTE: in this module "pixel id" means
the 1-based detector_number of the NeXus file (the ICD's pixelID), i.e. the 0-based
Geant4 pixel id + 1 (FIRST_PIXEL_ID = 1):
  pixel_id = bank_first_pixel_id + (tube * 7 + straw) * N + pixel_in_straw
  tube = layer * R + row (layer 0 = front layer, facing the sample).

Units: mm everywhere in this module.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .constants import (
    CENTRE_STRAW,
    DEFAULT_PIXELS_PER_STRAW,
    LAYERS_PER_BANK,
    NUMBER_OF_BANKS,
    STRAW_HEXAGON_ROTATION,
    STRAW_LENGTHS,
    STRAW_POSITIONS_IN_TUBE,
    STRAW_RADIUS,
    STRAW_TABLE_ROTATION_OFFSET,
    STRAWS_PER_TUBE,
    TUBE_GRID_BASE,
    TUBE_GRID_SIDE,
    TUBES_PER_BANK,
    TUBES_PER_LAYER,
    UPSIDE_DOWN_BANK_IDS,
)

__all__ = [
    "BankFrame",
    "BANK_ORIGINS",
    "BankPixels",
    "CENTRE_OF_MASS",
    "CENTRE_STRAW",
    "CORNER_TOLERANCE",
    "PITCH_TOLERANCE",
    "DEFAULT_PIXELS_PER_STRAW",
    "FIRST_PIXEL_END",
    "FIRST_PIXEL_ID",
    "MIRRORED_BANK_IDS",
    "NUMBER_OF_BANKS",
    "PIXEL_BASE",
    "PIXEL_CENTRE",
    "PIXEL_REFERENCES",
    "STRAW_POSITIONS_PACK_FRAME",
    "STRAW_RADIUS",
    "bank_first_pixel_id",
    "bank_origin_position",
    "centre_of_mass_offset",
    "bank_pixel_ids",
    "compute_bank_pixels",
    "compute_instrument_pixels",
    "local_pixel_offsets",
    "pixel_offsets_along_straw",
    "pixel_reference_fraction",
    "pixels_in_bank",
    "straw_offsets",
    "tube_offsets",
    "tubes_per_layer",
]

FIRST_PIXEL_ID = 1  # detector_number of the first pixel of bank 0 (ICD numbering)
FIRST_PIXEL_END = "first_pixel_end"  # bank origin = corner A[0]
CENTRE_OF_MASS = "centre_of_mass"  # bank origin = mean of the pixel centres
BANK_ORIGINS = (FIRST_PIXEL_END, CENTRE_OF_MASS)
MIRRORED_BANK_IDS = UPSIDE_DOWN_BANK_IDS  # only used as a cross-check: derived from the corners
CORNER_TOLERANCE = 0.1  # [mm] accepted inconsistency of the corners (rounding)
PITCH_TOLERANCE = 0.5  # [mm] accepted deviation of the tube pitches from nominal
PIXEL_CENTRE = "pixel_centre"  # pixel offsets point to the pixel centres
PIXEL_BASE = "pixel_base"  # pixel offsets point to the centre of the pixel's first face
PIXEL_REFERENCES = (PIXEL_CENTRE, PIXEL_BASE)
_PIXEL_REFERENCE_FRACTION = {PIXEL_CENTRE: 0.5, PIXEL_BASE: 0.0}  # position within the pixel

# Angle of the Geant4 tube frame in the (row, depth) frame of a normally mounted
# pack: 90 deg (frame convention) + the straw hexagon rotation (8.45 deg) + the
# offset of the Geant4 straw table (-30 deg) = 68.45 deg. (In Geant4 the tube is
# rotated by 8.45 - 30 - 13.45 deg in the pack, and the pack by +13.45 deg.)
# Checked against the Geant4 geometry in tests/test_bank_pixels.py.
_STRAW_PATTERN_ROTATION = np.radians(90.0 + STRAW_HEXAGON_ROTATION + STRAW_TABLE_ROTATION_OFFSET)
_STRAW_POSITIONS_TUBE_FRAME = np.array(STRAW_POSITIONS_IN_TUBE)


def _rotate_2d(points, angle):
    """Rotate (n, 2) points counterclockwise by angle [rad]."""
    c, s = np.cos(angle), np.sin(angle)
    return points @ np.array([[c, s], [-s, c]])


# Straw centres relative to the tube axis for a NORMALLY mounted pack, as
# (row, depth) coordinates [mm]: row = towards higher tube ids in the layer,
# depth = towards the back layers (away from the sample).
STRAW_POSITIONS_PACK_FRAME = _rotate_2d(_STRAW_POSITIONS_TUBE_FRAME, _STRAW_PATTERN_ROTATION)


# ---------------------------------------------------------------------------
# Bank frame from the 8 corners
# ---------------------------------------------------------------------------


def _unit(vector):
    return vector / np.linalg.norm(vector)


def tubes_per_layer(bank_id):
    """Number of tubes in a layer (= rows) of a bank."""
    return TUBES_PER_LAYER[bank_id]


@dataclass(frozen=True)
class BankFrame:
    """Geometry of a bank derived from its 8 corners (all vectors in mm).

    origin:       first pixel end of tube 0 (front layer, top) on its axis = A[0]
    along:        unit vector along the straws, from the first to the last pixel
    straw_length: effective straw length L
    row_step:     from a tube to the next one in the same layer (tube id + 1)
    layer_step:   from a tube to the one behind it (next layer, same row)
    row_axis:     unit vector of row_step
    depth_axis:   unit vector perpendicular to along and row_axis, towards the
                  back layers
    mirrored:     True if the packs are mounted upside down (mirrored bank)
    rows:         tubes per layer
    """

    origin: np.ndarray
    along: np.ndarray
    straw_length: float
    row_step: np.ndarray
    layer_step: np.ndarray
    row_axis: np.ndarray
    depth_axis: np.ndarray
    mirrored: bool
    rows: int

    @classmethod
    def from_corners(cls, corners, rows, tolerance=CORNER_TOLERANCE, straw_length=None):
        """Build the frame from corners in the REFERENCE_BANKS_DATA format.
        Raises ValueError if the corners fail the checks of the module docstring
        (tolerance [mm]: the accepted inconsistency of the corners; straw_length [mm]:
        the expected straw length, not checked if None)."""
        a = np.asarray(corners["A"], dtype=float)
        b = np.asarray(corners["B"], dtype=float)
        if a.shape != (4, 3) or b.shape != (4, 3):
            raise ValueError("corners must contain 4 'A' and 4 'B' points with 3 coordinates each")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise ValueError("corners must be finite numbers")
        straw_vector = b[0] - a[0]
        if (
            straw_length is not None
            and abs(np.linalg.norm(straw_vector) - straw_length) > tolerance
        ):
            raise ValueError(
                f"straw length {np.linalg.norm(straw_vector):.3f} mm (|B[0] - A[0]|) differs from the "
                f"nominal {straw_length} mm (tolerance {tolerance} mm): corners of untrimmed tube ends "
                "or of pixel centres?"
            )
        deviation = max(
            np.abs(a[3] - (a[1] + a[2] - a[0])).max(),
            np.abs((b - a) - straw_vector).max(),
        )
        if deviation > tolerance:
            raise ValueError(
                f"inconsistent corners: A[3] or B[1..3] deviate by {deviation:.3f} mm from the "
                f"bank defined by A[0], A[1], A[2], B[0] (tolerance {tolerance} mm)"
            )
        along = _unit(straw_vector)
        row_step = (a[2] - a[0]) / (rows - 1)
        layer_step = (a[1] - a[0]) / (LAYERS_PER_BANK - 1)
        # orthonormal (row, depth) frame perpendicular to the straws
        row_axis = _unit(row_step - np.dot(row_step, along) * along)
        depth = (
            layer_step - np.dot(layer_step, along) * along - np.dot(layer_step, row_axis) * row_axis
        )
        depth_axis = _unit(depth)
        row_pitch, layer_pitch = np.linalg.norm(row_step), np.linalg.norm(layer_step)
        if (
            abs(row_pitch - TUBE_GRID_SIDE) > PITCH_TOLERANCE
            or abs(layer_pitch - TUBE_GRID_BASE) > PITCH_TOLERANCE
        ):
            raise ValueError(
                f"tube pitches {row_pitch:.2f} mm (in a layer) and {layer_pitch:.2f} mm (between "
                f"layers) differ from the nominal {TUBE_GRID_SIDE} / {TUBE_GRID_BASE} mm: corners "
                "of another bank?"
            )
        if np.dot(np.cross(along, row_axis), depth_axis) <= 0:
            raise ValueError(
                "the corners form a left-handed frame: 'A' and 'B' (first / last pixel ends) "
                "swapped?"
            )
        # the layers are shifted towards higher tube ids only in mirrored banks
        mirrored = bool(np.dot(layer_step, row_axis) > 0)
        return cls(
            a[0],
            along,
            float(np.linalg.norm(straw_vector)),
            row_step,
            layer_step,
            row_axis,
            depth_axis,
            mirrored,
            rows,
        )


# ---------------------------------------------------------------------------
# Offsets of tubes, straws and pixels (relative to BankFrame.origin)
# ---------------------------------------------------------------------------


def tube_offsets(frame):
    """Tube axis offsets from the origin, shape (4 * rows, 3), in tube id order
    (tube = layer * rows + row)."""
    layers = np.arange(LAYERS_PER_BANK)[:, None, None]
    rows = np.arange(frame.rows)[None, :, None]
    offsets = layers * frame.layer_step + rows * frame.row_step
    return offsets.reshape(-1, 3)


def straw_offsets(frame):
    """Straw axis offsets from the tube axis, shape (7, 3), in straw id order.
    The pattern of a normally mounted pack is mirrored along the row direction
    for mirrored banks."""
    row_sign = -1.0 if frame.mirrored else 1.0
    row_coord = STRAW_POSITIONS_PACK_FRAME[:, 0:1] * row_sign
    depth_coord = STRAW_POSITIONS_PACK_FRAME[:, 1:2]
    return row_coord * frame.row_axis + depth_coord * frame.depth_axis


def pixel_reference_fraction(pixel_reference):
    """Position of the pixel reference point within the pixel, as a fraction of
    the pixel length from the pixel's first face: 0.5 (centre) or 0 (base)."""
    if pixel_reference not in PIXEL_REFERENCES:
        raise ValueError(
            f"pixel_reference must be one of {PIXEL_REFERENCES}, not {pixel_reference!r}"
        )
    return _PIXEL_REFERENCE_FRACTION[pixel_reference]


def pixel_offsets_along_straw(frame, pixels_per_straw, pixel_reference=PIXEL_CENTRE):
    """Offsets of the pixel reference points from the first pixel end of a
    straw, shape (N, 3): the pixel centres ((k + 0.5) * L / N) or the pixel
    bases (k * L / N), see the module docstring."""
    pixel_length = frame.straw_length / pixels_per_straw
    positions = (
        np.arange(pixels_per_straw) + pixel_reference_fraction(pixel_reference)
    ) * pixel_length
    return positions[:, None] * frame.along


def centre_of_mass_offset(frame):
    """Mean of all pixel centres of a bank relative to the frame origin (corner
    A[0]) [mm]: the mean tube axis position, the mean straw offset (0 up to
    rounding, the straws are symmetric around the tube axis) and half the straw
    length. The pixel centres are symmetric along each straw, so the result does
    not depend on the number of pixels per straw."""
    return (
        tube_offsets(frame).mean(axis=0)
        + straw_offsets(frame).mean(axis=0)
        + frame.along * frame.straw_length / 2
    )


def bank_origin_position(frame, bank_origin=FIRST_PIXEL_END):
    """Absolute position of the bank origin (the bank translation) [mm], see the
    module docstring."""
    if bank_origin == FIRST_PIXEL_END:
        return frame.origin
    if bank_origin == CENTRE_OF_MASS:
        return frame.origin + centre_of_mass_offset(frame)
    raise ValueError(f"bank_origin must be one of {BANK_ORIGINS}, not {bank_origin!r}")


def local_pixel_offsets(frame, pixels_per_straw, pixel_reference=PIXEL_CENTRE):
    """Offsets of the reference points of all pixels of a bank from its origin
    (corner A[0]), shape (tubes * 7 * N, 3), in pixel id order (tube, straw, pixel)."""
    offsets = (
        tube_offsets(frame)[:, None, None, :]
        + straw_offsets(frame)[None, :, None, :]
        + pixel_offsets_along_straw(frame, pixels_per_straw, pixel_reference)[None, None, :, :]
    )
    return offsets.reshape(-1, 3)


# ---------------------------------------------------------------------------
# Pixel ids
# ---------------------------------------------------------------------------


def pixels_in_bank(bank_id, pixels_per_straw=DEFAULT_PIXELS_PER_STRAW):
    return TUBES_PER_BANK[bank_id] * STRAWS_PER_TUBE * pixels_per_straw


def bank_first_pixel_id(
    bank_id, pixels_per_straw=DEFAULT_PIXELS_PER_STRAW, first_pixel_id=FIRST_PIXEL_ID
):
    """detector_number of the first pixel of a bank (banks are numbered consecutively)."""
    return first_pixel_id + sum(pixels_in_bank(b, pixels_per_straw) for b in range(bank_id))


def bank_pixel_ids(
    bank_id, pixels_per_straw=DEFAULT_PIXELS_PER_STRAW, first_pixel_id=FIRST_PIXEL_ID
):
    """detector_number values of a bank, in the order of local_pixel_offsets."""
    start = bank_first_pixel_id(bank_id, pixels_per_straw, first_pixel_id)
    return np.arange(start, start + pixels_in_bank(bank_id, pixels_per_straw), dtype=np.int64)


# ---------------------------------------------------------------------------
# Whole bank / instrument
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BankPixels:
    """Pixel data of one bank: the bank position (translation, the origin of the
    local offsets, see bank_origin), the detector numbers and the local offsets
    of the pixel reference points [mm] (pixel centres or pixel bases, see
    pixel_reference)."""

    bank_id: int
    frame: BankFrame
    pixels_per_straw: int
    pixel_ids: np.ndarray
    offsets: np.ndarray
    pixel_reference: str = PIXEL_CENTRE
    translation: np.ndarray = None  # bank position [mm]; None: corner A[0] (frame.origin)
    bank_origin: str = FIRST_PIXEL_END

    def __post_init__(self):
        if self.translation is None:
            object.__setattr__(self, "translation", self.frame.origin)

    @property
    def pixel_length(self):
        return self.frame.straw_length / self.pixels_per_straw

    @property
    def positions(self):
        """Absolute positions of the pixel reference points [mm]."""
        return self.translation + self.offsets

    @property
    def pixel_centres(self):
        """Absolute pixel centre positions [mm], whatever the pixel reference."""
        shift = (0.5 - pixel_reference_fraction(self.pixel_reference)) * self.pixel_length
        return self.positions + shift * self.frame.along


def compute_bank_pixels(
    bank_id,
    corners,
    pixels_per_straw=DEFAULT_PIXELS_PER_STRAW,
    first_pixel_id=FIRST_PIXEL_ID,
    pixel_reference=PIXEL_CENTRE,
    bank_origin=FIRST_PIXEL_END,
):
    """Pixel data of a bank from its corners (REFERENCE_BANKS_DATA format).
    bank_origin: the bank position, FIRST_PIXEL_END (corner A[0]) or
    CENTRE_OF_MASS (mean of the pixel centres); see the module docstring."""
    if bank_id not in range(NUMBER_OF_BANKS):
        raise ValueError(f"unknown bank id {bank_id!r}")
    if (
        not isinstance(pixels_per_straw, Integral)
        or isinstance(pixels_per_straw, bool)
        or pixels_per_straw < 1
    ):
        raise ValueError(f"pixels_per_straw must be a positive integer, not {pixels_per_straw!r}")
    pixels_per_straw = int(pixels_per_straw)
    rows = tubes_per_layer(bank_id)
    if "num_tubes" in corners and corners["num_tubes"] != TUBES_PER_BANK[bank_id]:
        raise ValueError(
            f"bank {bank_id}: num_tubes {corners['num_tubes']} != {TUBES_PER_BANK[bank_id]}"
        )
    frame = BankFrame.from_corners(corners, rows, straw_length=STRAW_LENGTHS[bank_id])
    if frame.mirrored != (bank_id in MIRRORED_BANK_IDS):
        raise ValueError(
            f"bank {bank_id}: the corners give a {'mirrored' if frame.mirrored else 'normal'} "
            f"straw pattern, expected {'mirrored' if bank_id in MIRRORED_BANK_IDS else 'normal'} "
            "(corners in the wrong order?)"
        )
    translation = bank_origin_position(frame, bank_origin)
    # the offsets from the frame origin (A[0]), moved to the chosen bank origin
    offsets = local_pixel_offsets(frame, pixels_per_straw, pixel_reference) - (
        translation - frame.origin
    )
    return BankPixels(
        bank_id,
        frame,
        pixels_per_straw,
        bank_pixel_ids(bank_id, pixels_per_straw, first_pixel_id),
        offsets,
        pixel_reference,
        translation,
        bank_origin,
    )


def compute_instrument_pixels(
    corners_by_bank,
    pixels_per_straw=DEFAULT_PIXELS_PER_STRAW,
    first_pixel_id=FIRST_PIXEL_ID,
    pixel_reference=PIXEL_CENTRE,
    bank_origin=FIRST_PIXEL_END,
):
    """Pixel data of all banks in corners_by_bank ({bank_id: corners}), as {bank_id: BankPixels}."""
    return {
        bank_id: compute_bank_pixels(
            bank_id, corners, pixels_per_straw, first_pixel_id, pixel_reference, bank_origin
        )
        for bank_id, corners in sorted(corners_by_bank.items())
    }
