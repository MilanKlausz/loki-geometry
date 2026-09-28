"""
Pixel positions of the LOKI detector banks from their 8 corner positions.

This is the geometry core used to produce the pixel data of the NeXus/JSON
instrument description (see loki_nexus_json.py). It follows the simple vector
approach of generate-nexus-files (examples/loki/LOKI_geometry.py), with the
straw layout inside the tubes taken from the Geant4 geometry
(dgcode_projects/LOKI/G4GeoLoki), which it reproduces exactly (see the tests).

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
the others.

Bank position (origin of the local offsets): the corner A[0] itself, i.e. the
FIRST PIXEL END of tube 0 (front layer, top tube, or leftmost tube in vertical
banks) on the tube axis. It is not a pixel position: the first pixel centre is
half a pixel further along the straw, and in straw 0 also 7.75 mm to the side.
This is the same convention as in the current loki.json.

Pixel reference point (what a pixel offset points to), selectable:
  PIXEL_CENTRE ("pixel_centre", default): the centre of the pixel,
      at (k + 0.5) * L / N along the straw;
  PIXEL_BASE   ("pixel_base"): the centre of the pixel's first face (the face
      towards the first pixel end), at k * L / N along the straw.
The pixel shape in the NeXus output (loki_nexus_json.py) is set accordingly.

Pixel numbering (as in Geant4 and the ICD, 1-based detector_number):
  pixel_id = bank_first_pixel_id + (tube * 7 + straw) * N + pixel_in_straw
  tube = layer * R + row (layer 0 = front layer, facing the sample).

Units: mm everywhere in this module.
"""

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Constants (TODO: move to a dedicated constants module)
# ---------------------------------------------------------------------------

NUMBER_OF_BANKS = 9
TUBES_PER_BANK = (224, 64, 48, 64, 48, 112, 128, 80, 128)  # 8 tubes per pack
LAYERS_PER_BANK = 4
STRAWS_PER_TUBE = 7
CENTRE_STRAW = 3
DEFAULT_PIXELS_PER_STRAW = 256
FIRST_PIXEL_ID = 1  # detector_number of the first pixel of bank 0 (ICD numbering)
STRAW_RADIUS = 3.75  # [mm] outer radius of a straw (pixel shape)
MIRRORED_BANK_IDS = (1, 2, 5, 6)  # packs mounted upside down (only used as a cross-check)
PIXEL_CENTRE = "pixel_centre"  # pixel offsets point to the pixel centres
PIXEL_BASE = "pixel_base"      # pixel offsets point to the centre of the pixel's first face
PIXEL_REFERENCES = (PIXEL_CENTRE, PIXEL_BASE)
_PIXEL_REFERENCE_FRACTION = {PIXEL_CENTRE: 0.5, PIXEL_BASE: 0.0}  # position within the pixel

# Straw centres in the tube, in the Geant4 tube frame (BcsTube::strawPositionsInTube) [mm]
_STRAW_STRAW_DISTANCE = 7.75
_HALF_WIDTH = 0.5 * _STRAW_STRAW_DISTANCE * np.tan(np.pi / 3)
_STRAW_POSITIONS_TUBE_FRAME = np.array([
    (-_HALF_WIDTH, -0.5 * _STRAW_STRAW_DISTANCE),
    (0.0, -_STRAW_STRAW_DISTANCE),
    (_HALF_WIDTH, -0.5 * _STRAW_STRAW_DISTANCE),
    (0.0, 0.0),                                    # straw 3 = centre straw
    (-_HALF_WIDTH, 0.5 * _STRAW_STRAW_DISTANCE),
    (0.0, _STRAW_STRAW_DISTANCE),
    (_HALF_WIDTH, 0.5 * _STRAW_STRAW_DISTANCE),
])
# Angle of the Geant4 tube frame in the (row, depth) frame of a normally mounted
# pack: 90 deg (frame convention) + tube rotation in the pack (8.45 - 30 - 13.45
# deg) + pack rotation (13.45 deg) = 68.45 deg. Checked against the Geant4
# geometry in tests/test_loki_bank_pixels.py.
_STRAW_PATTERN_ROTATION = np.radians(90.0 + (8.45 - 30.0 - 13.45) + 13.45)


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
    return TUBES_PER_BANK[bank_id] // LAYERS_PER_BANK


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
    def from_corners(cls, corners, rows):
        """Build the frame from corners in the REFERENCE_BANKS_DATA format."""
        a = np.asarray(corners["A"], dtype=float)
        b = np.asarray(corners["B"], dtype=float)
        if a.shape != (4, 3) or b.shape != (4, 3):
            raise ValueError("corners must contain 4 'A' and 4 'B' points with 3 coordinates each")
        straw_vector = b[0] - a[0]
        along = _unit(straw_vector)
        row_step = (a[2] - a[0]) / (rows - 1)
        layer_step = (a[1] - a[0]) / (LAYERS_PER_BANK - 1)
        # orthonormal (row, depth) frame perpendicular to the straws
        row_axis = _unit(row_step - np.dot(row_step, along) * along)
        depth = layer_step - np.dot(layer_step, along) * along - np.dot(layer_step, row_axis) * row_axis
        depth_axis = _unit(depth)
        # the layers are shifted towards higher tube ids only in mirrored banks
        mirrored = bool(np.dot(layer_step, row_axis) > 0)
        return cls(a[0], along, float(np.linalg.norm(straw_vector)), row_step, layer_step,
                   row_axis, depth_axis, mirrored, rows)


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
        raise ValueError(f"pixel_reference must be one of {PIXEL_REFERENCES}, not {pixel_reference!r}")
    return _PIXEL_REFERENCE_FRACTION[pixel_reference]


def pixel_offsets_along_straw(frame, pixels_per_straw, pixel_reference=PIXEL_CENTRE):
    """Offsets of the pixel reference points from the first pixel end of a
    straw, shape (N, 3): the pixel centres ((k + 0.5) * L / N) or the pixel
    bases (k * L / N), see the module docstring."""
    pixel_length = frame.straw_length / pixels_per_straw
    positions = (np.arange(pixels_per_straw) + pixel_reference_fraction(pixel_reference)) * pixel_length
    return positions[:, None] * frame.along


def local_pixel_offsets(frame, pixels_per_straw, pixel_reference=PIXEL_CENTRE):
    """Offsets of the reference points of all pixels of a bank from its origin
    (corner A[0]), shape (tubes * 7 * N, 3), in pixel id order (tube, straw, pixel)."""
    offsets = (tube_offsets(frame)[:, None, None, :]
               + straw_offsets(frame)[None, :, None, :]
               + pixel_offsets_along_straw(frame, pixels_per_straw, pixel_reference)[None, None, :, :])
    return offsets.reshape(-1, 3)


# ---------------------------------------------------------------------------
# Pixel ids
# ---------------------------------------------------------------------------

def pixels_in_bank(bank_id, pixels_per_straw=DEFAULT_PIXELS_PER_STRAW):
    return TUBES_PER_BANK[bank_id] * STRAWS_PER_TUBE * pixels_per_straw


def bank_first_pixel_id(bank_id, pixels_per_straw=DEFAULT_PIXELS_PER_STRAW, first_pixel_id=FIRST_PIXEL_ID):
    """detector_number of the first pixel of a bank (banks are numbered consecutively)."""
    return first_pixel_id + sum(pixels_in_bank(b, pixels_per_straw) for b in range(bank_id))


def bank_pixel_ids(bank_id, pixels_per_straw=DEFAULT_PIXELS_PER_STRAW, first_pixel_id=FIRST_PIXEL_ID):
    """detector_number values of a bank, in the order of local_pixel_offsets."""
    start = bank_first_pixel_id(bank_id, pixels_per_straw, first_pixel_id)
    return np.arange(start, start + pixels_in_bank(bank_id, pixels_per_straw), dtype=np.int64)


# ---------------------------------------------------------------------------
# Whole bank / instrument
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BankPixels:
    """Pixel data of one bank: the bank position (the origin of the local
    offsets = corner A[0], the first pixel end of tube 0 on its axis), the
    detector numbers and the local offsets of the pixel reference points [mm]
    (pixel centres or pixel bases, see pixel_reference)."""
    bank_id: int
    frame: BankFrame
    pixels_per_straw: int
    pixel_ids: np.ndarray
    offsets: np.ndarray
    pixel_reference: str = PIXEL_CENTRE

    @property
    def translation(self):
        """Bank position = corner A[0] [mm]."""
        return self.frame.origin

    @property
    def pixel_length(self):
        return self.frame.straw_length / self.pixels_per_straw

    @property
    def positions(self):
        """Absolute positions of the pixel reference points [mm]."""
        return self.frame.origin + self.offsets

    @property
    def pixel_centres(self):
        """Absolute pixel centre positions [mm], whatever the pixel reference."""
        shift = (0.5 - pixel_reference_fraction(self.pixel_reference)) * self.pixel_length
        return self.positions + shift * self.frame.along


def compute_bank_pixels(bank_id, corners, pixels_per_straw=DEFAULT_PIXELS_PER_STRAW,
                        first_pixel_id=FIRST_PIXEL_ID, pixel_reference=PIXEL_CENTRE):
    """Pixel data of a bank from its corners (REFERENCE_BANKS_DATA format)."""
    rows = tubes_per_layer(bank_id)
    if "num_tubes" in corners and corners["num_tubes"] != TUBES_PER_BANK[bank_id]:
        raise ValueError(f"bank {bank_id}: num_tubes {corners['num_tubes']} != {TUBES_PER_BANK[bank_id]}")
    frame = BankFrame.from_corners(corners, rows)
    return BankPixels(bank_id, frame, pixels_per_straw,
                      bank_pixel_ids(bank_id, pixels_per_straw, first_pixel_id),
                      local_pixel_offsets(frame, pixels_per_straw, pixel_reference),
                      pixel_reference)


def compute_instrument_pixels(corners_by_bank, pixels_per_straw=DEFAULT_PIXELS_PER_STRAW,
                              first_pixel_id=FIRST_PIXEL_ID, pixel_reference=PIXEL_CENTRE):
    """Pixel data of all banks in corners_by_bank ({bank_id: corners}), as {bank_id: BankPixels}."""
    return {bank_id: compute_bank_pixels(bank_id, corners, pixels_per_straw, first_pixel_id, pixel_reference)
            for bank_id, corners in sorted(corners_by_bank.items())}
