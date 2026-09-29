"""
Geometry constants of the LOKI detector banks: the single source of the
numbers used by all modules of the package.

Units: mm and degrees. Coordinates: x horizontal (positive to the left,
looking along the beam), y up, z along the beam, origin at the sample.

Sources: the LOKI Geant4 geometry (G4GeoLoki, https://github.com/ess-dg/dgcode_projects/tree/main/LOKI/G4GeoLoki:
BcsBanks.cc, BcsPack.cc, BcsTube.cc) and drawing SI-7615-463. The bank
distances and angles there are rounded CAD values; the exact CAD corner
positions are in reference_data.REFERENCE_BANKS_DATA.

Per-bank tuples are indexed by bank id:
  0 rear, 1 mid top, 2 mid left, 3 mid bottom, 4 mid right,
  5 front top, 6 front left, 7 front bottom, 8 front right.
"""

import math

# --- Banks -----------------------------------------------------------------

NUMBER_OF_BANKS = 9
BANK_IDS = tuple(range(NUMBER_OF_BANKS))
BANK_NAMES = (
    "rear",
    "mid top",
    "mid left",
    "mid bottom",
    "mid right",
    "front top",
    "front left",
    "front bottom",
    "front right",
)

# Banks with vertical tubes (the others have horizontal tubes).
VERTICAL_BANK_IDS = (2, 4, 6, 8)
# Banks whose packs are mounted upside down. They are mirror images of the
# other banks (not rotated copies), which mirrors the straw order in the tubes
# and reverses the tube numbering relative to the Geant4 pack numbering.
UPSIDE_DOWN_BANK_IDS = (1, 2, 5, 6)

# --- Packs, tubes, straws, pixels --------------------------------------------

PACKS_PER_BANK = (28, 8, 6, 8, 6, 14, 16, 10, 16)
TUBES_PER_PACK = 8  # 2 rows x 4 layers
LAYERS_PER_BANK = 4
TUBES_PER_BANK = tuple(packs * TUBES_PER_PACK for packs in PACKS_PER_BANK)
TUBES_PER_LAYER = tuple(tubes // LAYERS_PER_BANK for tubes in TUBES_PER_BANK)

STRAWS_PER_TUBE = 7
CENTRE_STRAW = 3  # the straw on the tube axis
DEFAULT_PIXELS_PER_STRAW = 256

# Effective straw length (active length, without the end electronics) [mm].
STRAW_LENGTHS = (1000.0, 1000.0, 500.0, 1000.0, 500.0, 1200.0, 1200.0, 1200.0, 1200.0)

TUBE_OUTER_RADIUS = 12.7  # [mm] 1 inch diameter tubes
STRAW_RADIUS = 3.75  # [mm] outer radius of a straw
STRAW_STRAW_DISTANCE = 7.75  # [mm] distance of the outer straw axes from the centre straw axis

# Straw axes in the tube, in the Geant4 tube frame (BcsTube::strawPositionsInTube) [mm].
_HALF_WIDTH = 0.5 * STRAW_STRAW_DISTANCE * math.tan(math.pi / 3)
STRAW_POSITIONS_IN_TUBE = (
    (-_HALF_WIDTH, -0.5 * STRAW_STRAW_DISTANCE),
    (0.0, -STRAW_STRAW_DISTANCE),
    (_HALF_WIDTH, -0.5 * STRAW_STRAW_DISTANCE),
    (0.0, 0.0),  # straw 3 = centre straw
    (-_HALF_WIDTH, 0.5 * STRAW_STRAW_DISTANCE),
    (0.0, STRAW_STRAW_DISTANCE),
    (_HALF_WIDTH, 0.5 * STRAW_STRAW_DISTANCE),
)
# Rotation of the straw hexagon in the bank (after the pack rotation) [deg],
# and the rotation of the Geant4 table above relative to that reference (BcsPack.cc).
STRAW_HEXAGON_ROTATION = 8.45
STRAW_TABLE_ROTATION_OFFSET = -30.0

# --- Tube grid in a bank -------------------------------------------------------
# The tube centres form a parallelogram grid: the tubes of a layer are
# TUBE_GRID_SIDE apart (the "row pitch"), the layers are TUBE_GRID_BASE apart,
# with the layer direction tilted by TUBE_GRID_ANGLE from the bank normal
# (the packs are rotated by this angle).

TUBE_GRID_SIDE = 28.40  # [mm] tube pitch within a layer
TUBE_GRID_BASE = 27.00  # [mm] layer pitch
TUBE_GRID_ANGLE = 13.45  # [deg] (90 - 76.55)

# Shift from the front layer to the back (4th) layer [mm]: along the bank
# normal, and within the tube plane (towards lower tube ids for normally
# mounted packs, towards higher ones for upside down packs).
_FRONT_TO_BACK = (LAYERS_PER_BANK - 1) * TUBE_GRID_BASE
FRONT_TO_BACK_NORMAL = _FRONT_TO_BACK * math.cos(math.radians(TUBE_GRID_ANGLE))  # 78.78
FRONT_TO_BACK_IN_PLANE = _FRONT_TO_BACK * math.sin(math.radians(TUBE_GRID_ANGLE))  # 18.84

# --- Nominal bank placement (drawing SI-7615-463, Geant4 BcsBanks.cc) -----------
# BANK_DISTANCES: distance from the sample to the centre of the bank's front
# face (the centre of the front layer tube centres, moved one tube radius
# towards the sample along the bank normal), measured in the section plane
# perpendicular to the tubes (y-z for horizontal, x-z for vertical banks)
# along the bank position angle. The BANK_OFFSETS (along the tubes) are
# added on top, so the 3D distance is sqrt(distance^2 + |offset|^2).

DEFAULT_REAR_BANK_DISTANCE = 5000.0  # [mm] the rear bank moves on the detector carriage
BANK_DISTANCES = (
    DEFAULT_REAR_BANK_DISTANCE,
    2950.0,
    3350.0,
    2950.0,
    3350.0,
    1364.68,
    1750.0,
    1340.0,
    1750.0,
)
# Position angle of the front face centre seen from the sample [deg], signed:
# positive = up (horizontal banks) or left (vertical banks).
BANK_POSITION_ANGLES = (0.0, 9.7, 6.5, -9.7, -6.5, 32.20, 25.6, -27.5, -25.6)
# Angle between the bank front face and the line from the sample to its
# centre [deg] (Geant4 bankTiltAngle; 90 = perpendicular).
BANK_FACE_ANGLES = (90.0, 94.0, 92.0, 94.0, 92.0, 104.7, 105.0, 100.0, 105.0)
# Offset of the bank along its tubes [mm] (x, y, z) (panel offsets in the drawing).
BANK_OFFSETS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (100.0, 0.0, 0.0),
    (0.0, -15.0, 0.0),
    (-100.0, 0.0, 0.0),
    (0.0, 50.0, 0.0),
)


def bank_tilt_angle(bank_id):
    """Tilt of the bank plane from the plane perpendicular to the beam [deg]
    (unsigned; 0 for the rear bank): 90 - (face angle - |position angle|).
    This is the angle between the tube plane and the x (vertical banks) or
    y (horizontal banks) axis."""
    return 90.0 - (BANK_FACE_ANGLES[bank_id] - abs(BANK_POSITION_ANGLES[bank_id]))


def bank_position_side(bank_id):
    """+1 for banks above / left of the beam (and the rear bank), -1 below / right."""
    return -1 if BANK_POSITION_ANGLES[bank_id] < 0 else 1


BANK_TILT_ANGLES = tuple(bank_tilt_angle(bank_id) for bank_id in BANK_IDS)  # 0, 5.7, 4.5, ...
