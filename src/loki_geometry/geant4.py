"""
Python port of the nominal pixel centre calculation of the LOKI Geant4 geometry
package G4GeoLoki (https://github.com/ess-dg/dgcode_projects/tree/main/LOKI/G4GeoLoki),
so that the pixel positions of the nominal Geant4 geometry (nominal-geant4-geometry)
can be computed in Python alone.

It ports the original implementation (dgcode_projects commit a653830):
AimHelper::getPixelCentreCoordinates and the BcsBanks / BcsPack / BcsTube /
PixelatedBanks helpers it uses, with the bank rotations as Euler angles and the
bank positions. G4GeoLoki later replaced these by one rigid transform per bank
(BcsBanks::getBankTransform, computed from the same tables, also used for the
bank calibrations); it gives the same pixel centres (within 1e-9 mm). The tube
numbering is the one of G4GeoLoki (BcsBanks::getTubeIdInBank). Units: mm,
radians internally. Coordinates: x horizontal (positive
left, beam's view), y up, z along the beam, origin at the sample.

Validated against the output of
`sb_loki_info --nominal_source_sample_distance_meters 0 -p <pixel>` (the pixel
info command of G4GeoLoki's LOKI project; without the option it adds the 23.6 m
source-sample distance to z), see _SB_LOKI_INFO_REFERENCE and validate().

Usage:
    from loki_geometry.geant4 import pixel_centre, pixel_id
    x, y, z = pixel_centre(pixel_id(bank_id=1, tube_id=13, straw_id=3, pixel_in_straw=0))
"""

import argparse
import sys
from numbers import Integral

import numpy as np

from . import constants as c

DEG = np.pi / 180.0
REAR_BANK_DISTANCE = c.DEFAULT_REAR_BANK_DISTANCE  # [mm] default of sb_loki_info
STRAW_PIXEL_NUMBER = c.DEFAULT_PIXELS_PER_STRAW

# Shared geometry constants (see constants.py), under the names used below
STRAW_LENGTH = c.STRAW_LENGTHS
PACKS = c.PACKS_PER_BANK
INVERSELY_NUMBERED_BANKS = c.UPSIDE_DOWN_BANK_IDS  # tubes numbered inversely (packs upside down)
TUBE_OUTER_RADIUS = c.TUBE_OUTER_RADIUS

# ---------------- BcsTube ----------------
# (x, y) in the tube; straw 3 is the centre straw
_STRAW_POSITIONS_IN_TUBE = c.STRAW_POSITIONS_IN_TUBE

# ---------------- BcsPack ----------------
_GRID_BASE = c.TUBE_GRID_BASE
_GRID_SIDE = c.TUBE_GRID_SIDE
_GRID_ANGLE = c.TUBE_GRID_ANGLE * DEG
_TUBE_ROTATION_ANGLE = (c.STRAW_HEXAGON_ROTATION + c.STRAW_TABLE_ROTATION_OFFSET) * DEG
# Geant4 model internals (only used here)
_PACK_BOX_WIDTH = 265.7
_PACK_BOX_HEIGHT = 55.20
_TUBE_CENTRE_DISTANCE_FROM_PACK_FRONT = 36.50


def _pack_tube_rotation():
    return _TUBE_ROTATION_ANGLE - _GRID_ANGLE


def _vertical_tube_distance_in_pack():
    return _GRID_SIDE * np.cos(_GRID_ANGLE)


def _horizontal_tube_offset(in_pack_tube_id):
    in_row = in_pack_tube_id % 4
    return (
        -0.5 * _PACK_BOX_WIDTH
        + _TUBE_CENTRE_DISTANCE_FROM_PACK_FRONT
        + in_row * _GRID_BASE
        + (_GRID_SIDE * np.sin(_GRID_ANGLE) if in_pack_tube_id < 4 else 0.0)
    )


def _vertical_tube_offset(in_pack_tube_id):
    return 0.5 * _vertical_tube_distance_in_pack() * (1 if in_pack_tube_id < 4 else -1)


# ---------------- BcsBanks ----------------
_POSITION_ANGLE = [abs(angle) for angle in c.BANK_POSITION_ANGLES]  # Geant4: unsigned + direction
_POS_DIR = [c.bank_position_side(b) for b in c.BANK_IDS]
_TILT_ANGLE = c.BANK_FACE_ANGLES
_BANK_DISTANCE = c.BANK_DISTANCES  # (bank 0 uses the rear bank distance argument instead)
_POSITION_OFFSET = c.BANK_OFFSETS
# Geant4 model internals (only used here)
_PACK_HOLDER_FROM_PACK_TOP = 7.6
_PACK_HOLDER_FROM_PACK_FRONT = 7.5
_BANK_SIZE = [
    [1265.0, 1806.0, 285.00 + 15],
    [1265.0, 590.18, 297.82],
    [765.0, 447.45, 298.92],
    [1265.0, 590.18, 297.82],
    [765.0, 447.45, 298.92],
    [1465.0, 945.00, 303.09],
    [1465.0, 1016.11, 302.99],
    [1465.0, 681.43, 302.99],
    [1465.0, 1016.11, 302.99],
]
_TOPMOST_PACK_HOLDER = [
    [117.26, 35.65],
    [77.12, 31.33],
    [77.38, 31.33],
    [77.12, 31.33],
    [77.38, 31.33],
    [77.94, 31.33],
    [77.92, 31.33],
    [77.92, 31.33],
    [77.92, 31.33],
]


def _bank_rotation_z(b):
    return c.bank_tilt_angle(b) * DEG  # 90 - (tiltAngle - positionAngle)


_BANK_ROTATION = [
    [0.0, 0.5 * np.pi],
    [np.pi, 0.5 * np.pi],
    [1.5 * np.pi, 0.5 * np.pi],
    [0.0, 0.5 * np.pi],
    [0.5 * np.pi, 0.5 * np.pi],
    [np.pi, 0.5 * np.pi],
    [1.5 * np.pi, 0.5 * np.pi],
    [0.0, 0.5 * np.pi],
    [0.5 * np.pi, 0.5 * np.pi],
]


def _bank_rotation(b, axis):
    return _bank_rotation_z(b) if axis == 2 else _BANK_ROTATION[b][axis]


def _bank_size(b, axis):
    return _BANK_SIZE[b][axis] + 0.05


def _pack_holder_to_pack_centre(axis):
    return (
        0.5 * _PACK_BOX_HEIGHT - _PACK_HOLDER_FROM_PACK_TOP
        if axis == 1
        else 0.5 * _PACK_BOX_WIDTH - _PACK_HOLDER_FROM_PACK_FRONT
    )


def _pack_holder_to_first_tube_centre(axis):
    if axis == 1:
        return (
            0.5 * _PACK_BOX_HEIGHT
            + 0.5 * _vertical_tube_distance_in_pack()
            - _PACK_HOLDER_FROM_PACK_TOP
        )
    return _TUBE_CENTRE_DISTANCE_FROM_PACK_FRONT - _PACK_HOLDER_FROM_PACK_FRONT


def _topmost_pack_position(b, axis):
    rot = _GRID_ANGLE  # pack rotation
    if axis == 0:
        return 0.0
    if axis == 1:
        holder_y = 0.5 * _bank_size(b, 1) - _TOPMOST_PACK_HOLDER[b][0]
        return (
            holder_y
            + _pack_holder_to_pack_centre(2) * np.sin(rot)
            - _pack_holder_to_pack_centre(1) * np.cos(rot)
        )
    holder_z = -(0.5 * _bank_size(b, 2) - _TOPMOST_PACK_HOLDER[b][1])
    return (
        holder_z
        + _pack_holder_to_pack_centre(2) * np.cos(rot)
        + _pack_holder_to_pack_centre(1) * np.sin(rot)
    )


def _pack_position_in_bank(b, pack, axis):
    if axis in (0, 2):
        return _topmost_pack_position(b, axis)
    return _topmost_pack_position(b, axis) - pack * 2 * _GRID_SIDE


def _detector_front_from_bank_front(b):
    rot = _GRID_ANGLE
    return (
        _TOPMOST_PACK_HOLDER[b][1]
        + _pack_holder_to_first_tube_centre(2) * np.cos(rot)
        + _pack_holder_to_first_tube_centre(1) * np.sin(rot)
        - TUBE_OUTER_RADIUS
    )


def _detector_centre_from_bank_top(b):
    rot = _GRID_ANGLE
    second_row = abs(
        _pack_holder_to_first_tube_centre(2) * np.sin(rot)
        - _pack_holder_to_first_tube_centre(1) * np.cos(rot)
    )
    first_row = second_row - _GRID_SIDE
    size_y = (PACKS[b] * 2 - 1) * _GRID_SIDE
    return _TOPMOST_PACK_HOLDER[b][0] + first_row + 0.5 * size_y


def _detector_centre_offset_in_bank(b, axis):
    if axis == 1:
        return 0.5 * _bank_size(b, 1) - _detector_centre_from_bank_top(b)
    return 0.5 * _bank_size(b, 2) - _detector_front_from_bank_front(b)


def _bank_position(b, axis, rear_bank_distance=REAR_BANK_DISTANCE):
    if b == 0:
        return [
            0.0,
            -_detector_centre_offset_in_bank(0, 1),
            rear_bank_distance + _detector_centre_offset_in_bank(0, 2),
        ][axis]
    rz = _bank_rotation_z(b)
    pos_angle = _POSITION_ANGLE[b] * DEG
    z = (
        _BANK_DISTANCE[b] * np.cos(pos_angle)
        + _detector_centre_offset_in_bank(b, 2) * np.cos(rz)
        - _detector_centre_offset_in_bank(b, 1) * np.sin(rz)
    )
    xy = (
        _BANK_DISTANCE[b] * np.sin(pos_angle)
        + _detector_centre_offset_in_bank(b, 2) * np.sin(rz)
        + _detector_centre_offset_in_bank(b, 1) * np.cos(rz)
    )
    xy *= 1 if _POS_DIR[b] > 0 else -1
    base = [xy, 0.0, z] if b in c.VERTICAL_BANK_IDS else [0.0, xy, z]
    return base[axis] + _POSITION_OFFSET[b][axis]


# ---------------- PixelatedBanks / AimHelper ----------------
def _is_integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def _check_pixels_per_straw(straw_pixel_number):
    if not _is_integer(straw_pixel_number) or straw_pixel_number < 1:
        raise ValueError(f"pixels per straw must be a positive integer, not {straw_pixel_number!r}")


def _check_range(name, value, size):
    if not _is_integer(value):
        raise ValueError(f"{name} must be an integer, not {value!r}")
    if not 0 <= value < size:
        raise ValueError(f"{name} {value} out of range 0 .. {size - 1}")


def bank_pixel_offset(b, straw_pixel_number=STRAW_PIXEL_NUMBER):
    """Zero-based id of the first pixel of bank b (b = 9: the total number of pixels)."""
    _check_pixels_per_straw(straw_pixel_number)
    _check_range("bank id", b, len(PACKS) + 1)
    return sum(PACKS[i] * 8 * 7 * straw_pixel_number for i in range(b))


def total_pixels(straw_pixel_number=STRAW_PIXEL_NUMBER):
    """Number of pixels of all banks (1,605,632 with 256 pixels per straw)."""
    return bank_pixel_offset(len(PACKS), straw_pixel_number)


def pixel_id(bank_id, tube_id, straw_id, pixel_in_straw, straw_pixel_number=STRAW_PIXEL_NUMBER):
    """Zero-based Geant4 pixel id (NeXus/ICD detector_number = this + 1).
    tube_id: Geant4 (new) tube numbering; tubes 0 .. 2*packs-1 are the front layer rows."""
    _check_pixels_per_straw(straw_pixel_number)
    _check_range("bank id", bank_id, len(PACKS))
    _check_range("tube id", tube_id, PACKS[bank_id] * 8)
    _check_range("straw id", straw_id, 7)
    _check_range("pixel in straw", pixel_in_straw, straw_pixel_number)
    return (
        bank_pixel_offset(bank_id, straw_pixel_number)
        + (tube_id * 7 + straw_id) * straw_pixel_number
        + pixel_in_straw
    )


def _rotate(x, y, angle):
    return np.cos(angle) * x - np.sin(angle) * y, np.sin(angle) * x + np.cos(angle) * y


def pixel_centre(
    pixel, straw_pixel_number=STRAW_PIXEL_NUMBER, rear_bank_distance=REAR_BANK_DISTANCE
):
    """The nominal Geant4 pixel centre [mm] (port of the original AimHelper::getPixelCentreCoordinates)."""
    _check_pixels_per_straw(straw_pixel_number)
    _check_range("pixel id", pixel, total_pixels(straw_pixel_number))
    b = next(i for i in c.BANK_IDS if pixel < bank_pixel_offset(i + 1, straw_pixel_number))
    in_bank = pixel - bank_pixel_offset(b, straw_pixel_number)
    tube = in_bank // (straw_pixel_number * 7)
    straw = (in_bank - tube * 7 * straw_pixel_number) // straw_pixel_number
    n_packs = PACKS[b]
    inverse = b in INVERSELY_NUMBERED_BANKS
    old_id = ((tube % 2) * 4) + (tube // (n_packs * 2))
    in_pack_tube = (old_id + 4) % 8 if inverse else old_id % 8
    normal_pack = (tube % (n_packs * 2)) // 2
    pack = (n_packs - 1) - normal_pack if inverse else normal_pack

    # pixel in straw
    loc = pixel % straw_pixel_number
    pixel_length = STRAW_LENGTH[b] / straw_pixel_number
    pz = -0.5 * STRAW_LENGTH[b] + (loc + 0.5) * pixel_length
    pz = -pz if inverse else pz
    # straw in tube
    px, py = _STRAW_POSITIONS_IN_TUBE[straw]
    # tube in pack
    px, py = _rotate(px, py, _pack_tube_rotation())
    px += _horizontal_tube_offset(in_pack_tube)
    py += _vertical_tube_offset(in_pack_tube)
    # pack in bank
    px, py = _rotate(px, py, _GRID_ANGLE)
    px += _pack_position_in_bank(b, pack, 2)
    py += _pack_position_in_bank(b, pack, 1)
    pz += _pack_position_in_bank(b, pack, 0)
    # bank rotations (same order and argument order as the C++ code)
    py, px = _rotate(py, px, _bank_rotation(b, 2))
    pz, py = _rotate(pz, py, _bank_rotation(b, 0))
    pz, px = _rotate(pz, px, -_bank_rotation(b, 1))
    # bank position
    return (
        px + _bank_position(b, 0, rear_bank_distance),
        py + _bank_position(b, 1, rear_bank_distance),
        pz + _bank_position(b, 2, rear_bank_distance),
    )


# ---------------- vectorised helpers (built on pixel_centre) ----------------
def bank_pixel_centres(
    bank_id, straw_pixel_number=STRAW_PIXEL_NUMBER, rear_bank_distance=REAR_BANK_DISTANCE
):
    """Geant4 pixel centres of a whole bank [mm], shape (n_pixels, 3), in pixel id order.

    The pixels of a straw lie on a straight line at equal distances, so only the
    first and last pixel of each straw are computed with pixel_centre (slow,
    scalar) and the others are interpolated linearly (exact for this geometry)."""
    _check_pixels_per_straw(straw_pixel_number)
    _check_range("bank id", bank_id, len(PACKS))
    n_straws = PACKS[bank_id] * 8 * 7
    first_id = bank_pixel_offset(bank_id, straw_pixel_number)
    starts = np.empty((n_straws, 3))
    ends = np.empty((n_straws, 3))
    for straw in range(n_straws):
        straw_first_pixel = first_id + straw * straw_pixel_number
        starts[straw] = pixel_centre(straw_first_pixel, straw_pixel_number, rear_bank_distance)
        ends[straw] = pixel_centre(
            straw_first_pixel + straw_pixel_number - 1, straw_pixel_number, rear_bank_distance
        )
    fraction = np.linspace(0.0, 1.0, straw_pixel_number) if straw_pixel_number > 1 else np.zeros(1)
    pixels = starts[:, None, :] + (ends - starts)[:, None, :] * fraction[None, :, None]
    return pixels.reshape(-1, 3)


def bank_corners(
    bank_id, straw_pixel_number=STRAW_PIXEL_NUMBER, rear_bank_distance=REAR_BANK_DISTANCE
):
    """The 8 corner positions of a bank in the Geant4 geometry, in the
    REFERENCE_BANKS_DATA format: {'A': [4 points], 'B': [4 points], 'num_tubes'}.
    The points are the straw ENDS on the tube axis (centre straw, moved half a
    pixel outwards from the first / last pixel centre) of the corner tubes:
    [front layer top tube, back layer top tube, front layer bottom tube,
    back layer bottom tube] (top / bottom = lowest / highest tube id in the
    layer; in vertical banks tube 0 is the leftmost, beam's view);
    'A' = first pixel end, 'B' = last pixel end.
    The straw ends do not depend on the number of pixels per straw; with 1 pixel
    per straw they are computed with 2 (a single pixel centre gives no direction)."""
    _check_pixels_per_straw(straw_pixel_number)
    _check_range("bank id", bank_id, len(PACKS))
    straw_pixel_number = max(straw_pixel_number, 2)
    rows = PACKS[bank_id] * 2  # tubes per layer
    corner_tubes = [0, 3 * rows, rows - 1, 4 * rows - 1]
    centre_straw = 3
    a_points, b_points = [], []
    for tube in corner_tubes:
        first = np.array(
            pixel_centre(
                pixel_id(bank_id, tube, centre_straw, 0, straw_pixel_number),
                straw_pixel_number,
                rear_bank_distance,
            )
        )
        last = np.array(
            pixel_centre(
                pixel_id(bank_id, tube, centre_straw, straw_pixel_number - 1, straw_pixel_number),
                straw_pixel_number,
                rear_bank_distance,
            )
        )
        half_pixel = (last - first) / (straw_pixel_number - 1) / 2
        a_points.append(tuple(float(v) for v in first - half_pixel))
        b_points.append(tuple(float(v) for v in last + half_pixel))
    return {"A": a_points, "B": b_points, "num_tubes": PACKS[bank_id] * 8}


# Pixel centres printed by `sb_loki_info --nominal_source_sample_distance_meters 0 -p <pixel>`
_SB_LOKI_INFO_REFERENCE = {
    425472: (498.046875, 342.877351, 2935.976184),
    425727: (-498.046875, 342.877351, 2935.976184),
    837632: (697.656250, 582.053972, 1213.865341),
    839424: (697.656250, 554.968411, 1222.405386),
    848384: (697.656250, 419.540604, 1265.105609),
    846592: (697.656250, 446.626165, 1256.565565),
    850176: (697.656250, 392.455043, 1273.645654),
    1041920: (604.951741, -612.656250, 1619.423401),
    1042175: (604.951741, 582.656250, 1619.423401),
    1265920: (497.656250, -852.789377, 1128.116342),
    1266175: (-697.656250, -852.789377, 1128.116342),
    1267712: (497.656250, -879.874939, 1119.576297),
    1394944: (-604.951741, -547.656250, 1619.423401),
    1395199: (-604.951741, 647.656250, 1619.423401),
}


def validate(tolerance=1e-6):
    """Compare the port with the sb_loki_info reference values (6 decimals);
    prints the pixels deviating by more than tolerance [mm] and returns the
    largest deviation [mm] (inf if a position is not a finite number)."""
    worst = 0.0
    for pix, ref in _SB_LOKI_INFO_REFERENCE.items():
        dev = np.max(np.abs(np.array(pixel_centre(pix)) - np.array(ref)))
        worst = max(worst, float(dev)) if np.isfinite(dev) else np.inf
        if not dev <= tolerance:
            print(
                f"pixel {pix}: port {np.round(pixel_centre(pix), 6)} vs sb_loki_info {ref} (dev {dev:.2e} mm)"
            )
    return worst


def main(argv=None):
    """Command line entry point (loki-geant4-validate). Returns 1 if the port
    deviates from the sb_loki_info reference values by more than the tolerance."""
    parser = argparse.ArgumentParser(
        description="Validate the Python port of the Geant4 pixel centres against sb_loki_info values."
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-6, help="[mm] largest accepted deviation"
    )
    args = parser.parse_args(argv)
    worst = validate(args.tolerance)
    print(f"Largest deviation from the sb_loki_info reference values: {worst:.2e} mm")
    return 0 if worst <= args.tolerance else 1


if __name__ == "__main__":
    sys.exit(main())
