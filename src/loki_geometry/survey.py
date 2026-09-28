"""
Convert the survey (SAM-606) measurement of the LOKI detector tube ends into the
calibration-data format used by calibration.py (the format of
geant4_example_data.py).

Input
-----
data/ESS.NSS.D03.LOKI.OB.DETECTOR_PIPE_CENTRES.TANK_WINDOW_CS.txt (from the SAM-606 survey)
  One line per measured point: "<label>, X, Y, Z" [mm], in the TANK WINDOW
  coordinate system of the survey:
    X - along the beam (downstream positive)
    Y - horizontal, positive to the LEFT when looking along the beam
    Z - vertical, positive UP
  The labels look like "C2RP16I":
    C1 / C2 - front / middle detector carriage
    U/D/L/R - up / down / left / right bank of the carriage
    P<n>    - "pipe" (tube) number, the surveyors' own scheme, which does NOT
              match our tube numbering (and is wrong in two cases, see below)
    I / II  - the two ends of the measured tube
  Each point is on the tube axis at the PHYSICAL end of the tube, i.e. it
  includes the (inseparable) end electronics, not only the effective straw
  length used in the models.

Output
------
A list of dictionaries in the same format as example_calibration_data:
  [{"bank_id": b,
    "positions": [{"tube_id": <zero-indexed local tube id>,
                   "first_pixel_end": [x, y, z],  # effective straw end on the
                                                  # side of the first (lowest id) pixel
                   "last_pixel_end": [x, y, z],   # effective straw end on the
                                                  # side of the last pixel
                   "distance": <|first - last|>}, # = nominal straw length
                  {...second tube...}]},
   ...]
  Coordinates are in [mm] in the NeXus/Geant4 frame used by the other scripts
  (x horizontal positive LEFT, y up, z along the beam), with the origin at the
  sample position (see SAMPLE_TO_TANK_WINDOW_DISTANCE below).
  The local pixel ids of a straw increase from +x to -x in horizontal banks and
  from -y to +y in vertical banks (NeXus coordinate system), so first_pixel_end
  is the +x end (horizontal banks) or the -y end (vertical banks).

Bank 0 (rear bank) was NOT measured, so it is missing from the output.
"""

import os
import pprint

import numpy as np

from . import constants as c

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MEASUREMENT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "ESS.NSS.D03.LOKI.OB.DETECTOR_PIPE_CENTRES.TANK_WINDOW_CS.txt",
)

# Distance [mm] along the beam from the nominal sample position to the origin
# of the survey's TANK WINDOW coordinate system (window is downstream of the
# sample, so z_sample = X_window + this value).
# TODO: PROVISIONAL value! Drawing SI-7615-463 (sheet 2, detail E) gives
# 119.00 mm from the nominal sample point to the vessel window flange face and
# 108.0 mm to the sapphire window; which surface is the survey origin still has
# to be confirmed. 100 mm puts the front carriage at its intended distance.
SAMPLE_TO_TANK_WINDOW_DISTANCE = 100.0  # [mm]

# Mapping of the survey tube labels (without the I/II end suffix) to
# (bank_id, zero-indexed local tube id).
# Local tube id convention (the REAL numbering used in the calibration):
#   - it counts the tubes of the FRONT layer (the one facing the sample; all
#     surveyed tubes are in the front layer),
#   - horizontal banks (0, 1, 3, 5, 7): 0 is the TOPmost tube,
#   - vertical banks (2, 4, 6, 8): 0 is the LEFTmost tube (looking along the
#     beam, i.e. the one with the largest x).
# Identification was done by comparing the measured positions with the
# nominal tube positions (see check_tube_ids_against_nominal, which re-checks
# it every time the module runs).
TUBE_LABEL_MAP = {
    # --- middle carriage (C2) ---
    "C2UP16": (1, 0),  # mid top, topmost tube (1st of 16 rows)
    "C2UP3": (1, 13),  # mid top, 3rd tube from the bottom
    "C2LP16": (2, 0),  # mid left, leftmost (outermost) tube
    "C2LP1": (2, 11),  # mid left, rightmost (innermost) tube, 12th of 12
    "C2DP1": (3, 0),  # mid bottom, topmost (innermost) tube
    "C2DP16": (3, 15),  # mid bottom, bottom tube, 16th of 16
    "C2RP1": (4, 0),  # mid right, leftmost (innermost) tube
    "C2RP16": (4, 11),  # mid right, rightmost (outermost) tube, 12th of 12
    # --- front carriage (C1) ---
    # Front top: the "C1UP8" label is used for points of TWO different tubes:
    # C1UP8I is on tube "UP9" (local id 19), C1UP8II is on tube "UP8" (local
    # id 20). C1UP8II is replaced by an extrapolated end of tube UP9 early on
    # (see fix_bank5_mislabelled_tube), so after that "C1UP8" means tube UP9.
    "C1UP8": (5, 19),  # front top, tube "UP9" (9th from bottom of 28 rows)
    "C1UP3": (5, 25),  # front top, 3rd tube from the bottom
    "C1LP11": (6, 21),  # front left, 22nd tube from the left (of 32)
    "C1LP1": (6, 31),  # front left, rightmost (innermost) tube
    "C1DP1": (7, 0),  # front bottom, topmost (innermost) tube
    "C1DP31": (7, 18),  # front bottom, 19th tube of 20 (label "31" is misleading)
    "C1RP1": (8, 0),  # front right, leftmost (innermost) tube
    "C1RP11": (8, 10),  # front right, 11th tube from the left (of 32)
}

# Tolerances of the sanity checks (only warnings are printed, nothing stops)
PARALLEL_TOLERANCE_DEG = 0.5  # max angle between the 2 tubes of a bank
TUBE_ID_TOLERANCE = 0.25  # [tube pitch] allowed deviation from integer
EXCESS_LENGTH_RANGE = (20.0, 40.0)  # [mm] expected measured - effective length


def _calibration_constants():
    """The shared geometry constants used here (see constants.py)."""
    return {
        "tube_length_by_bank": c.STRAW_LENGTHS,  # [mm] effective straw length
        "vertical_bank_ids": c.VERTICAL_BANK_IDS,
        "tube_tube_distance": c.TUBE_GRID_SIDE,  # [mm] 28.40
    }


# ---------------------------------------------------------------------------
# Step 1: read the file and convert to our coordinate frame
# ---------------------------------------------------------------------------


def read_measured_points(filename=MEASUREMENT_FILE):
    """Read the survey txt file into {label: np.array([X, Y, Z])} (survey frame).

    The file has no header and no standard format, just
    "<label>, X, Y, Z" lines (possibly with empty lines)."""
    points = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = [s.strip() for s in line.split(",")]
            if len(fields) != 4:
                raise ValueError(f"Unexpected line format: '{line}'")
            label = fields[0]
            if label in points:
                raise ValueError(f"Duplicate label in the file: {label}")
            points[label] = np.array([float(v) for v in fields[1:]])
    return points


def survey_to_nexus_frame(point, sample_to_window_distance):
    """Convert a survey point (X beam, Y left, Z up; origin at the tank window)
    to the NeXus/Geant4 frame (x left, y up, z beam; origin at the sample).

    Both frames are right-handed and the axes are just cyclically permuted:
      x_nexus = Y_survey, y_nexus = Z_survey, z_nexus = X_survey
    Then the origin is moved from the window to the sample position, which
    lies upstream of the window by sample_to_window_distance."""
    X, Y, Z = point
    return np.array([Y, Z, X + sample_to_window_distance])


# ---------------------------------------------------------------------------
# Step 2: fix the bank 5 (front top) mislabelled measurement
# ---------------------------------------------------------------------------


def fix_bank5_mislabelled_tube(points, tube_tube_distance):
    """In bank 5 the two "C1UP8" points belong to two different tubes:
    C1UP8I is the start of tube UP9, C1UP8II is the end of tube UP8.

    Assuming that the tubes of a bank are parallel (tested for the other
    banks in check_tube_parallelism), the missing end of tube UP9 is
    extrapolated from its measured start (C1UP8I), using the vector between
    the two ends of the other measured tube of the bank (C1UP3I -> C1UP3II).
    The measured C1UP8II is discarded (after a sanity check that it really is
    one tube pitch away from the extrapolated UP9 end).

    This is done at the very beginning, so that everything downstream can
    treat bank 5 like any other bank. Operates in place on `points`."""
    up9_start = points["C1UP8I"]
    up3_vector = points["C1UP3II"] - points["C1UP3I"]
    up9_end = up9_start + up3_vector  # extrapolated end of tube UP9

    # Sanity check: the discarded point (end of tube UP8) should be about one
    # tube pitch (28.4 mm) away from the extrapolated end of UP9.
    discarded = points["C1UP8II"]
    separation = np.linalg.norm(discarded - up9_end)
    print(
        f"Bank 5 fix: C1UP8II replaced by the extrapolated end of tube UP9. "
        f"Distance of the discarded point (tube UP8) from it: {separation:.1f} mm "
        f"(expected ~{tube_tube_distance} mm = 1 tube pitch)"
    )
    if abs(separation - tube_tube_distance) > TUBE_ID_TOLERANCE * tube_tube_distance:
        print("  WARNING: that is not ~1 tube pitch, check the bank 5 assumption!")

    points["C1UP8II"] = up9_end
    return points


# ---------------------------------------------------------------------------
# Step 3: group the points into tubes and decide which end is which
# ---------------------------------------------------------------------------


def group_points_into_tubes(points, vertical_bank_ids):
    """Group the I/II end points by tube and bank.

    Returns {bank_id: [tube, tube]}, each tube being a dictionary with
    "label", "tube_id", "end_A" and "end_B".
    "end_A" is the first pixel end (lowest local pixel id), "end_B" the last
    pixel end (same A/B naming as in REFERENCE_BANKS_DATA / detector_banks_geo). From the nominal geometry (REFERENCE_BANKS_DATA and the Geant4 model):
      - horizontal banks: A end is on the left (+x),
      - vertical banks:   A end is at the bottom (-y).
    The ends are assigned based on the coordinates, not on the I/II labels, but
    a warning is printed if the survey's I label is not the A end (in the
    current data I is always the A end)."""
    banks = {}
    for label, (bank_id, tube_id) in TUBE_LABEL_MAP.items():
        p_I, p_II = points[label + "I"], points[label + "II"]
        # coordinate along which the A and B ends are separated
        axis = 1 if bank_id in vertical_bank_ids else 0
        # A end: largest x for horizontal banks, smallest y for vertical banks
        I_is_A = (p_I[axis] > p_II[axis]) if axis == 0 else (p_I[axis] < p_II[axis])
        end_A, end_B = (p_I, p_II) if I_is_A else (p_II, p_I)
        if not I_is_A:
            print(f"Note: for {label} the II point is the A (low pixel id) end.")
        banks.setdefault(bank_id, []).append(
            {"label": label, "tube_id": tube_id, "end_A": end_A, "end_B": end_B}
        )

    # Order the two tubes of each bank by tube id (not required, just tidy)
    for bank_id in banks:
        banks[bank_id].sort(key=lambda tube: tube["tube_id"])
        if len(banks[bank_id]) != 2:
            raise ValueError(
                f"Bank {bank_id}: expected 2 measured tubes, got {len(banks[bank_id])}"
            )
    return banks


# ---------------------------------------------------------------------------
# Step 4: sanity checks on the measured (still untrimmed) tubes
# ---------------------------------------------------------------------------


def _unit(v):
    return v / np.linalg.norm(v)


def check_tube_parallelism(banks):
    """The two measured tubes of a bank should be parallel (the packs are
    mounted in a common frame). This also justifies the bank 5 extrapolation,
    which assumes it. For bank 5 the test is trivially passed, because one of
    the tubes was constructed parallel to the other one."""
    print("\nParallelism of the two measured tubes in each bank:")
    for bank_id, (t1, t2) in sorted(banks.items()):
        u1 = _unit(t1["end_B"] - t1["end_A"])
        u2 = _unit(t2["end_B"] - t2["end_A"])
        angle = np.degrees(np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0)))
        note = " (constructed parallel, see bank 5 fix)" if bank_id == 5 else ""
        flag = "  WARNING: not parallel!" if angle > PARALLEL_TOLERANCE_DEG else ""
        print(f"  bank {bank_id}: {angle:.3f} deg{note}{flag}")


def check_tube_spacing(banks, tube_tube_distance):
    """The perpendicular distance between the two measured tubes divided by the
    tube pitch (28.4 mm) must be (nearly) an integer, equal to the difference
    of the assigned tube ids. This checks the relative tube identification."""
    print("\nPerpendicular distance of the two tubes in units of the tube pitch:")
    for bank_id, (t1, t2) in sorted(banks.items()):
        u = _unit(t1["end_B"] - t1["end_A"])
        mid1 = (t1["end_A"] + t1["end_B"]) / 2
        mid2 = (t2["end_A"] + t2["end_B"]) / 2
        d = mid2 - mid1
        d_perp = d - np.dot(d, u) * u  # remove the component along the tubes
        steps = np.linalg.norm(d_perp) / tube_tube_distance
        expected = abs(t2["tube_id"] - t1["tube_id"])
        flag = "" if abs(steps - expected) < TUBE_ID_TOLERANCE else "  WARNING: mismatch!"
        print(
            f"  bank {bank_id}: {steps:.2f} pitches (assigned tube ids "
            f"{t1['tube_id']} and {t2['tube_id']} -> {expected}){flag}"
        )


def check_tube_ids_against_nominal(banks, vertical_bank_ids, tube_tube_distance):
    """Absolute check of the tube identification: compare the transverse
    position of each measured tube with the nominal position of the front-layer
    tubes (REFERENCE_BANKS_DATA). Only the coordinate across the tubes within
    the bank plane (y for horizontal, x for vertical banks) is used, which is
    insensitive to the bank shifts along the beam (z) seen in the survey."""
    from .reference_data import REFERENCE_BANKS_DATA

    print("\nTube id estimated from the nominal geometry (front layer):")
    for bank_id, tubes in sorted(banks.items()):
        ref = REFERENCE_BANKS_DATA[bank_id]
        top = np.array(ref["A"][0], dtype=float)  # front layer, tube 0
        bottom = np.array(ref["A"][2], dtype=float)  # front layer, last tube
        step = _unit(bottom - top) * tube_tube_distance  # one tube pitch
        axis = 0 if bank_id in vertical_bank_ids else 1  # transverse coordinate
        for tube in tubes:
            mid = (tube["end_A"] + tube["end_B"]) / 2
            estimated = (mid[axis] - top[axis]) / step[axis]
            flag = (
                ""
                if abs(estimated - tube["tube_id"]) < TUBE_ID_TOLERANCE
                else "  WARNING: mismatch!"
            )
            print(
                f"  bank {bank_id} {tube['label']:7s}: estimated {estimated:6.2f}, "
                f"assigned {tube['tube_id']}{flag}"
            )


# ---------------------------------------------------------------------------
# Step 5: remove the extra length of the end electronics
# ---------------------------------------------------------------------------


def find_effective_tube_ends(end_A, end_B, effective_length):
    """Return the effective straw end positions of a tube, given two points on
    the tube axis near its two ends.

    The two points are moved symmetrically, in 3D along the line through them,
    until their distance equals effective_length. This works both ways:
      - measured tube ends (survey) are ~30 mm too far apart, because the
        physical tube includes the end electronics (~15 mm on each end), while
        the models use the effective straw length -> both ends move INWARDS,
      - end pixel centres (e.g. from the Geant4 geometry) are one pixel length
        too close (half a pixel on each end) -> both ends move OUTWARDS.
    The ends cannot be located separately, so the difference is assumed to be
    split equally between the two ends.

    Returns (effective_A, effective_B, input_length, excess), where
    excess = input_length - effective_length (positive: shortened,
    negative: stretched)."""
    vector = end_B - end_A
    input_length = np.linalg.norm(vector)
    direction = vector / input_length
    excess = input_length - effective_length
    # excess > 0 moves the points towards each other, excess < 0 away
    effective_A = end_A + direction * excess / 2
    effective_B = end_B - direction * excess / 2
    return effective_A, effective_B, input_length, excess


def adjust_calibration_data_to_effective_tube_lengths(calibration_data, verbose=True):
    """Apply find_effective_tube_ends to data that is already in the
    calibration format (e.g. example_calibration_data from the Geant4 end
    pixel centres), returning a new, corrected copy.

    "first_pixel_end" and "last_pixel_end" are assumed to be the two ends of
    the tube (e.g. first and last pixel centre, or physical tube ends); both
    are moved symmetrically along the tube to the effective straw ends.
    NOTE: calibration.py only uses the tube midpoints and directions,
    which this symmetric adjustment does not change, so it is only needed when
    the straw end positions themselves are wanted."""
    import copy

    tube_length_by_bank = _calibration_constants()["tube_length_by_bank"]
    adjusted = copy.deepcopy(calibration_data)  # don't modify the input
    for bank_data in adjusted:
        bank_id = bank_data["bank_id"]
        effective_length = tube_length_by_bank[bank_id]
        for pos in bank_data["positions"]:
            first = np.array(pos["first_pixel_end"], dtype=float)
            last = np.array(pos["last_pixel_end"], dtype=float)
            eff_first, eff_last, input_length, excess = find_effective_tube_ends(
                first, last, effective_length
            )
            if verbose:
                print(
                    f"bank {bank_id} tube {pos['tube_id']}: {input_length:.2f} mm -> "
                    f"{effective_length} mm ({'shortened' if excess > 0 else 'stretched'} "
                    f"by {abs(excess):.2f} mm)"
                )
            pos["first_pixel_end"] = [float(v) for v in eff_first]
            pos["last_pixel_end"] = [float(v) for v in eff_last]
            pos["distance"] = float(np.linalg.norm(eff_last - eff_first))
    return adjusted


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def get_measured_calibration_data(
    filename=MEASUREMENT_FILE,
    sample_to_window_distance=SAMPLE_TO_TANK_WINDOW_DISTANCE,
    verbose=True,
):
    """Full chain: read -> fix bank 5 -> convert frame -> group into tubes ->
    sanity checks -> trim the tube ends -> calibration data list."""
    constants = _calibration_constants()
    tube_length_by_bank = constants["tube_length_by_bank"]
    vertical_bank_ids = constants["vertical_bank_ids"]
    tube_tube_distance = constants["tube_tube_distance"]

    if sample_to_window_distance == 0.0:
        print(
            "WARNING: sample to tank window distance is 0 -> positions are "
            "relative to the tank window, not to the sample!"
        )
    elif verbose:
        print(
            f"NOTE: using sample to tank window distance {sample_to_window_distance} mm "
            f"(provisional, see SAMPLE_TO_TANK_WINDOW_DISTANCE)"
        )

    # 1) read the raw survey points (survey frame)
    raw_points = read_measured_points(filename)

    # 2) fix the mislabelled bank 5 tube as early as possible (the fix only
    #    uses differences of points, so the frame does not matter here)
    raw_points = fix_bank5_mislabelled_tube(raw_points, tube_tube_distance)

    # 3) convert every point to the NeXus/Geant4 frame with sample origin
    points = {
        label: survey_to_nexus_frame(p, sample_to_window_distance)
        for label, p in raw_points.items()
    }

    # 4) group the end points into tubes, assign the A/B ends
    banks = group_points_into_tubes(points, vertical_bank_ids)

    # 5) sanity checks of the geometry and of the tube identification
    if verbose:
        check_tube_parallelism(banks)
        check_tube_spacing(banks, tube_tube_distance)
        check_tube_ids_against_nominal(banks, vertical_bank_ids, tube_tube_distance)
        print("\nMeasured tube lengths and removed extra length:")

    # 6) trim the tube ends and build the output in the calibration format
    calibration_data = []
    for bank_id, tubes in sorted(banks.items()):
        effective_length = tube_length_by_bank[bank_id]
        positions = []
        for tube in tubes:
            eff_A, eff_B, measured_length, excess = find_effective_tube_ends(
                tube["end_A"], tube["end_B"], effective_length
            )
            if verbose:
                flag = (
                    ""
                    if EXCESS_LENGTH_RANGE[0] <= excess <= EXCESS_LENGTH_RANGE[1]
                    else "  WARNING: unexpected extra length!"
                )
                print(
                    f"  bank {bank_id} {tube['label']:7s}: measured {measured_length:7.1f} mm, "
                    f"effective {effective_length:6.1f} mm, removed {excess:5.1f} mm{flag}"
                )
            positions.append(
                {
                    "tube_id": tube["tube_id"],  # zero-indexed local tube id
                    "first_pixel_end": [float(v) for v in eff_A],  # low pixel id end
                    "last_pixel_end": [float(v) for v in eff_B],  # other end
                    "distance": float(np.linalg.norm(eff_B - eff_A)),
                }
            )
        calibration_data.append({"bank_id": bank_id, "positions": positions})

    return calibration_data


def main():
    """Command line entry point (loki-survey): print the checks and the calibration data."""
    data = get_measured_calibration_data()
    print("\nCalibration data:")
    pprint.pprint(data, sort_dicts=False)


if __name__ == "__main__":
    main()
