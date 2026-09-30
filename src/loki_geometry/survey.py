"""
Convert a survey of the LOKI detector tube ends into the calibration-data
format used by calibration.py (the format of geant4_example_data.py).

Input
-----
A survey file known to survey_files.py (default: the SAM-606 survey,
data/ESS.NSS.D03.LOKI.OB.DETECTOR_PIPE_CENTRES.TANK_WINDOW_CS.txt).
  One line per measured point: "<label>, X, Y, Z" [mm], in the coordinate
  frame of the survey. The label is the surveyors' tube label followed by the
  end, I or II. Each point is on the tube axis at the PHYSICAL end of the tube,
  i.e. it includes the (inseparable) end electronics, not only the effective
  straw length used in the models.
  What is specific to a file (the tube label map, corrections of the file and
  its coordinate frame) is in its survey_files.py entry.

Chain (get_measured_calibration_data)
-------------------------------------
read the points -> apply the fixes of the file -> convert to the NeXus frame
-> group the points into tubes (label map), assign the first / last pixel ends
-> sanity checks (verbose) -> trim the end electronics -> calibration data.

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
  sample position (the frame conversion of the file's survey_files.py entry).
  The local pixel ids of a straw increase from +x to -x in horizontal banks and
  from -y to +y in vertical banks (NeXus coordinate system), so first_pixel_end
  is the +x end (horizontal banks) or the -y end (vertical banks).

Only the banks in the label map are in the output (e.g. SAM-606 has no bank 0).
"""

import argparse
import os
import pprint
import sys

import numpy as np

from . import constants as c
from .survey_files import DEFAULT_SURVEY_FILE, SURVEY_FILES, survey_file_info

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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
# Reading the file (the fixes and the frame conversion are in survey_files.py)
# ---------------------------------------------------------------------------


def read_measured_points(filename=DEFAULT_SURVEY_FILE):
    """Read the survey txt file into {label: np.array([X, Y, Z])} (survey frame).

    The file has no header and no standard format, just
    "<label>, X, Y, Z" lines (possibly with empty lines; a UTF-8 byte order
    mark, e.g. from a spreadsheet export, is ignored)."""
    points = {}
    with open(filename, encoding="utf-8-sig") as f:
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


# ---------------------------------------------------------------------------
# Grouping the points into tubes, choosing the first / last pixel ends
# ---------------------------------------------------------------------------


def group_points_into_tubes(points, label_map, vertical_bank_ids, verbose=True):
    """Group the I/II end points by tube and bank, using the label map of the
    survey file ({tube label: (bank_id, tube id)}). Every point must belong to a
    tube of the label map and every tube must have both ends (ValueError).

    Returns {bank_id: [tube, tube]}, each tube being a dictionary with
    "label", "tube_id", "end_A" and "end_B".
    "end_A" is the first pixel end (lowest local pixel id), "end_B" the last
    pixel end (same A/B naming as in REFERENCE_BANKS_DATA). From the nominal
    geometry (REFERENCE_BANKS_DATA and the Geant4 model):
      - horizontal banks: A end is on the left (+x),
      - vertical banks:   A end is at the bottom (-y).
    The ends are assigned based on the coordinates, not on the I/II labels, but
    a note is printed (verbose) if the survey's I label is not the A end.

    The label map itself is checked (ValueError): known bank ids, tube ids of
    the front layer (0 .. tubes per layer - 1), no tube mapped twice."""
    seen = {}
    for label, (bank_id, tube_id) in label_map.items():
        if bank_id not in c.BANK_IDS:
            raise ValueError(f"label map: {label}: unknown bank id {bank_id}")
        if not 0 <= tube_id < c.TUBES_PER_LAYER[bank_id]:
            raise ValueError(
                f"label map: {label}: tube id {tube_id} is not a front layer tube of bank "
                f"{bank_id} (0 .. {c.TUBES_PER_LAYER[bank_id] - 1})"
            )
        if (bank_id, tube_id) in seen:
            raise ValueError(
                f"label map: {seen[bank_id, tube_id]} and {label} are both tube {tube_id} of "
                f"bank {bank_id}"
            )
        seen[bank_id, tube_id] = label
    expected = {label + end for label in label_map for end in ("I", "II")}
    missing, unmapped = sorted(expected - set(points)), sorted(set(points) - expected)
    if missing or unmapped:
        raise ValueError(
            f"survey points do not match the label map: missing {missing}, not in the map {unmapped}"
        )
    banks = {}
    for label, (bank_id, tube_id) in label_map.items():
        p_I, p_II = points[label + "I"], points[label + "II"]
        # coordinate along which the A and B ends are separated
        axis = 1 if bank_id in vertical_bank_ids else 0
        # A end: largest x for horizontal banks, smallest y for vertical banks
        I_is_A = (p_I[axis] > p_II[axis]) if axis == 0 else (p_I[axis] < p_II[axis])
        end_A, end_B = (p_I, p_II) if I_is_A else (p_II, p_I)
        if not I_is_A and verbose:
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
# Sanity checks of the measured (still untrimmed) tubes
# ---------------------------------------------------------------------------


def _unit(v):
    return v / np.linalg.norm(v)


def _report(title, lines, verbose):
    """Print the results of a check: all lines (verbose), or only the failing
    ones (quiet). lines: [(text, ok)]. Returns the number of failing lines."""
    failing = [text for text, ok in lines if not ok]
    if verbose:
        print(f"\n{title}:")
        for text, _ in lines:
            print(text)
    elif failing:
        print(f"{title}:")
        for text in failing:
            print(text)
    return len(failing)


def check_tube_parallelism(banks, bank_notes=None, verbose=True):
    """The two measured tubes of a bank should be parallel (the packs are
    mounted in a common frame). bank_notes ({bank_id: note}, from the survey
    file entry) are printed with the banks, e.g. if a fix of the file made a
    tube parallel by construction. Returns the number of warnings."""
    bank_notes = bank_notes or {}
    lines = []
    for bank_id, (t1, t2) in sorted(banks.items()):
        u1 = _unit(t1["end_B"] - t1["end_A"])
        u2 = _unit(t2["end_B"] - t2["end_A"])
        angle = np.degrees(np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0)))
        note = f" ({bank_notes[bank_id]})" if bank_id in bank_notes else ""
        ok = angle <= PARALLEL_TOLERANCE_DEG
        flag = "" if ok else "  WARNING: not parallel!"
        lines.append((f"  bank {bank_id}: {angle:.3f} deg{note}{flag}", ok))
    return _report("Parallelism of the two measured tubes in each bank", lines, verbose)


def check_tube_spacing(banks, tube_tube_distance, verbose=True):
    """The perpendicular distance between the two measured tubes divided by the
    tube pitch (28.4 mm) must be (nearly) an integer, equal to the difference
    of the assigned tube ids. This checks the relative tube identification.
    Returns the number of warnings."""
    lines = []
    for bank_id, (t1, t2) in sorted(banks.items()):
        u = _unit(t1["end_B"] - t1["end_A"])
        mid1 = (t1["end_A"] + t1["end_B"]) / 2
        mid2 = (t2["end_A"] + t2["end_B"]) / 2
        d = mid2 - mid1
        d_perp = d - np.dot(d, u) * u  # remove the component along the tubes
        steps = np.linalg.norm(d_perp) / tube_tube_distance
        expected = abs(t2["tube_id"] - t1["tube_id"])
        ok = abs(steps - expected) < TUBE_ID_TOLERANCE
        flag = "" if ok else "  WARNING: mismatch!"
        lines.append(
            (
                f"  bank {bank_id}: {steps:.2f} pitches (assigned tube ids "
                f"{t1['tube_id']} and {t2['tube_id']} -> {expected}){flag}",
                ok,
            )
        )
    return _report(
        "Perpendicular distance of the two tubes in units of the tube pitch", lines, verbose
    )


def check_tube_ids_against_nominal(banks, vertical_bank_ids, tube_tube_distance, verbose=True):
    """Absolute check of the tube identification: compare the transverse
    position of each measured tube with the nominal position of the front-layer
    tubes (REFERENCE_BANKS_DATA). Only the coordinate across the tubes within
    the bank plane (y for horizontal, x for vertical banks) is used, which is
    insensitive to the bank shifts along the beam (z) seen in the survey.
    The estimated id depends only on the position, so it also tells the right
    id of a wrongly assigned tube. Returns the number of warnings."""
    from .reference_data import REFERENCE_BANKS_DATA

    lines = []
    for bank_id, tubes in sorted(banks.items()):
        ref = REFERENCE_BANKS_DATA[bank_id]
        top = np.array(ref["A"][0], dtype=float)  # front layer, tube 0
        bottom = np.array(ref["A"][2], dtype=float)  # front layer, last tube
        step = _unit(bottom - top) * tube_tube_distance  # one tube pitch
        axis = 0 if bank_id in vertical_bank_ids else 1  # transverse coordinate
        for tube in tubes:
            mid = (tube["end_A"] + tube["end_B"]) / 2
            estimated = (mid[axis] - top[axis]) / step[axis]
            ok = abs(estimated - tube["tube_id"]) < TUBE_ID_TOLERANCE
            flag = "" if ok else "  WARNING: mismatch!"
            lines.append(
                (
                    f"  bank {bank_id} {tube['label']:7s}: estimated {estimated:6.2f}, "
                    f"assigned {tube['tube_id']}{flag}",
                    ok,
                )
            )
    return _report("Tube id estimated from the nominal geometry (front layer)", lines, verbose)


# ---------------------------------------------------------------------------
# Removing the extra length of the end electronics
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


def _checked_frame(frame, label, point):
    """A point converted by the frame function of an entry, as a float array (3,)."""
    converted = np.asarray(frame(point), dtype=float)
    if converted.shape != (3,) or not np.all(np.isfinite(converted)):
        raise ValueError(f"frame conversion of {label}: expected 3 finite numbers, got {converted}")
    return converted


def get_measured_calibration_data(filename=DEFAULT_SURVEY_FILE, verbose=True):
    """Full chain: read -> fixes of the file -> convert frame -> group into
    tubes -> sanity checks -> trim the tube ends -> calibration data list.
    The file must be known to survey_files.py (ValueError otherwise); a bare
    file name (no directory) that does not exist in the current directory is
    taken from the package data directory. The frame conversion is the one of the file's entry; an entry
    without one uses the coordinates as given (NeXus frame, origin at the
    sample).

    The sanity checks always run: with verbose all results are printed, without
    it only the failing ones (WARNING lines)."""
    name = os.fspath(filename)
    if os.path.basename(name) == name and not os.path.exists(name) and name in SURVEY_FILES:
        filename = SURVEY_FILES[name].path  # a packaged file, given by its name only
    survey_file = survey_file_info(filename)
    constants = _calibration_constants()
    tube_length_by_bank = constants["tube_length_by_bank"]
    vertical_bank_ids = constants["vertical_bank_ids"]
    tube_tube_distance = constants["tube_tube_distance"]

    if survey_file.frame is None and verbose:
        print(
            "NOTE: no frame conversion for this file: its X, Y, Z are used as NeXus "
            "coordinates relative to the sample position"
        )
    if survey_file.checksum is None and verbose:
        print(f"NOTE: the entry of {survey_file.file_name} has no checksum: content not checked")

    # (the numbers are the steps of the README)
    # 1) read the raw survey points (frame of the file)
    raw_points = read_measured_points(filename)

    # 2) the corrections of this file, as early as possible (in its own frame)
    for fix in survey_file.fixes:
        try:
            raw_points = fix(raw_points, verbose)
        except KeyError as error:
            raise ValueError(
                f"fix {getattr(fix, '__name__', fix)} of {survey_file.file_name}: "
                f"point {error} not in the file"
            ) from error

    # 3) convert every point to the NeXus/Geant4 frame with sample origin
    if survey_file.frame is None:
        points = dict(raw_points)
    else:
        points = {
            label: _checked_frame(survey_file.frame, label, p) for label, p in raw_points.items()
        }

    # 4) group the end points into tubes, assign the A/B ends
    banks = group_points_into_tubes(points, survey_file.label_map, vertical_bank_ids, verbose)

    #    the tubes must have a plausible length (at least half the straw length)
    for bank_id, tubes in sorted(banks.items()):
        for tube in tubes:
            length = np.linalg.norm(tube["end_B"] - tube["end_A"])
            if not length >= tube_length_by_bank[bank_id] / 2:
                raise ValueError(
                    f"{survey_file.file_name}: tube {tube['label']} (bank {bank_id}) is "
                    f"{length:.1f} mm long, the straws are {tube_length_by_bank[bank_id]} mm"
                )

    # 5) sanity checks of the geometry and of the tube identification
    if verbose:
        print(f"Survey file: {survey_file.file_name}")
        if survey_file.description:
            print(f"  {survey_file.description}")
    check_tube_parallelism(banks, survey_file.bank_notes, verbose)
    check_tube_spacing(banks, tube_tube_distance, verbose)
    check_tube_ids_against_nominal(banks, vertical_bank_ids, tube_tube_distance, verbose)

    # 6) trim the tube ends and build the output in the calibration format
    calibration_data = []
    length_lines = []
    for bank_id, tubes in sorted(banks.items()):
        effective_length = tube_length_by_bank[bank_id]
        positions = []
        for tube in tubes:
            eff_A, eff_B, measured_length, excess = find_effective_tube_ends(
                tube["end_A"], tube["end_B"], effective_length
            )
            ok = EXCESS_LENGTH_RANGE[0] <= excess <= EXCESS_LENGTH_RANGE[1]
            flag = "" if ok else "  WARNING: unexpected extra length!"
            length_lines.append(
                (
                    f"  bank {bank_id} {tube['label']:7s}: measured {measured_length:7.1f} mm, "
                    f"effective {effective_length:6.1f} mm, removed {excess:5.1f} mm{flag}",
                    ok,
                )
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
    _report("Measured tube lengths and removed extra length", length_lines, verbose)

    return calibration_data


def main(argv=None):
    """Command line entry point (loki-survey): print the checks and the calibration data."""
    parser = argparse.ArgumentParser(
        description="Convert a survey of the LOKI tube ends into calibration data "
        "(prints the sanity checks and the data)."
    )
    parser.add_argument(
        "--survey-file",
        default=DEFAULT_SURVEY_FILE,
        help="survey file (a path, or the name of a packaged file); it must have an entry in "
        f"survey_files.py (default: {os.path.basename(DEFAULT_SURVEY_FILE)})",
    )
    args = parser.parse_args(argv)
    try:
        data = get_measured_calibration_data(args.survey_file)
    except OSError as error:
        parser.error(f"cannot read {args.survey_file}: {error.strerror or error}")
    except ValueError as error:
        parser.error(str(error))
    print("\nCalibration data:")
    pprint.pprint(data, sort_dicts=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
