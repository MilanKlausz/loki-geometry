"""
The known survey files of the LOKI detector tubes and what is specific to each
of them: how their tube labels map to (bank, tube id), which corrections they
need, and how their coordinate frame converts to the NeXus frame.

survey.py does the generic part (read the points, group them into tubes, check
them, trim the end electronics); it looks up the entry of the file it is given
with survey_file_info(). A file that is not listed here, or whose content does
not match its entry (if the entry has a checksum), is rejected: the labels
follow the surveyors' own numbering, so every new survey file needs its own
entry. How to add one is described in the README ("A new survey").

An entry (SurveyFile), only file_name and label_map are required:
  file_name     the file name (in src/loki_geometry/data/ for packaged files)
  label_map     {tube label (without the I / II end suffix): (bank_id, tube id)}
                tube id: zero-indexed local tube id of the FRONT layer
                (horizontal banks: 0 = topmost tube, vertical banks: 0 =
                leftmost tube looking along the beam)
  frame         optional: the file's coordinate frame, as a function
                point [mm] -> NeXus point [mm] (x left, y up, z along the beam,
                origin at the sample); everything the conversion needs (e.g. the
                position of the survey origin relative to the sample) is part of
                the function.
                None (default): the X, Y, Z values of the file are used as they
                are, i.e. they must already be NeXus coordinates WITH RESPECT TO
                THE SAMPLE POSITION.
  checksum      optional: SHA-256 of the file content (see content_checksum).
                None (default): the content is not checked, so a changed file
                would be used with this entry unnoticed.
  description   optional: where the file comes from
  fixes         optional: functions (points, verbose) -> points, applied in order
                to the points (in the file's frame) right after reading them
  bank_notes    optional: {bank_id: note} printed with the checks of that bank
  calibration_name  optional, but needed to use the survey in loki-pixel-json and
                loki-info (--calibration NAME; the calibration of the default
                survey is their default): the name of the bank calibration made
                from this file; also the default name of the Geant4 calibration
                file (loki-calibrate --geant4-calibration-file)
  carriage_position  optional [mm]: the position of the detector carriage during the
                survey of the rear bank (0), from its zero position; the bank 0 points
                are moved back along z by it (default 0; an error if bank 0 is not in
                the label map)
  excess_length_range  optional [mm]: (min, max) of the expected measured - effective
                tube length, checked for every tube (default DEFAULT_EXCESS_LENGTH_RANGE,
                20-40 mm, for points at the physical tube ends)
"""

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import constants as c

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# [mm] expected measured - effective tube length for points at the physical tube ends (the end
# electronics, ~15 mm at each end)
DEFAULT_EXCESS_LENGTH_RANGE = (20.0, 40.0)


@dataclass(frozen=True)
class SurveyFile:
    """What is specific to one survey file (see the module docstring)."""

    file_name: str
    label_map: dict[str, tuple[int, int]]
    frame: Optional[Callable] = None  # None: the values are NeXus coordinates from the sample
    checksum: Optional[str] = None  # None: the content is not checked
    description: str = ""
    fixes: tuple[Callable, ...] = ()
    bank_notes: dict[int, str] = field(default_factory=dict)
    calibration_name: Optional[str] = (
        None  # the name of its calibration (loki-pixel-json, loki-info)
    )
    # [mm] the position of the detector carriage (the rear bank, 0) during the survey, from its zero
    # position (the 5 m position, where the bank is placed by the calibration and as
    # detector_carriage_zero): the points of bank 0 are moved back by it along the beam (z)
    carriage_position: float = 0.0
    # [mm] the expected measured - effective tube length (the length removed by the trimming)
    excess_length_range: tuple[float, float] = DEFAULT_EXCESS_LENGTH_RANGE

    @property
    def path(self):
        """Path of the file in the package data directory."""
        return os.path.join(DATA_DIR, self.file_name)


def content_checksum(filename):
    """SHA-256 of the non-empty lines of a file, stripped and joined with '\\n'
    (so that line endings, trailing spaces and a UTF-8 byte order mark do not
    matter)."""
    with open(filename, encoding="utf-8-sig") as file:
        lines = [line.strip() for line in file if line.strip()]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Coordinate frames
# ---------------------------------------------------------------------------


# Distance [mm] along the beam from the nominal sample position to the origin
# of the survey's TANK WINDOW coordinate system (the window is downstream of the
# sample, so z_sample = X_window + this value).
# From the poses of the TANK WINDOW frame and of the sample in the common TCS
# frame: TANK WINDOW (9425.77, 21848.983, -400.168), sample (9393.029, 21772.594,
# -400.181) [mm], both with Euler XYZ (0, 0, 66.8) deg. Same orientation, so the
# two frames differ by a translation only: the window origin is 83.110 mm from the
# sample along the beam (X), with -0.001 mm along Y and +0.013 mm along Z
# (neglected). The window pose reproduces the SAM-606 points of the TCS file from
# the TANK WINDOW file within their 0.1 mm rounding.
# It shifts every bank equally along z: it does not change the fit quality
# numbers or the tilts, only the distances compared with the intended geometry.
SAMPLE_TO_TANK_WINDOW_DISTANCE = 83.11  # [mm]


def tank_window_frame(point):
    """TANK WINDOW coordinate system of the survey (X along the beam, Y left,
    Z up; origin at the tank window) -> NeXus frame (x left, y up, z along the
    beam; origin at the sample).

    Both frames are right-handed and the axes are just cyclically permuted:
      x_nexus = Y_survey, y_nexus = Z_survey, z_nexus = X_survey
    Then the origin is moved from the window to the sample position, which lies
    upstream of the window by SAMPLE_TO_TANK_WINDOW_DISTANCE."""
    X, Y, Z = point
    return np.array([Y, Z, X + SAMPLE_TO_TANK_WINDOW_DISTANCE])


# ---------------------------------------------------------------------------
# SAM-606 (2026): tube ends of all 9 banks in the TANK WINDOW coordinate system
# ---------------------------------------------------------------------------

SAM606_FILE_NAME = "ESS.NSS.D03.LOKI.OB.DETECTOR_PIPE_CENTRES.TANK_WINDOW_CS.txt"

# Labels like "C2RP16": C1 / C2 = front / middle carriage, U/D/L/R = up / down /
# left / right bank of the carriage, P<n> = the surveyors' "pipe" number, which
# does NOT follow our tube numbering. C3 = the rear carriage (bank 0), measured later
# (SAM-606 Rev2, "C3LP1" / "C3RP1" = the left / right end of pipe 1): its points are
# in the file as "C3P1I" / "C3P1II" (I = left = the first pixel end), only the two
# outermost tubes (P1, P56; the file Rev2 also has P28, consistent with them).
# The tubes were identified by comparing the measured positions with the nominal
# tube positions (survey.py re-checks this every time it runs, see
# check_tube_ids_against_nominal).
SAM606_LABEL_MAP = {
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
    # id 20). C1UP8II is replaced by an extrapolated end of tube UP9 (see
    # sam606_fix_bank5_mislabelled_tube), so after that "C1UP8" means tube UP9.
    "C1UP8": (5, 19),  # front top, tube "UP9" (9th from bottom of 28 rows)
    "C1UP3": (5, 25),  # front top, 3rd tube from the bottom
    "C1LP11": (6, 21),  # front left, 22nd tube from the left (of 32)
    "C1LP1": (6, 31),  # front left, rightmost (innermost) tube
    "C1DP1": (7, 0),  # front bottom, topmost (innermost) tube
    "C1DP31": (7, 18),  # front bottom, 19th tube of 20 (label "31" is misleading)
    "C1RP1": (8, 0),  # front right, leftmost (innermost) tube
    "C1RP11": (8, 10),  # front right, 11th tube from the left (of 32)
    # --- rear carriage (C3), SAM-606 Rev2; here the P numbers are the rows + 1 ---
    "C3P1": (0, 0),  # rear, topmost tube (1st of 56 rows)
    "C3P56": (0, 55),  # rear, bottom tube (56th of 56 rows)
}

BANK5_FIX_TOLERANCE = 0.25  # [tube pitch] accepted deviation of the discarded point


def sam606_fix_bank5_mislabelled_tube(points, verbose=True):
    """In bank 5 the two "C1UP8" points belong to two different tubes:
    C1UP8I is the start of tube UP9, C1UP8II is the end of tube UP8.

    Assuming that the tubes of a bank are parallel (checked for the other banks
    in survey.check_tube_parallelism), the missing end of tube UP9 is
    extrapolated from its measured start (C1UP8I), using the vector between the
    two ends of the other measured tube of the bank (C1UP3I -> C1UP3II). The
    measured C1UP8II is discarded, after checking that it really is one tube
    pitch away from the extrapolated UP9 end (a warning is printed otherwise).

    Works in any frame (it only uses differences of points). Modifies and
    returns `points`."""
    pitch = c.TUBE_GRID_SIDE
    up9_end = points["C1UP8I"] + (points["C1UP3II"] - points["C1UP3I"])
    separation = np.linalg.norm(points["C1UP8II"] - up9_end)
    if verbose:
        print(
            f"Bank 5 fix: C1UP8II replaced by the extrapolated end of tube UP9. "
            f"Distance of the discarded point (tube UP8) from it: {separation:.1f} mm "
            f"(expected ~{pitch} mm = 1 tube pitch)"
        )
    if abs(separation - pitch) > BANK5_FIX_TOLERANCE * pitch:
        print(
            f"WARNING: bank 5 fix: the discarded point C1UP8II is {separation:.1f} mm from the "
            "extrapolated end of tube UP9, not ~1 tube pitch; check the bank 5 assumption!"
        )
    points["C1UP8II"] = up9_end
    return points


SAM606 = SurveyFile(
    file_name=SAM606_FILE_NAME,
    checksum="cbdc6a9693206af3198f54d0230450c065d3e5df138e71c0ea0e6c46cfaab7d1",
    description="SAM-606 survey (2026): both ends of 2 front layer tubes of each of the 9 banks "
    "(36 points; the bank 0 points from the revision SAM-606 Rev2), on the tube axis at the physical "
    "tube ends (including the end electronics), in the TANK WINDOW coordinate system (sample to window "
    f"distance {SAMPLE_TO_TANK_WINDOW_DISTANCE} mm), with the detector carriage at 5.0 m along z "
    "(carriage_position 5000 mm).",
    frame=tank_window_frame,
    label_map=SAM606_LABEL_MAP,
    fixes=(sam606_fix_bank5_mislabelled_tube,),
    bank_notes={5: "tube UP9 constructed parallel to UP3 (bank 5 fix)"},
    calibration_name="2026-September-SAM-606",
    carriage_position=5000.0,  # the carriage at 5.0 m (the rear bank at 10 m) during the bank 0 measurement
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SURVEY_FILES = {entry.file_name: entry for entry in (SAM606,)}
DEFAULT_SURVEY_FILE = SAM606.path  # the survey used when no file is given


def survey_file_info(filename):
    """The SurveyFile entry of a survey file, found by its file name. Raises
    ValueError if the file is not known, or if the entry has a checksum and the
    content does not match it (changed file: it needs a new entry, see the README)."""
    name = os.path.basename(filename)
    if name not in SURVEY_FILES:
        raise ValueError(
            f"unknown survey file {name!r}: add an entry for it in survey_files.py "
            f"(known files: {', '.join(sorted(SURVEY_FILES))})"
        )
    entry = SURVEY_FILES[name]
    if entry.checksum is None:
        return entry
    checksum = content_checksum(filename)
    if checksum != entry.checksum:
        raise ValueError(
            f"the content of {filename} does not match the survey_files.py entry of {name!r} "
            f"(checksum {checksum}, expected {entry.checksum}); a changed survey file needs "
            "its own entry"
        )
    return entry
