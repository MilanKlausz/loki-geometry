"""
Geometry information of the LOKI detector banks (banks, tubes, straws and
pixels: positions, directions and ids) for a chosen bank calibration. The
calibrations (calibration_corners) are also those of the NeXus-JSON output
(loki-pixel-json), so both give the same pixel positions.

Usage
-----
  loki-info [--calibration NAME] [--pixels-per-straw N] [--rear-detector-distance MM]
            [--list-calibrations | --pixel ID | --detector-number N | --bank B [--tube T [--straw S]]]
            [--unit mm|m] [--precision P]
Without a query it prints a summary of all banks.

Calibrations (by name, see available_calibrations())
-----------------------------------------------------
  nominal-cad-geometry     the intended geometry: the corners of
                           REFERENCE_BANKS_DATA (exact CAD values)
  nominal-geant4-geometry  the nominal geometry of the Geant4 model (G4GeoLoki,
                           https://github.com/ess-dg/dgcode_projects/tree/main/LOKI/G4GeoLoki,
                           built-in tables; via the Python port geant4.py). It is
                           the bank_calibration=nominal-geant4-geometry of Geant4.
                           It differs from the CAD values by up to ~1 mm (rounded
                           CAD values in Geant4).
  <calibration name>       a survey calibration: the calibration_name of a
                           survey_files.py entry (e.g. 2026-September-SAM-606).
                           Banks that are not in the survey are taken from
                           nominal-cad-geometry (as in the Geant4 calibration file
                           written by loki-calibrate).
The default is the calibration of the default survey file (DEFAULT_SURVEY_FILE of
survey_files.py: 2026-September-SAM-606), the same as the default of the Geant4
geometry.

Model and conventions
---------------------
The pixel geometry of each bank is built from its 8 corners by bank_pixels.py
(it reproduces the Geant4 pixel centres exactly for the same placement).
Positions are in the NeXus frame (x horizontal, positive to the left looking
along the beam, y up, z along the beam, origin at the sample) in mm.
  pixel id         0-based, as in Geant4 and the analysis programs:
                   first pixel id of the bank + (tube * 7 + straw) * N + pixel in straw
  detector number  pixel id + 1 (the NeXus detector_number / ICD numbering)
  tube id          layer * (tubes per layer) + row; layer 0 = front layer (facing
                   the sample), row 0 = the top tube (horizontal banks) or the
                   leftmost tube looking along the beam (vertical banks)
  straw id         0-6, straw 3 on the tube axis
  pixel in straw   0 at the first pixel end of the straw
The rear bank (0) moves along the beam: with rear_detector_distance [mm] it is
moved along z so that the z of its front face centre is that distance (as the
Geant4 geometry places it); default: as in the calibration (CAD: 4999.8 mm,
Geant4 nominal: 5000 mm).

Per bank (BankGeometry): the front face centre F (the plane touching the front
layer tubes, in the middle of the front layer and of the straw length) and the
unit vectors u (along the tubes, from the last to the first pixel end), w
(across the tubes in the tube plane, towards tube 0) and n = u x w (the layer
normal, away from the sample), as in the Geant4 calibration file.
"""

import argparse
import contextlib
import io
import os
import sys
from dataclasses import dataclass

import numpy as np

from . import bank_pixels as lbp
from . import constants as c
from . import geant4
from .geant4_calibration import NOMINAL_NAME as _GEANT4_NOMINAL_NAME
from .survey_files import DEFAULT_SURVEY_FILE, SURVEY_FILES

NOMINAL_CAD = "nominal-cad-geometry"
NOMINAL_GEANT4 = _GEANT4_NOMINAL_NAME  # "nominal-geant4-geometry", the same name as in Geant4
# the calibration of the default survey file (2026-September-SAM-606)
DEFAULT_CALIBRATION = (
    SURVEY_FILES[os.path.basename(DEFAULT_SURVEY_FILE)].calibration_name or NOMINAL_CAD
)


def available_calibrations():
    """{name: description} of the calibrations that can be used."""
    from .survey_files import SURVEY_FILES

    names = {
        NOMINAL_CAD: "the intended geometry (REFERENCE_BANKS_DATA, exact CAD values)",
        NOMINAL_GEANT4: "the nominal geometry of the Geant4 model (G4GeoLoki built-in tables)",
    }
    for entry in SURVEY_FILES.values():
        if entry.calibration_name:
            names[entry.calibration_name] = f"survey {entry.file_name}"
    return names


def _survey_corners(calibration):
    """The corners (straw ends, not rounded) of a survey calibration, by its name."""
    from .calibration import Detectors
    from .survey import get_measured_calibration_data
    from .survey_files import SURVEY_FILES

    entry = next(e for e in SURVEY_FILES.values() if e.calibration_name == calibration)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        data = get_measured_calibration_data(entry.path, verbose=False)
        detectors = Detectors(data, True, intended=None, run_checks=False)
    _report_survey_warnings(calibration, output.getvalue())
    return detectors.formatted_data(as_straw_ends=True, decimals=None)


_reported_survey_warnings = set()


def _report_survey_warnings(calibration, survey_output):
    """The failed survey checks (the WARNING lines of the survey output, which is not shown
    otherwise) on stderr, once per calibration and set of warnings."""
    warnings = tuple(line.strip() for line in survey_output.splitlines() if "WARNING" in line)
    if not warnings or (calibration, warnings) in _reported_survey_warnings:
        return
    _reported_survey_warnings.add((calibration, warnings))
    print(
        f"WARNING: {len(warnings)} survey check(s) of the calibration {calibration} failed (the "
        "geometry may be wrong; run loki-survey for the details):",
        file=sys.stderr,
    )
    for line in warnings:
        print(f"    {line}", file=sys.stderr)


def calibration_corners(calibration=DEFAULT_CALIBRATION, rear_detector_distance=None):
    """The corners of all 9 banks (REFERENCE_BANKS_DATA format, straw ends [mm]) of a
    calibration (see the module docstring). Raises ValueError for an unknown name."""
    from .reference_data import REFERENCE_BANKS_DATA

    if calibration not in available_calibrations():
        raise ValueError(
            f"unknown calibration {calibration!r} (known: {', '.join(available_calibrations())})"
        )
    if calibration == NOMINAL_GEANT4:
        rear = (
            c.DEFAULT_REAR_BANK_DISTANCE
            if rear_detector_distance is None
            else rear_detector_distance
        )
        return {b: geant4.bank_corners(b, rear_bank_distance=rear) for b in c.BANK_IDS}
    corners = {b: REFERENCE_BANKS_DATA[b] for b in c.BANK_IDS}
    if calibration != NOMINAL_CAD:
        corners.update(_survey_corners(calibration))
    if rear_detector_distance is not None:
        corners[0] = _rear_bank_at(corners[0], rear_detector_distance)
    return {b: _frozen_corners(corners[b]) for b in c.BANK_IDS}


def _frozen_corners(corners):
    """A copy of bank corners with tuples only (the package constants can not be changed through it)."""
    return {
        "A": tuple(tuple(float(v) for v in point) for point in corners["A"]),
        "B": tuple(tuple(float(v) for v in point) for point in corners["B"]),
        "num_tubes": corners["num_tubes"],
    }


def calibration_bank_sources(calibration=DEFAULT_CALIBRATION):
    """{bank_id: where the corners of the bank come from} for a calibration."""
    if calibration not in available_calibrations():
        raise ValueError(
            f"unknown calibration {calibration!r} (known: {', '.join(available_calibrations())})"
        )
    if calibration == NOMINAL_GEANT4:
        return {b: "the nominal Geant4 geometry (geant4.py)" for b in c.BANK_IDS}
    cad = "REFERENCE_BANKS_DATA (the intended CAD geometry)"
    if calibration == NOMINAL_CAD:
        return {b: cad for b in c.BANK_IDS}
    entry = next(e for e in SURVEY_FILES.values() if e.calibration_name == calibration)
    surveyed = set(_survey_corners(calibration))
    survey = f"survey {entry.file_name}" + (f" (sha256 {entry.checksum})" if entry.checksum else "")
    return {b: survey if b in surveyed else cad for b in c.BANK_IDS}


def _rear_bank_at(corners, distance):
    """The rear bank corners moved along z so that the z of the front face centre is distance."""
    frame = lbp.BankFrame.from_corners(
        corners, lbp.tubes_per_layer(0), straw_length=c.STRAW_LENGTHS[0]
    )
    shift = np.array([0.0, 0.0, distance - _front_face_centre(frame)[2]])
    moved = {
        key: [tuple(float(v) for v in np.asarray(p) + shift) for p in corners[key]] for key in "AB"
    }
    moved["num_tubes"] = corners["num_tubes"]
    return moved


def _front_face_centre(frame):
    """Centre of the front face: the middle of the front layer tube axes, one tube radius
    towards the sample."""
    return (
        frame.origin
        + frame.row_step * (frame.rows - 1) / 2
        + frame.along * frame.straw_length / 2
        - frame.depth_axis * c.TUBE_OUTER_RADIUS
    )


@dataclass(frozen=True)
class BankGeometry:
    """Placement summary of a bank [mm]."""

    bank_id: int
    name: str
    front_face_centre: np.ndarray  # F
    along_tubes: np.ndarray  # u: last -> first pixel end
    across_tubes: np.ndarray  # w: towards tube 0, in the tube plane
    layer_normal: np.ndarray  # n = u x w, away from the sample
    straw_length: float
    tubes_per_layer: int
    number_of_tubes: int
    first_pixel_id: int
    number_of_pixels: int

    @property
    def distance(self):
        """Distance of the front face centre from the sample [mm]."""
        return float(np.linalg.norm(self.front_face_centre))


@dataclass(frozen=True)
class PixelInfo:
    """A pixel: ids and geometry [mm]."""

    pixel_id: int
    bank_id: int
    tube_id: int
    straw_id: int
    pixel_in_straw: int
    layer: int
    row: int
    centre: np.ndarray
    straw_first_end: np.ndarray  # the first pixel end of the straw
    straw_last_end: np.ndarray
    pixel_length: float

    @property
    def detector_number(self):
        return self.pixel_id + 1

    @property
    def straw_direction(self):
        """Unit vector along the straw, from the first to the last pixel end."""
        d = self.straw_last_end - self.straw_first_end
        return d / np.linalg.norm(d)


class LokiGeometry:
    """Geometry of all 9 banks for a calibration (see the module docstring)."""

    def __init__(
        self,
        calibration=DEFAULT_CALIBRATION,
        pixels_per_straw=c.DEFAULT_PIXELS_PER_STRAW,
        rear_detector_distance=None,
    ):
        if (
            not isinstance(pixels_per_straw, (int, np.integer))
            or isinstance(pixels_per_straw, bool)
            or pixels_per_straw < 1
        ):
            raise ValueError(
                f"pixels_per_straw must be a positive integer, not {pixels_per_straw!r}"
            )
        self.calibration = calibration
        self.pixels_per_straw = int(pixels_per_straw)
        self.rear_detector_distance = rear_detector_distance
        self.corners = calibration_corners(calibration, rear_detector_distance)
        self._bank_pixels_cache = {}  # bank id -> BankPixels (computed when needed)
        self._frames = {
            b: lbp.BankFrame.from_corners(
                self.corners[b], lbp.tubes_per_layer(b), straw_length=c.STRAW_LENGTHS[b]
            )
            for b in c.BANK_IDS
        }
        self._first_ids = [
            int(lbp.bank_first_pixel_id(b, self.pixels_per_straw, first_pixel_id=0))
            for b in c.BANK_IDS
        ]
        self.total_number_of_pixels = int(
            lbp.bank_first_pixel_id(c.NUMBER_OF_BANKS, self.pixels_per_straw, first_pixel_id=0)
        )

    # --- ids ------------------------------------------------------------------
    def first_pixel_id(self, bank_id):
        self._check_bank(bank_id)
        return self._first_ids[bank_id]

    def number_of_pixels(self, bank_id):
        self._check_bank(bank_id)
        return int(lbp.pixels_in_bank(bank_id, self.pixels_per_straw))

    def pixel_id(self, bank_id, tube_id, straw_id, pixel_in_straw):
        """0-based pixel id of a pixel given by its bank, tube, straw and index in the straw."""
        self._check_bank(bank_id)
        _check_index("tube id", tube_id, c.TUBES_PER_BANK[bank_id])
        _check_index("straw id", straw_id, c.STRAWS_PER_TUBE)
        _check_index("pixel in straw", pixel_in_straw, self.pixels_per_straw)
        n = self.pixels_per_straw
        return (
            self._first_ids[bank_id] + (tube_id * c.STRAWS_PER_TUBE + straw_id) * n + pixel_in_straw
        )

    def decode(self, pixel_id):
        """(bank, tube, straw, pixel in straw) of a 0-based pixel id."""
        _check_index("pixel id", pixel_id, self.total_number_of_pixels)
        bank_id = max(b for b in c.BANK_IDS if self._first_ids[b] <= pixel_id)
        local = pixel_id - self._first_ids[bank_id]
        n = self.pixels_per_straw
        straw_in_bank, pixel_in_straw = divmod(local, n)
        tube_id, straw_id = divmod(straw_in_bank, c.STRAWS_PER_TUBE)
        return bank_id, tube_id, straw_id, pixel_in_straw

    # --- geometry ---------------------------------------------------------------
    def frame(self, bank_id):
        """The BankFrame (bank_pixels.py) of a bank."""
        self._check_bank(bank_id)
        return self._frames[bank_id]

    def bank(self, bank_id):
        """Placement summary of a bank (BankGeometry)."""
        f = self.frame(bank_id)
        u, w, n = -f.along, -f.row_axis, f.depth_axis
        return BankGeometry(
            bank_id,
            c.BANK_NAMES[bank_id],
            _front_face_centre(f),
            u,
            w,
            n,
            f.straw_length,
            f.rows,
            c.TUBES_PER_BANK[bank_id],
            self._first_ids[bank_id],
            self.number_of_pixels(bank_id),
        )

    def tube_ends(self, bank_id, tube_id):
        """(first pixel end, last pixel end) of a tube, on its axis (the centre straw)."""
        return self.straw_ends(bank_id, tube_id, c.CENTRE_STRAW)

    def straw_ends(self, bank_id, tube_id, straw_id):
        """(first pixel end, last pixel end) of a straw, on its axis."""
        f = self.frame(bank_id)
        _check_index("tube id", tube_id, c.TUBES_PER_BANK[bank_id])
        _check_index("straw id", straw_id, c.STRAWS_PER_TUBE)
        first = f.origin + lbp.tube_offsets(f)[tube_id] + lbp.straw_offsets(f)[straw_id]
        return first, first + f.along * f.straw_length

    def pixel_centre(self, pixel_id):
        """Centre of a pixel (0-based pixel id) [mm]."""
        return self.pixel(pixel_id).centre

    def pixel(self, pixel_id):
        """Ids and geometry of a pixel (PixelInfo), by its 0-based pixel id."""
        bank_id, tube_id, straw_id, k = self.decode(pixel_id)
        first, last = self.straw_ends(bank_id, tube_id, straw_id)
        length = self._frames[bank_id].straw_length / self.pixels_per_straw
        rows = self._frames[bank_id].rows
        return PixelInfo(
            pixel_id,
            bank_id,
            tube_id,
            straw_id,
            k,
            tube_id // rows,
            tube_id % rows,
            first + (last - first) * (k + 0.5) / self.pixels_per_straw,
            first,
            last,
            length,
        )

    def bank_pixel_centres(self, bank_id):
        """Pixel centres of a bank, shape (pixels, 3), in pixel id order."""
        return self._bank_pixels(bank_id).pixel_centres

    def _bank_pixels(self, bank_id):
        self._check_bank(bank_id)
        if bank_id not in self._bank_pixels_cache:
            self._bank_pixels_cache[bank_id] = lbp.compute_bank_pixels(
                bank_id, self.corners[bank_id], self.pixels_per_straw
            )
        return self._bank_pixels_cache[bank_id]

    def _check_bank(self, bank_id):
        _check_index("bank id", bank_id, c.NUMBER_OF_BANKS)


def _check_index(name, value, size):
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or not 0 <= value < size:
        raise ValueError(f"{name} must be an integer in 0 .. {size - 1}, not {value!r}")


# --- command line ---------------------------------------------------------------


def _number(value, precision):
    text = f"{value:.{precision}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text  # no "-0.000"


def _fmt(vector, scale, precision):
    return "(" + ", ".join(_number(v / scale, precision) for v in vector) + ")"


def _print_bank(geometry, b, unit, scale, precision, detailed):
    bank = geometry.bank(b)
    last = bank.first_pixel_id + bank.number_of_pixels - 1
    if not detailed:
        print(
            f"{b:4}  {bank.name:13} {bank.number_of_tubes:5} {bank.number_of_pixels:9}  "
            f"{bank.first_pixel_id:8}-{last:<8}  {_fmt(bank.front_face_centre, scale, precision):>32}  "
            f"{bank.distance / scale:.{precision}f}"
        )
        return
    print(
        f"Bank {b} ({bank.name}): {bank.number_of_tubes} tubes ({bank.tubes_per_layer} per layer, 4 layers), "
        f"straw length {bank.straw_length / scale:.{precision}f} {unit}"
    )
    print(
        f"  pixel ids {bank.first_pixel_id}..{last} (detector numbers {bank.first_pixel_id + 1}..{last + 1})"
    )
    print(
        f"  front face centre F [{unit}]: {_fmt(bank.front_face_centre, scale, precision)}, "
        f"distance from the sample {bank.distance / scale:.{precision}f}"
    )
    print(f"  u (along the tubes, last -> first pixel end): {_fmt(bank.along_tubes, 1, 9)}")
    print(f"  w (across the tubes, towards tube 0):          {_fmt(bank.across_tubes, 1, 9)}")
    print(f"  n (layer normal, away from the sample):        {_fmt(bank.layer_normal, 1, 9)}")
    print(f"  corners (straw ends on the tube axes) [{unit}]:")
    for key, label in (("A", "first pixel end"), ("B", "last pixel end")):
        for tube_label, point in zip(
            ("front top", "back top", "front bottom", "back bottom"), geometry.corners[b][key]
        ):
            print(f"    {key} {tube_label:12} ({label}): {_fmt(point, scale, precision)}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Geometry information of the LOKI detector banks (banks, tubes, straws, "
        "pixels) for a bank calibration. Positions: NeXus frame (x left, y up, z along the beam, "
        "origin at the sample)."
    )
    parser.add_argument(
        "-c",
        "--calibration",
        default=DEFAULT_CALIBRATION,
        help=f"bank calibration (default: {DEFAULT_CALIBRATION}; see --list-calibrations)",
    )
    parser.add_argument("--list-calibrations", action="store_true", help="list the calibrations")
    parser.add_argument(
        "-n",
        "--pixels-per-straw",
        type=int,
        default=c.DEFAULT_PIXELS_PER_STRAW,
        help=f"pixels per straw (default: {c.DEFAULT_PIXELS_PER_STRAW})",
    )
    parser.add_argument(
        "-r",
        "--rear-detector-distance",
        type=float,
        default=None,
        help="the z of the front face centre of the rear bank [mm], i.e. the rear detector distance "
        "(default: as in the calibration)",
    )
    query = parser.add_mutually_exclusive_group()
    query.add_argument("-p", "--pixel", type=int, help="a pixel by its 0-based pixel id (Geant4)")
    query.add_argument(
        "-d", "--detector-number", type=int, help="a pixel by its detector number (pixel id + 1)"
    )
    query.add_argument(
        "-b", "--bank", type=int, help="a bank (with --tube: a tube, with --straw: a straw)"
    )
    parser.add_argument("-t", "--tube", type=int, help="a tube of --bank")
    parser.add_argument("-s", "--straw", type=int, help="a straw of --tube")
    parser.add_argument(
        "--unit", choices=["mm", "m"], default="mm", help="position unit (default: mm)"
    )
    parser.add_argument(
        "--precision", type=int, default=3, help="decimals of the positions (default: 3)"
    )
    args = parser.parse_args(argv)
    if args.tube is not None and args.bank is None:
        parser.error("--tube needs --bank")
    if args.straw is not None and args.tube is None:
        parser.error("--straw needs --tube")
    if args.precision < 0:
        parser.error("--precision must not be negative")

    if args.list_calibrations:
        for name, description in available_calibrations().items():
            default = " (default)" if name == DEFAULT_CALIBRATION else ""
            print(f"{name}{default}: {description}")
        return 0

    try:
        geometry = LokiGeometry(
            args.calibration, args.pixels_per_straw, args.rear_detector_distance
        )
    except ValueError as error:
        parser.error(str(error))
    unit, precision = args.unit, args.precision
    scale = 1000.0 if unit == "m" else 1.0
    rear = (
        ""
        if args.rear_detector_distance is None
        else f", rear detector distance {args.rear_detector_distance} mm"
    )
    print(
        f"Calibration: {geometry.calibration}, {geometry.pixels_per_straw} pixels per straw{rear}"
    )

    try:
        if args.pixel is not None or args.detector_number is not None:
            if (
                args.detector_number is not None
                and not 1 <= args.detector_number <= geometry.total_number_of_pixels
            ):
                parser.error(
                    f"detector number must be in 1 .. {geometry.total_number_of_pixels}, not {args.detector_number}"
                )
            pixel_id = args.pixel if args.pixel is not None else args.detector_number - 1
            p = geometry.pixel(pixel_id)
            print(f"Pixel id {p.pixel_id} (detector number {p.detector_number})")
            print(
                f"  bank {p.bank_id}, tube {p.tube_id} (layer {p.layer}, row {p.row}), straw {p.straw_id}, "
                f"pixel {p.pixel_in_straw} in the straw"
            )
            print(f"  centre [{unit}]: {_fmt(p.centre, scale, precision)}")
            print(
                f"  pixel length: {p.pixel_length / scale:.{precision}f} {unit}, straw direction (first -> "
                f"last pixel end): {_fmt(p.straw_direction, 1, 9)}"
            )
            print(
                f"  straw ends [{unit}]: first pixel end {_fmt(p.straw_first_end, scale, precision)}, "
                f"last pixel end {_fmt(p.straw_last_end, scale, precision)}"
            )
        elif args.bank is not None and args.tube is not None:
            b, t = args.bank, args.tube
            if args.straw is None:
                first, last = geometry.tube_ends(b, t)
                rows = geometry.frame(b).rows
                print(f"Bank {b}, tube {t} (layer {t // rows}, row {t % rows})")
                print(
                    f"  axis [{unit}]: first pixel end {_fmt(first, scale, precision)}, "
                    f"last pixel end {_fmt(last, scale, precision)}"
                )
                straws = range(c.STRAWS_PER_TUBE)
            else:
                print(f"Bank {b}, tube {t}, straw {args.straw}")
                straws = [args.straw]
            for s in straws:
                first, last = geometry.straw_ends(b, t, s)
                ids = (
                    geometry.pixel_id(b, t, s, 0),
                    geometry.pixel_id(b, t, s, geometry.pixels_per_straw - 1),
                )
                print(
                    f"  straw {s}: pixel ids {ids[0]}..{ids[1]}, ends [{unit}] "
                    f"{_fmt(first, scale, precision)} -> {_fmt(last, scale, precision)}"
                )
        elif args.bank is not None:
            _print_bank(geometry, args.bank, unit, scale, precision, True)
        else:
            print(
                f"bank  name          tubes    pixels  pixel ids          {'front face centre [' + unit + ']':>32}  "
                "distance"
            )
            for b in c.BANK_IDS:
                _print_bank(geometry, b, unit, scale, precision, False)
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
