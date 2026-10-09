"""
Bank calibration file for the Geant4 geometry (G4GeoLoki).

The Geant4 geometry places each calibrated bank as a rigid body from four
vectors per bank (see G4GeoLoki/libinc/BankCalibration.hh in
https://github.com/ess-dg/dgcode_projects/tree/main/LOKI/G4GeoLoki), selected
there with the geometry parameter bank_calibration=<name>. The file of the
calibration <name> is G4GeoLoki/data/bank_calibration_<name>.txt.

Format (version 1): lines starting with '#' are comments; the first other line
is "loki_bank_calibration 1", then "name <name>", then one line per bank:
  bank <id> Fx Fy Fz ux uy uz wx wy wz nx ny nz
in the NeXus frame (x horizontal, positive to the left looking along the beam,
y up, z along the beam, origin at the sample), [mm]:
  F  front_face_centre: centre of the front face of the bank (the plane
     touching the front layer tubes, in the middle of the front layer and of
     the straw length)
  u  vector_along_tubes: from the last pixel end to the first pixel end
  w  vector_perpendicular_tubes: in the tube plane, towards tube 0
  n  vector_perpendicular_layers: u x w, pointing away from the sample
(the attributes of calibration.Bank). All 9 banks are written; a bank that is
not in the calibration input (e.g. bank 0 of SAM-606) can be taken from another
geometry (loki-calibrate: REFERENCE_BANKS_DATA), marked with a comment. The rear
bank (0) moves along the beam: Geant4 uses its axes and the x and y of its F,
while the z of F is given by the rear detector distance of the simulation.
The numbers are written with full double precision (17 significant digits).
"""

import os
import re
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
FILE_PREFIX = "bank_calibration_"
FILE_SUFFIX = ".txt"
# the built-in nominal placements of the Geant4 geometry (no file)
NOMINAL_NAME = "nominal-geant4-geometry"
BANK_IDS = tuple(range(9))
_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # used with fullmatch


def file_name_for(name):
    """The file name of the calibration <name>: bank_calibration_<name>.txt"""
    check_name(name)
    return f"{FILE_PREFIX}{name}{FILE_SUFFIX}"


def check_name(name):
    """A calibration name: letters, digits, '.', '_' and '-' (starting with a
    letter or digit), not the reserved nominal-geant4-geometry (the built-in nominal
    placement of Geant4). Raises ValueError."""
    if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"invalid calibration name {name!r}: use letters, digits, '.', '_' and '-' only "
            "(starting with a letter or digit)"
        )
    if name == NOMINAL_NAME:
        raise ValueError(
            f"the calibration name {NOMINAL_NAME!r} is reserved for the nominal geometry"
        )


def format_calibration(detectors, name, comments=(), fallback=None, fallback_label=None):
    """The text of the calibration file of the banks of a Detectors object
    (calibration.Detectors). comments: lines written as '# ' comments in the
    header (provenance). fallback: another Detectors object, from which the banks
    that are not in detectors are taken (with a comment naming fallback_label).
    Raises ValueError for an invalid name or if there is no bank to write."""
    check_name(name)
    sources = {}
    for bank_id in BANK_IDS:
        if bank_id in detectors.banks:
            sources[bank_id] = detectors.banks[bank_id]
        elif fallback is not None and bank_id in fallback.banks:
            sources[bank_id] = fallback.banks[bank_id]
    if not sources:
        raise ValueError("no bank to write to the Geant4 calibration file")
    lines = [
        "# LOKI detector bank calibration for the Geant4 geometry (G4GeoLoki):",
        f"#   bank_calibration={name}",
    ]
    lines += [f"# {comment}" if comment else "#" for comment in comments]
    for bank_id in sorted(sources):
        if bank_id not in detectors.banks:
            lines.append(
                f"# Bank {bank_id} is not in the calibration input: written from {fallback_label}."
            )
    lines += [
        "# Frame: NeXus (x horizontal, positive to the left looking along the beam, y up,",
        "#   z along the beam, origin at the sample), [mm].",
        "# F: front face centre; u: along the tubes (last -> first pixel end);",
        "# w: across the tubes in the tube plane, towards tube 0; n = u x w (away from the sample).",
        "# Bank 0 (rear): Geant4 uses the axes and the x, y of F; the z of F is set by the rear",
        "#   detector distance of the simulation.",
        "# Banks that are not listed keep their nominal Geant4 placement.",
        f"loki_bank_calibration {FORMAT_VERSION}",
        f"name {name}",
        "# bank Fx Fy Fz ux uy uz wx wy wz nx ny nz",
    ]
    for bank_id, bank in sorted(sources.items()):
        values = np.concatenate(
            [
                bank.front_face_centre,
                bank.vector_along_tubes,
                bank.vector_perpendicular_tubes,
                bank.vector_perpendicular_layers,
            ]
        )
        lines.append(f"bank {bank_id} " + " ".join(f"{float(v):.17g}" for v in values))
    return "\n".join(lines) + "\n"


def write_calibration_file(detectors, path, name, comments=(), fallback=None, fallback_label=None):
    """Write the calibration file (see format_calibration). path: a file, or an
    existing directory (the file bank_calibration_<name>.txt is written in it,
    e.g. G4GeoLoki/data/ of dgcode_projects). Returns the Path written. Raises
    ValueError for a directory that does not exist (a path ending with '/')."""
    text = format_calibration(detectors, name, comments, fallback, fallback_label)
    is_directory = str(path).endswith(("/", os.sep))
    path = Path(path)
    if is_directory and not path.is_dir():
        raise ValueError(f"the directory {path} does not exist")
    if path.is_dir():
        path = path / file_name_for(name)
    path.write_text(text)
    return path


def read_calibration_file(path):
    """Read a calibration file: (name, {bank_id: {"F", "u", "w", "n"}}) with
    numpy arrays. Checks the format like the Geant4 reader (ValueError)."""
    name = None
    banks = {}
    format_seen = False
    with open(path) as file:
        for line_number, line in enumerate(file, 1):
            words = line.split()
            if not words or words[0].startswith("#"):
                continue

            def fail(message, line_number=line_number):
                raise ValueError(f"{path}:{line_number}: {message}")

            if not format_seen:
                if words != ["loki_bank_calibration", str(FORMAT_VERSION)]:
                    fail(f"the first line must be 'loki_bank_calibration {FORMAT_VERSION}'")
                format_seen = True
            elif words[0] == "name":
                if name is not None or len(words) != 2:
                    fail("invalid or repeated 'name' line")
                name = words[1]
                try:
                    check_name(name)
                except ValueError as error:
                    fail(str(error))
            elif words[0] == "bank":
                if name is None:
                    fail("the 'name' line must come before the bank lines")
                if len(words) != 14:
                    fail("a bank line needs the bank id and 12 numbers (F, u, w, n)")
                try:
                    bank_id = int(words[1])
                except ValueError:
                    fail(f"invalid bank id {words[1]!r}")
                if bank_id not in BANK_IDS or bank_id in banks:
                    fail(f"invalid or repeated bank id {bank_id}")
                try:
                    values = np.array([float(v) for v in words[2:]])
                except ValueError:
                    fail("the 12 values of a bank line must be numbers")
                if not np.all(np.isfinite(values)):
                    fail("the numbers must be finite")
                bank = dict(zip("Fuwn", values.reshape(4, 3)))
                basis = np.array([bank["u"], bank["w"], bank["n"]])
                if not np.allclose(basis @ basis.T, np.eye(3), rtol=0, atol=1e-9):
                    fail("u, w and n must be perpendicular unit vectors")
                if np.dot(np.cross(bank["u"], bank["w"]), bank["n"]) < 0:
                    fail("n must be u x w (right-handed basis)")
                if np.dot(bank["n"], bank["F"]) <= 0:
                    fail("the layer normal n must point away from the sample")
                banks[bank_id] = bank
            else:
                fail(f"unknown line {words[0]!r}")
    if not format_seen or name is None:
        raise ValueError(f"{path}: incomplete file (the format and name lines are required)")
    return name, banks
