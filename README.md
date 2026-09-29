# loki-geometry – LOKI detector bank geometry

Tools to derive the 8 corner positions of each LOKI detector bank from the as-built survey of the
detector tubes (SAM-606), and to turn corner positions into the pixel geometry (bank positions,
detector numbers and pixel offsets of all 1,605,632 pixels) for the NeXus/JSON instrument file.
The pixel positions are checked against the LOKI Geant4 geometry, [G4GeoLoki](https://github.com/ess-dg/dgcode_projects/tree/main/LOKI/G4GeoLoki),
through a Python port of its pixel position code (`geant4`).

The Python package `loki_geometry` (distribution name `loki-geometry`) needs only Python ≥ 3.9 and numpy.
The tests need pytest.

## Installation

```
pip install .                          # from a clone of the repository
pip install -e ".[test]"               # editable install for development, with pytest
pip install "git+<repository URL>"     # directly from the git repository
```

This installs the package and the command-line tools `loki-calibrate`, `loki-survey`, `loki-pixel-json`,
`loki-verify-json`, `loki-geant4-validate` and `loki-nominal-corners`. The same tools can be run without installing
as `python3 -m loki_geometry.<module>` with `PYTHONPATH=src`. Run the tests with `python3 -m pytest` from the
repository root; no installation is needed for that.

## Conventions

- **Coordinates:** NeXus/Geant4 frame, in mm: x horizontal (positive to the LEFT, looking along the beam),
  y up, z along the beam, origin at the nominal sample position.
- **Banks:** 0 rear, 1 mid top, 2 mid left, 3 mid bottom, 4 mid right, 5 front top, 6 front left,
  7 front bottom, 8 front right. Vertical tubes: banks 2, 4, 6, 8. Packs mounted upside down: 1, 2, 5, 6.
- **Local tube id:** zero-indexed, counted in the FRONT layer (the one facing the sample; all surveyed tubes are front layer tubes).
  Horizontal banks: 0 = topmost tube. Vertical banks: 0 = leftmost tube (beam's view).
  Front layer rows are also the Geant4 tube ids 0 … 2·packs−1.
- **Tube ends:** `first_pixel_end` / `last_pixel_end` are points on the tube axis (centre line, not the
  tube surface). The local pixel ids of a straw increase from +x to −x (horizontal banks) and from −y to
  +y (vertical banks), so the first pixel end is the +x / −y end. The same ends are called 'A' / 'B' in
  `REFERENCE_BANKS_DATA`.
- **Corners:** for each bank, 'A' and 'B' each list 4 points: [front layer top tube, back layer top tube,
  front layer bottom tube, back layer bottom tube] ("top/bottom" = lowest/highest tube id).
- **Bank distance** (drawing SI-7615-463, `BcsBanks.cc` of G4GeoLoki): distance from the sample to the centre of the
  bank's front face, measured in the section plane perpendicular to the tubes, along the bank position angle.
  The front face centre is the centre of the front-layer tube centres, moved one tube radius (12.7 mm) towards
  the sample along the bank normal. The panel offset (along the tubes: +100 / −15 / −100 / +50 mm for banks 5–8)
  comes on top, so the 3D distance is √(distance² + offset²).

## Package contents

Modules of `loki_geometry` (in `src/loki_geometry/`):

| Module | What it is |
| --- | --- |
| `data/ESS.NSS.D03.LOKI.OB.DETECTOR_PIPE_CENTRES.TANK_WINDOW_CS.txt` (package data) | SAM-606 survey of the tube ends (the default survey): 32 points (2 tubes × 2 ends for banks 1–8) as `label, X, Y, Z` [mm] in the TANK WINDOW coordinate system (X along the beam, Y left, Z up). The points are on the tube axis at the physical tube ends, including ~15 mm of end electronics on each end. Bank 0 was not surveyed. |
| `constants` | The geometry constants shared by all modules (single source): bank layout, straw and tube grid dimensions, nominal bank placement (distances, position and face angles, offsets) and the derived tilt angles. |
| `survey_files` | What is specific to each known survey file, one `SurveyFile` entry per file: its tube label map (label → bank, tube id), and optionally its coordinate frame, a checksum of its content, fixes (e.g. the SAM-606 bank 5 fix) and notes. Unknown files, and files that do not match their checksum, are rejected. See [A new survey](#a-new-survey). |
| `survey` (`loki-survey`) | Converts a survey file into calibration data, using the file's `survey_files` entry: fixes, frame conversion, grouping into tubes, sanity checks, trimming of the end electronics. |
| `calibration` (`loki-calibrate`) | The calibration: calibration data → 8 corner positions per bank, quality report, comparison with the intended geometry, output in the `REFERENCE_BANKS_DATA` format. |
| `reference_data` | `REFERENCE_BANKS_DATA`: the intended corner positions (exact CAD values) and a converter to the calibration data format. This is the "intended" geometry of the comparisons. |
| `geant4_example_data` | Calibration data from the Geant4 geometry (first and last pixel centres of the centre straw of two front-layer tubes per bank, all 9 banks), generated with `geant4`. Used to test the calibration. |
| `geant4` (`loki-geant4-validate`) | Python port of the Geant4 pixel centre calculation (`AimHelper::getPixelCentreCoordinates` of [G4GeoLoki](https://github.com/ess-dg/dgcode_projects/tree/main/LOKI/G4GeoLoki)). It matches the C++ code to 1e-12 mm for all pixels (checked once with 1, 2, 3, 256 and 512 pixels per straw), and the `sb_loki_info` values hard-coded in the tests to 1e-6 mm. |
| `bank_pixels` | Pixel geometry core: from the 8 corners of a bank to the detector numbers and the pixel centre offsets of all its pixels (tube grid from vector steps between the corners, straw layout of Geant4, mirrored straw pattern for the upside-down banks 1, 2, 5, 6 derived from the corners). Inconsistent corners (> 0.1 mm) and a straw pattern that does not match the bank are rejected. |
| `nexus_json` (`loki-pixel-json`) | Writes the pixel geometry as NeXus-JSON nodes (the geometry parts of the `loki_detector_<id>` groups, plus `detector_carriage_zero` for the rear bank), ready to be copied into the instrument JSON file. Uses `REFERENCE_BANKS_DATA` for now. |
| `verify_json` (`loki-verify-json`) | Compares the pixel positions of a NeXus-JSON file (the output above, or a full file like `loki.json`) with the Geant4 geometry. Python counterpart of `scripts/verifyJSON` of G4GeoLoki. |
| `nominal_corners` (`loki-nominal-corners`) | Rebuilds the intended corner positions from the nominal bank distances, angles and offsets, and compares them with `REFERENCE_BANKS_DATA`. |

Repository layout:
- `src/loki_geometry/`: the package; `src/loki_geometry/data/` holds the survey files.
- `tests/`: the pytest tests (pixel geometry, JSON export and verification, Geant4 port, survey and calibration,
  constants, command-line tools and this README).
- `pyproject.toml`: packaging (setuptools), command-line tools, pytest settings.
- `*_local/` (e.g. `sources_local/`): local material that is not version controlled, like documents, drawings, raw survey
  files and large JSON files. The package and the tests never read it; the values of the current instrument file
  (`loki.json`) that the tests compare with are hard-coded in `tests/test_pixel_details.py`.

## From the measured points to the 8 corner positions

```
loki-calibrate --data measured --verbose --compare-to-intended --print
```

The survey is the default SAM-606 file; another known file is given with `--survey-file` (see [A new survey](#a-new-survey)).
Steps 1–4 use the file's entry in `survey_files` (here `SAM606`):

1. **Read the survey** (`survey.read_measured_points`) and look up its entry (`survey_files.survey_file_info`: by file
   name, and the content must match the entry's checksum). `label, X, Y, Z` lines.
   SAM-606 labels: `C1`/`C2` = front/middle carriage, `U/D/L/R` = up/down/left/right bank, `P<n>` = the surveyors' pipe
   number, `I`/`II` = the two ends of the tube.
2. **Apply the fixes of the file**. SAM-606 has one, `sam606_fix_bank5_mislabelled_tube`: `C1UP8I` and `C1UP8II` are on two
   different tubes (UP9 and UP8). Assuming parallel tubes, the missing end of tube UP9 is rebuilt as
   `C1UP8I + (C1UP3II − C1UP3I)`, and `C1UP8II` is dropped. A check confirms that the dropped point is one tube pitch
   (28.4 mm) from the rebuilt one.
3. **Convert the frame** (the entry's `frame`; SAM-606: `tank_window_frame`): x = Y, y = Z, z = X + D, where
   D = `survey.SAMPLE_TO_TANK_WINDOW_DISTANCE` is the sample-to-window distance. **D is provisional (100 mm)**; override it
   with `--sample-to-window-distance`. D shifts every bank equally along z and doesn't affect any of the quality numbers.
4. **Identify the tubes** (the entry's `label_map`, `survey.group_points_into_tubes`): each label is mapped to (bank, local
   tube id). The surveyors' P numbers do not follow our numbering. `C1DP31` is really tube 18, for example. Every point
   must belong to a mapped tube and every mapped tube must have both ends. The first pixel end of each tube is chosen from
   the coordinates (+x / −y).
5. **Sanity checks** (with `--verbose`): the two tubes of a bank must be parallel; their distance divided by 28.4 mm must equal
   the tube id difference; each tube's position across the bank must match its assigned id in the nominal geometry.
6. **Remove the end electronics** (`find_effective_tube_ends`): the measured tubes are ~30 mm longer than the effective straw
   length. Both ends are moved inwards by half the excess, along the line through the two points.
   The same function stretches Geant4 end pixel centres to the straw ends.
7. **Build the bank** (`calibration.Bank`, full description in the module docstring):
   - average direction along the tubes
   - least-squares plane through the 4 points (numpy SVD)
   - orthonormal basis: along the tubes, across the tubes in the plane, normal
   - tube 0 position averaged from both tubes' midpoints; midpoints don't depend on symmetric errors of the ends
   - front layer corners from the effective straw length and the 28.40 mm tube pitch
   - back layer corners from +78.78 mm along the normal and ±18.84 mm in the plane
   - input checks (error): the two tube ids must differ, both tubes must point the same way, and the normal must
     point away from the sample (swapped first / last pixel ends would put the back layers in front of the bank)
8. **Quality report** (`--verbose`, `Detectors.print_quality_report`), per bank:
   - plane fit residual (with 4 points on 2 lines all 4 residuals are equal; it measures the twist of the two tube axes)
   - angle between the two tubes
   - spacing in tube pitches
   - disagreement of the two tubes' predictions of tube 0
   - the input tube midpoints recomputed from the final corners (end-to-end check)
   - tilt angle, front face distance and front face centre shift vs the intended geometry
   - the placement in the drawing's terms (section distance, position angle, offset along the tubes)
9. **Output** (`--print`, `Detectors.formatted_data`): the corners as straw ends in the `REFERENCE_BANKS_DATA` format,
   the input of `bank_pixels` / `nexus_json`. Bank 0 is missing because it was not measured, so take it
   from `REFERENCE_BANKS_DATA` or measure it. (Feeding the measured corners into `loki-pixel-json` is not wired up yet.)

## A new survey

A new survey file needs its own entry in `src/loki_geometry/survey_files.py`: the surveyors' labels, possible mistakes in
the file and its coordinate frame are specific to each file. A file without an entry is rejected with an error.

The smallest entry has only the file name and the label map:

```python
NEW_SURVEY = SurveyFile(
    "NEW.txt",
    {"C1UP3": (5, 25), "C1UP8": (5, 19), ...},  # tube label (without I/II): (bank id, local tube id)
)
SURVEY_FILES = {entry.file_name: entry for entry in (SAM606, NEW_SURVEY)}
```

**Without a `frame`, the X, Y, Z values of the file are used as they are: they must already be in the NeXus frame (x left,
y up, z along the beam, in mm) and RELATIVE TO THE SAMPLE POSITION.** A sample-to-window distance does not apply to such
a file (`--sample-to-window-distance` is an error). If the survey is in another coordinate system, give a `frame`.

1. **The file.** Put it in `src/loki_geometry/data/` with a new name ending in `.txt` (installed as package data). Format:
   one `label, X, Y, Z` line per point, in mm. The label is the surveyors' tube label followed by `I` or `II` for the two
   ends. Each point must be on the tube axis at the tube end. Every bank needs exactly 2 tubes of the FRONT layer
   (best: far apart), both ends of each. Banks that are not in the file are simply missing from the result.
2. **The entry.** Add a `SurveyFile` next to `SAM606` and add it to the tuple of `SURVEY_FILES`. Required:
   - `file_name`: the name of the file.
   - `label_map`: {tube label without `I`/`II`: (bank id, local tube id)} (see [Conventions](#conventions)). To find the
     ids, start with guessed ids and run step 3: the check "Tube id estimated from the nominal geometry" prints each
     tube's id estimated from its position alone.

   Optional:
   - `frame`: a function `frame(point, sample_to_window_distance)` that converts a point of the file to the NeXus frame
     relative to the sample (default: none, see above). For a file in the TANK WINDOW coordinate system of the survey
     (X along the beam, Y left, Z up, origin at the tank window) use `tank_window_frame`; for any other system write a
     new function, a proper rotation (determinant +1) plus a shift.
   - `checksum`: the checksum of the file content, from
     `python3 -c "from loki_geometry.survey_files import content_checksum; print(content_checksum('src/loki_geometry/data/NEW.txt'))"`.
     With a checksum, a file whose content differs from the entry (e.g. a corrected version under the same name) is
     rejected. Without one (default), a changed file is used with the entry unnoticed, so adding it is recommended.
   - `description`: where the file comes from, printed with the checks.
   - `fixes`: a function for each mistake in the file (a mislabelled point, a missing end, …), called as
     `fix(points, verbose)` with `{label: np.array([X, Y, Z])}` in the file's frame, returning the corrected points; it
     should check its assumption and print a `WARNING` if it fails. See `sam606_fix_bank5_mislabelled_tube`.
   - `bank_notes`: notes printed with the checks of a bank (e.g. a tube made parallel by a fix).

   To make the new survey the default, set `DEFAULT_SURVEY_FILE` to its path.
3. **Check it.** `loki-survey --survey-file src/loki_geometry/data/NEW.txt` prints the checks: parallel tubes, the tube
   spacing in pitches, the tube ids estimated from the nominal geometry, and the removed end-electronics length per tube.
   There should be no `WARNING`.
4. **Calibrate.** `loki-calibrate --survey-file src/loki_geometry/data/NEW.txt --verbose --compare-to-intended --print`
   (see the steps above for the quality report).
5. **Tests.** Add tests for the entry to `tests/test_survey_files.py`; the SAM-606 tests there (label map against the
   nominal tube positions, the fix, the excess lengths) are the template. `test_a_minimal_entry_uses_the_coordinates_as_given`
   and `test_a_new_survey_file_with_its_own_entry` go through these steps with synthetic files.

## Pixel geometry for the instrument JSON file

The people producing the instrument JSON file need, for each bank, the bank translation (from the sample to the
bank position) and the detector numbers with the local pixel offsets. `loki-pixel-json` writes exactly these, as
NeXus-JSON nodes in the structure of `loki.json`:

```
loki-pixel-json -o loki_detector_pixels.json               # all banks, 256 pixels per straw
loki-pixel-json -o test.json --pixels-per-straw 2 --banks 1 2
loki-pixel-json -o base.json --pixel-reference pixel_base  # offsets to the pixel bases
loki-verify-json loki_detector_pixels.json               # compare with Geant4
loki-verify-json path/to/loki.json                       # works on the full instrument file too
loki-verify-json path/to/loki.json --ignore-pixel-shape  # offsets only, without the pixel shape
```

- Output: `{"info": {...}, "loki_detector_0": <group>, "detector_carriage_zero": <translation>, "loki_detector_1": <group>, …, "loki_detector_8": <group>}`
  (`detector_carriage_zero` only if bank 0 is written).
  Each group holds `detector_number`, `pixel_shape`, `x/y/z_pixel_offset` [m], `transformations` (banks 1–8: one
  translation `trans_<id+1>`; bank 0 has none, see below) and `depends_on`.
- **Bank position** (the bank translation, the origin of the pixel offsets) = the bank corner `A[0]`: the first pixel end of
  tube 0 (front layer, top tube; leftmost tube in vertical banks) on the tube axis, as in the current `loki.json`.
  It is not a pixel position: the first pixel centre is half a pixel further along the straw, and in straw 0 also 7.75 mm to
  the side.
- **Pixel reference point** (`--pixel-reference`), i.e. what the pixel offsets point to:
  - `pixel_centre` (default): the pixel centres. The pixel shape is centred on them.
  - `pixel_base`: the centre of the pixel's first face (the face towards the first pixel end). The pixel shape starts there
    and extends one pixel length along the straw.

  Both describe the same pixel volumes; `info.pixel_reference` in the output records the choice.
- Bank 0 is on the moving carriage: its group depends on `/entry/instrument/detector_carriage/value`, and its translation is the
  `detector_carriage_zero` dataset of `/entry/instrument/detector_carriage/transformations`.
- `pixel_shape` is a cylinder of the straw radius (3.75 mm) and the pixel length along the straw. Its vertices are the centre of
  the first face, a point on the edge of that face, and the centre of the second face.
- **Known issue in the current `loki.json`:** its offsets point to the pixel centres, but its pixel shape still starts
  at the offset point, as if the offsets pointed to the pixel bases. So its pixel
  volumes are shifted by half a pixel along the straws (0.49 / 0.98 / 1.17 mm for the 500 / 1000 / 1200 mm straws with 512 pixels per straw). `loki-verify-json`
  shows this unless `--ignore-pixel-shape` is given.
- `loki-verify-json` computes the pixel centres as translation + offset + centre of the pixel shape cylinder, so it
  checks either convention and catches an inconsistent shape.
- Pixel ids: `bank first id + (tube · 7 + straw) · N + pixel`, with tube = layer · rows + row, as in Geant4 and the ICD.
- Checks (all in `tests/`):
  - With corners taken from the Geant4 geometry, all 1,605,632 pixels agree with the Geant4 port within 1e-6 mm.
  - With `REFERENCE_BANKS_DATA` they agree within 1 mm, which is the rounding of the CAD values in Geant4.
  - With 512 pixels per straw, the current `loki.json` offsets are reproduced within 0.011 mm, which is its 5-decimal rounding
    (sample values of `loki.json` hard-coded in the tests); the ids, translations and group structure are the same.
  - Both pixel reference conventions give the same pixel centres, and an inconsistent pixel shape is detected.

## Other commands

```
# Self-test with the intended geometry: the corners must reproduce REFERENCE_BANKS_DATA (within 0.07 mm, the
# rounding of the CAD values; exit code 1 otherwise). The quality report compares the geometry with itself (0.00).
loki-calibrate --data reference --verbose

# Test with the Geant4 geometry: reproduces the Geant4 tilts, section distances and offsets exactly
loki-calibrate --data example --verbose

# Measured data with another sample-to-window distance
loki-calibrate --sample-to-window-distance 119 --verbose

# Only the conversion of the survey file (prints the checks and the calibration data)
loki-survey
loki-survey --survey-file path/to/NEW.txt   # another survey file (it needs a survey_files entry)

# Validate the Geant4 port against sb_loki_info values (exit code 1 if it deviates)
loki-geant4-validate

# Rebuild the intended corners from distances / angles / offsets and compare with REFERENCE_BANKS_DATA
loki-nominal-corners          # all banks, or only some: loki-nominal-corners 1 2

# Run the tests
python3 -m pytest
```

Using the pieces from Python:

```python
from loki_geometry import REFERENCE_BANKS_DATA, compute_instrument_pixels
from loki_geometry.survey import get_measured_calibration_data
from loki_geometry.calibration import Detectors
from loki_geometry.geant4 import pixel_centre, pixel_id

measured = Detectors(
    get_measured_calibration_data(sample_to_window_distance=100.0),
    use_pixel_centre_position=True,
    verbose=True,
)
measured.compare_to(measured.intended, label="intended")  # corner shifts vs REFERENCE_BANKS_DATA
corners = measured.formatted_data()  # REFERENCE_BANKS_DATA format (straw ends)
x, y, z = pixel_centre(
    pixel_id(bank_id=1, tube_id=13, straw_id=3, pixel_in_straw=0)
)  # Geant4 pixel centre [mm]
pixels = compute_instrument_pixels(
    REFERENCE_BANKS_DATA
)  # {bank_id: BankPixels}: ids, offsets, positions [mm]
```

## Open points / known issues

- **Sample-to-window distance D:** 119 mm (flange face) or 108 mm (sapphire window) from drawing SI-7615-463, depending on
  which surface is the survey origin; provisional value 100 mm.
- **End electronics:** assumed equally long on both ends. This matters for bank 7, where the two measured tubes are shifted
  2.8 mm relative to each other along the tubes, and which comes out 8 mm off its intended offset.
- **Bank 0:** not surveyed.
- **Geant4 geometry** ([G4GeoLoki](https://github.com/ess-dg/dgcode_projects/tree/main/LOKI/G4GeoLoki)): it uses rounded CAD values, so `REFERENCE_BANKS_DATA` differs slightly from it (up to 0.56 mm and
  0.05° in bank 8). The geometry building (`pycpp_GeoBCSBanks/geometry_module.cc`) and the pixel position code
  (`libsrc/AimHelper.cc`) both take all positions from `BcsBanks.cc`, but apply the rotations separately.

## Development

```
pip install -e ".[dev]"     # pytest, ruff
python3 -m pytest           # tests
ruff format .               # formatting (line length 100, settings in pyproject.toml)
ruff check .                # lint: pycodestyle, pyflakes, isort, bugbear, pyupgrade, numpy rules
git config blame.ignoreRevsFile .git-blame-ignore-revs   # hide formatting-only commits in git blame
```

## License

BSD 3-Clause License, see [LICENSE](LICENSE).
