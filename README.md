# loki-geometry – LOKI detector bank geometry

The actual pixel positions of the LOKI detector for its NeXus files. The detector banks were surveyed (e.g. SAM-606);
loki-geometry turns the surveyed tube positions into the 8 corner positions of each bank (the bank calibration), and the
corners into the pixel geometry of all 1,605,632 pixels (with 256 pixels per straw): the bank positions, detector numbers
and pixel offsets, written as NeXus-JSON nodes. These replace the geometry of the detector banks in the instrument's
file-writer JSON template, which describes the nominal CAD geometry so far. See
[Updating the instrument JSON file](#updating-the-instrument-json-file).

The main tools:
- `loki-survey`, `loki-calibrate`: the bank calibration from a survey, with checks and a quality report.
- `loki-pixel-json`: the pixel geometry of a calibration as NeXus-JSON nodes (default: the measured geometry), and
  `loki-verify-json` to check a JSON file.

Additions, also used to test the calibration:
- `loki-info`: positions and ids of the banks, tubes, straws and pixels for a calibration.
- The bank calibration file for the Geant4 geometry (`loki-calibrate --geant4-calibration-file`).

The pixel positions are checked against the LOKI Geant4 geometry, [G4GeoLoki](https://github.com/ess-dg/dgcode_projects/tree/main/LOKI/G4GeoLoki),
through a Python port of its pixel position code (`geant4`) and hard-coded Geant4 values in the tests.

The Python package `loki_geometry` (distribution name `loki-geometry`) needs only Python ≥ 3.9 and numpy.
The tests need pytest.

## Installation

```
pip install .                          # from a clone of the repository
pip install -e ".[test]"               # editable install for development, with pytest
pip install "git+https://github.com/MilanKlausz/loki-geometry.git"   # directly from the git repository
```

This installs the package and the command-line tools `loki-info`, `loki-calibrate`, `loki-survey`, `loki-pixel-json`,
`loki-verify-json`, `loki-geant4-validate` and `loki-nominal-corners`. The same tools can be run without installing
as `python3 -m loki_geometry.<module>` with `PYTHONPATH=src`. Run the tests with `python3 -m pytest` from the
repository root; no installation is needed for that.

## Updating the instrument JSON file

1. **The survey.** The survey file and its entry in `survey_files` (with a `calibration_name`); see
   [A new survey](#a-new-survey). For the SAM-606 survey (`2026-September-SAM-606`, the default) this is done.
2. **Check and calibrate.** `loki-survey` (the survey checks) and
   `loki-calibrate --verbose --compare-to-intended` (the calibration and its quality report, see
   [From the measured points to the 8 corner positions](#from-the-measured-points-to-the-8-corner-positions)).
3. **Write the pixel geometry**, with the pixel number of the instrument file (`loki.json`: 512 per straw):
   `loki-pixel-json -o loki_detector_pixels.json --pixels-per-straw 512` (the default calibration, i.e. the measured
   geometry; another one with `-c NAME`). The `info` of the output records the calibration and, per bank, the source of
   its corners (the survey, or `REFERENCE_BANKS_DATA` for a bank that is not in it) and the package version.
4. **Check it.** `loki-verify-json loki_detector_pixels.json` compares it with the pixel positions of its calibration;
   `loki-verify-json loki_detector_pixels.json -c nominal-geant4-geometry` shows how far the measured geometry is from
   the nominal one (tens of mm for some banks).
   With the Geant4 geometry (in a dgcode_projects environment, the file is read like the instrument file):
   `sb_g4geoloki_verifyjson --json loki_detector_pixels.json -rear 4.98565` compares it with the Geant4 pixel centres
   of the calibration recorded in the file (`-rear`: the distance of the rear bank front face of the file, at its
   carriage zero position, `loki-info -b 0`: 4.98565 m for SAM-606, 4.9998 m for the CAD geometry; Geant4 places the
   rear bank at the rear detector distance).
5. **Copy it into the template** (in the repository of the instrument's JSON file). The output has the tree of the
   instrument file, with only these nodes in `/entry/instrument`: in each `loki_detector_<id>` group,
   replace the geometry children (`detector_number`, `pixel_shape`, `x/y/z_pixel_offset`, `transformations`,
   `depends_on`) by those of the output, and the `detector_carriage_zero` dataset of
   `/entry/instrument/detector_carriage/transformations` by the one of the output (the rear bank). The other children
   of the groups (e.g. `detector_events`) and of the detector carriage (its `value` is the motor stream) stay. This also fixes the half-pixel shift of the pixel shape of the current
   `loki.json` (see [Pixel geometry for the instrument JSON file](#pixel-geometry-for-the-instrument-json-file)).
6. **Check the updated template.** It has no `info` (no recorded calibration), so give the calibration:
   `loki-verify-json path/to/loki.json -c 2026-September-SAM-606` and, with the Geant4 geometry,
   `sb_g4geoloki_verifyjson --json path/to/loki.json --bankCalibration 2026-September-SAM-606 -rear 4.98565`. Both read
   the banks by name and follow their `depends_on` chains (bank 0 through the detector carriage), so a broken chain, a
   transformation that is not a translation or a wrong unit is an error or shows as a deviation.

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
- **Ids:** pixel id = first pixel id of the bank + (tube × 7 + straw) × pixels per straw + pixel in the straw, 0-based
  as in Geant4 and the analysis programs; the detector number (NeXus `detector_number`, ICD) is pixel id + 1. Tube id =
  layer × (tubes per layer) + row, layer 0 = the front layer; straw 3 is on the tube axis; pixel 0 is at the first pixel
  end of the straw.
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
| `data/ESS.NSS.D03.LOKI.OB.DETECTOR_PIPE_CENTRES.TANK_WINDOW_CS.txt` (package data) | SAM-606 survey of the tube ends (the default survey): 36 points (2 tubes × 2 ends for each of the 9 banks; bank 0 measured later, SAM-606 Rev2, with the detector carriage at 5.0 m, i.e. the rear bank at 10 m) as `label, X, Y, Z` [mm] in the TANK WINDOW coordinate system (X along the beam, Y left, Z up). The points are on the tube axis at the physical tube ends, including the end electronics (the measured tubes are about 30 mm longer than the effective straw length). |
| `survey_files` | What is specific to each known survey file, one `SurveyFile` entry per file: its tube label map (label → bank, tube id), and optionally its coordinate frame, a checksum of its content, fixes (e.g. the SAM-606 bank 5 fix) and notes. Unknown files, and files that do not match their checksum, are rejected. See [A new survey](#a-new-survey). |
| `survey` (`loki-survey`) | Converts a survey file into calibration data, using the file's `survey_files` entry: fixes, frame conversion, grouping into tubes, sanity checks, trimming of the end electronics. |
| `calibration` (`loki-calibrate`) | The calibration: calibration data → 8 corner positions per bank, quality report, comparison with the intended geometry, output in the `REFERENCE_BANKS_DATA` format. |
| `bank_pixels` | Pixel geometry core: from the 8 corners of a bank to the detector numbers and the pixel offsets (pixel centres or bases) of all its pixels (tube grid from vector steps between the corners, straw layout of Geant4, mirrored straw pattern for the upside-down banks 1, 2, 5, 6 derived from the corners). Corners are rejected if they are not finite, not consistent with each other (> 0.1 mm), have other tube pitches than 28.40 / 27.00 mm (> 0.5 mm, e.g. corners of another bank), are in the wrong order (first / last pixel ends swapped give a left-handed frame, A[1] / A[2] swapped give wrong tube pitches) give a straw pattern that does not match the bank, or have another straw length than the nominal one of the bank (> 0.1 mm: e.g. untrimmed tube ends or pixel-centre corners). |
| `nexus_json` (`loki-pixel-json`) | The main output: the pixel geometry of a bank calibration (default: the measured geometry of the default survey) as NeXus-JSON nodes (the geometry parts of the `loki_detector_<id>` groups, plus `detector_carriage_zero` for the rear bank) in the instrument tree of the instrument JSON file, ready to be copied into it. |
| `verify_json` (`loki-verify-json`) | Compares the pixel positions of a NeXus-JSON file with the expected ones: those of the calibration recorded in the file (the output above), otherwise (e.g. the full file `loki.json`) those of the nominal Geant4 geometry, as `scripts/verifyJSON` of G4GeoLoki does. |
| `geometry_info` (`loki-info`) | Geometry information for a chosen bank calibration: the placement of each bank (front face centre, axes, corners), the tube and straw ends, the pixel centres, and the id conversions (pixel id, detector number, bank / tube / straw / pixel). See [Geometry information](#geometry-information). |
| `geant4_calibration` | Writes (and reads) the bank calibration file of the Geant4 geometry: per bank the front face centre and the three bank axes, selected in Geant4 with the geometry parameter `bank_calibration`. See [Bank calibration for the Geant4 geometry](#bank-calibration-for-the-geant4-geometry). |
| `reference_data` | `REFERENCE_BANKS_DATA`: the intended corner positions (exact CAD values) and a converter to the calibration data format. This is the "intended" geometry of the comparisons. |
| `nominal_corners` (`loki-nominal-corners`) | Rebuilds the intended corner positions from the nominal bank distances, angles and offsets, and compares them with `REFERENCE_BANKS_DATA`. |
| `geant4` (`loki-geant4-validate`) | Python port of the nominal pixel centre calculation of [G4GeoLoki](https://github.com/ess-dg/dgcode_projects/tree/main/LOKI/G4GeoLoki) (its original implementation, `AimHelper::getPixelCentreCoordinates`; the later implementation with one transform per bank, `BcsBanks::getBankTransform`, gives the same positions within 1e-9 mm). It matched the original C++ code to 1e-12 mm for all pixels (checked once with 1, 2, 3, 7, 256 and 512 pixels per straw), and the Geant4 values hard-coded in `geant4.py` (`sb_loki_info` output, used by `loki-geant4-validate`) and in `tests/test_geant4.py` (`sb_loki_info` and `LokiAimHelper` output) to 1e-6 mm. |
| `geant4_example_data` | Calibration data from the Geant4 geometry (first and last pixel centres of the centre straw of two front-layer tubes per bank, all 9 banks), generated with `geant4`. Used to test the calibration. |
| `constants` | The geometry constants shared by all modules (single source): bank layout, straw and tube grid dimensions, nominal bank placement (distances, position and face angles, offsets) and the derived tilt angles. |

Repository layout:
- `src/loki_geometry/`: the package; `src/loki_geometry/data/` holds the survey files.
- `tests/`: the pytest tests, one file per module (`tests/test_<module>.py`), plus `tests/test_cli_and_docs.py` for the
  command-line tools, this README and the packaging.
- `pyproject.toml`: packaging (setuptools), command-line tools, pytest settings.
- `*_local/` (e.g. `sources_local/`): local material that is not version controlled, like documents, drawings, raw survey
  files and large JSON files. The package and the tests never read it; the values of the current instrument file
  (`loki.json`) that the tests compare with are hard-coded in `tests/test_nexus_json.py`.

## Pixel geometry for the instrument JSON file

The people producing the instrument JSON file need, for each bank, the bank translation (from the sample to the
bank position) and the detector numbers with the local pixel offsets. `loki-pixel-json` writes exactly these, as
NeXus-JSON nodes in the structure of `loki.json`, for a bank calibration (`-c NAME`, as in
[Geometry information](#geometry-information)): by default the measured geometry of the default survey
(`2026-September-SAM-606`), or the intended CAD geometry (`nominal-cad-geometry`, the geometry of the current
`loki.json`) or the nominal Geant4 geometry (`nominal-geant4-geometry`).

```
loki-pixel-json -o loki_detector_pixels.json                       # all banks, the measured geometry, 256 pixels per straw
loki-pixel-json -o loki_detector_pixels.json --pixels-per-straw 512  # as many pixels as loki.json
loki-pixel-json -o cad.json -c nominal-cad-geometry                # the intended CAD geometry
loki-pixel-json -o test.json --pixels-per-straw 2 --banks 1 2
loki-pixel-json -o base.json --pixel-reference pixel_base          # offsets to the pixel bases
loki-pixel-json -o com.json --bank-origin centre_of_mass           # bank positions at the centres of mass
loki-pixel-json --visualize -o view.json --banks 1 2 3 4 5 6 7 8   # for NeXus-JSON geometry viewers
loki-verify-json loki_detector_pixels.json                         # against the positions of its calibration
loki-verify-json loki_detector_pixels.json -c nominal-geant4-geometry   # measured vs nominal Geant4 geometry
loki-verify-json path/to/loki.json                                 # the full instrument file, against nominal Geant4
loki-verify-json path/to/loki.json --ignore-pixel-shape            # offsets only, without the pixel shape
```

- Output: the instrument tree of the instrument file with only the detector geometry in it,
  `{"info": {...}, "children": [entry → instrument → [loki_detector_0, …, loki_detector_8, detector_carriage]]}`,
  so that it can be read like `loki.json` (e.g. by the Geant4 verification `sb_g4geoloki_verifyjson` and by
  NeXus-JSON viewers). `detector_carriage` (only if bank 0 is written) holds `transformations/detector_carriage_zero`
  and, as in `loki.json`, the carriage position `value` (0 here; in the instrument file it is the motor position) on
  which bank 0 depends. `info` records the calibration, the source of the corners of
  each bank (`bank_corner_sources`: the survey file with its checksum, or `REFERENCE_BANKS_DATA` for a bank that was not
  surveyed), the package version and the options.
  Each group holds `detector_number`, `pixel_shape`, `x/y/z_pixel_offset` [m], `transformations` (banks 1–8: one
  translation `trans_<id+1>`; bank 0 has none, see below) and `depends_on`.
- **Bank position** (`--bank-origin`), i.e. the bank translation, the origin of the pixel offsets. The absolute pixel
  positions are the same for both choices; only their split into translation + offsets differs:
  - `first_pixel_end` (default): the bank corner `A[0]`, the first pixel end of the centre straw of tube 0 (front layer,
    top tube; leftmost tube in vertical banks) on the tube axis, as in the current `loki.json`. It is not a pixel
    position: the first pixel centre of that straw is half a pixel further along it (the other straws of the tube are
    7.75 mm to the side).
  - `centre_of_mass`: the mean of all pixel centres of the bank, i.e. the centre of the tube grid, halfway along the
    straws. It does not depend on the number of pixels per straw.

  `info.bank_origin` in the output records the choice. For bank 0, `detector_carriage_zero` is this position.
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
- `loki-verify-json` computes the pixel centres as bank position + offset + centre of the pixel shape cylinder, so it
  checks either convention and catches an inconsistent shape. The bank position is the sum of the translations of the
  `depends_on` chain of the bank group (bank 0: the carriage position, 0 if it is a stream, and
  `detector_carriage_zero`); a broken chain, a transformation that is not a translation along a unit vector, or a
  length without a length unit is an error. It compares the pixel centres with the calibration recorded in the
  file (`info.calibration`), otherwise with the nominal Geant4 geometry; `-c NAME` selects any calibration.
- `--decimals N` rounds every value to 10^-N m. A pixel position adds several rounded values (translation, offset,
  pixel shape), so with the coarsest `--decimals 4` (0.1 mm) it can be off by ~0.11 mm. `info.decimals` records it.
- `--visualize`: 4 pixels per straw by default, for NeXus-JSON geometry viewers (e.g.
  [jsondisplay](https://github.com/chribe/jsondisplay), which draws the pixel offsets as points). Bank 0 is placed by
  its own copy of `detector_carriage_zero` instead of the detector carriage (same positions), because such viewers
  follow only the transformations inside a group. Not for the instrument file.
- Pixel ids: `bank first id + (tube · 7 + straw) · N + pixel`, with tube = layer · rows + row, as in Geant4 and the ICD.
- Checks (all in `tests/`):
  - With the default calibration (the SAM-606 survey), the pixel positions read back from the file equal the pixel
    centres of the Geant4 geometry with the same calibration (hard-coded G4GeoLoki values) within 1e-4 mm (banks 1–8;
    bank 0, which the file has at its carriage zero position, agrees within 1e-6 mm at the same rear distance).
  - With corners taken from the Geant4 geometry, all 1,605,632 pixels agree with the Geant4 port within 1e-6 mm.
  - With `REFERENCE_BANKS_DATA` they agree within 1 mm, which is the rounding of the CAD values in Geant4.
  - With 512 pixels per straw, the current `loki.json` offsets are reproduced within 0.011 mm, which is its 5-decimal rounding
    (sample values of `loki.json` hard-coded in the tests); the ids, translations and group structure are the same.
  - Both pixel reference conventions give the same pixel centres, and an inconsistent pixel shape is detected.
  - Both bank origins give the same absolute pixel positions (also after writing and reading the JSON file).

## From the measured points to the 8 corner positions

```
loki-calibrate --data measured --verbose --compare-to-intended --print
```

The survey is the default SAM-606 file; another known file is given with `--survey-file` (see [A new survey](#a-new-survey)).
Steps 1–4 use the file's entry in `survey_files` (here `SAM606`):

1. **Read the survey** (`survey.read_measured_points`) and look up its entry (`survey_files.survey_file_info`: by file
   name, and, if the entry has a checksum, the content must match it). `label, X, Y, Z` lines.
   SAM-606 labels: `C1`/`C2` = front/middle carriage, `U/D/L/R` = up/down/left/right bank, `P<n>` = the surveyors' pipe
   number, `I`/`II` = the two ends of the tube; `C3P<n>` = the rear bank (bank 0), `P<n>` = row n − 1.
2. **Apply the fixes of the file**. SAM-606 has one, `sam606_fix_bank5_mislabelled_tube`: `C1UP8I` and `C1UP8II` are on two
   different tubes (UP9 and UP8). Assuming parallel tubes, the missing end of tube UP9 is rebuilt as
   `C1UP8I + (C1UP3II − C1UP3I)`, and `C1UP8II` is dropped. A check confirms that the dropped point is one tube pitch
   (28.4 mm) from the rebuilt one.
3. **Convert to the sample frame** (the entry's `frame`): the survey can be in any coordinate system, as long as its
   entry defines the conversion to the NeXus frame with the origin at the sample (without a `frame` the values are used
   as given). Example: SAM-606 is in the TANK WINDOW coordinate system (X along the beam, Y left, Z up, origin at the
   tank window); `tank_window_frame` gives x = Y, y = Z, z = X + D, with D = `SAMPLE_TO_TANK_WINDOW_DISTANCE` the
   distance from the sample to the window. Such a shift along z moves every bank equally; it does not change the fit
   quality numbers or the tilts, only the distances and position angles compared with the intended geometry.
4. **Identify the tubes** (the entry's `label_map`, `survey.group_points_into_tubes`): each label is mapped to (bank, local
   tube id). The surveyors' P numbers do not follow our numbering. `C1DP31` is really tube 18, for example. Every point
   must belong to a mapped tube and every mapped tube must have both ends; the label map must use known banks and front
   layer tube ids, each tube once; each tube must be at least half a straw length long (errors otherwise). The first
   pixel end of each tube is chosen from the coordinates (+x / −y).
5. **Sanity checks** (always run; all results are printed by `loki-survey` and `loki-calibrate --verbose`, otherwise only the failing ones as
   `WARNING` lines): the two tubes of a bank must be parallel; their distance divided by 28.4 mm must equal the tube id
   difference; each tube's position across the bank must match its assigned id in the nominal geometry; the length
   removed in step 6 must be in the `excess_length_range` of the entry (default 20–40 mm,
   `survey_files.DEFAULT_EXCESS_LENGTH_RANGE`, for points at the physical tube ends).
   `loki-pixel-json` and `loki-info` run these checks too when they use a survey calibration, and report failed ones
   on stderr.
6. **Adjust the tube ends to the straw length** (`find_effective_tube_ends`): if the distance of the two measured points
   of a tube differs from the effective straw length of the bank, both points are moved symmetrically along the line
   through them (closer or farther) until it matches, i.e. the difference is assumed to be split equally between the
   two ends. This allows measuring points that are not the ends of the effective straw length, e.g. when the end
   electronics make those inaccessible (the SAM-606 points are at the physical tube ends, ~30 mm longer). The same
   function stretches Geant4 end pixel centres to the straw ends.
7. **Build the bank** (`calibration.Bank`, full description in the module docstring):
   - average direction along the tubes
   - least-squares plane through the 4 points (numpy SVD)
   - orthonormal basis: along the tubes, across the tubes in the plane, normal
   - tube 0 position averaged from both tubes' midpoints; midpoints don't depend on symmetric errors of the ends
   - front layer corners from the effective straw length and the 28.40 mm tube pitch
   - back layer corners from +78.78 mm along the normal and ±18.84 mm in the plane
   - input checks (error): exactly 2 tubes per bank, the two tube ids must differ, both tubes must point the same way,
     and the normal must point away from the sample (swapped first / last pixel ends, or a survey frame conversion that
     mirrors the coordinates, would put the back layers in front of the bank)
8. **Quality report** (`--verbose`, `Detectors.print_quality_report`), per bank:
   - plane fit residual (with 4 points on 2 lines all 4 residuals are nearly equal in magnitude, with alternating signs; it
     measures the twist of the two tube axes)
   - angle between the two tubes
   - spacing in tube pitches
   - disagreement of the two tubes' predictions of tube 0
   - the input tube midpoints recomputed from the final corners (end-to-end check)
   - tilt angle (in the section plane) vs the intended geometry, and the roll angle (rotation of the tubes in the bank
     plane)
   - front face distance and front face centre shift vs the intended geometry
   - the placement in the drawing's terms (section distance, position angle, offset along the tubes)
9. **Output** (`--print`, `Detectors.formatted_data`): the corners as straw ends in the `REFERENCE_BANKS_DATA` format,
   the input of `bank_pixels`. Only the banks in the survey are in the output.
10. **Pixel geometry**: `loki-pixel-json` and `loki-info` use these corners when the calibration is selected by its
    name (the `calibration_name` of the survey entry; the default survey's calibration is their default); banks that
    are not in the survey are taken from `REFERENCE_BANKS_DATA`. See
    [Updating the instrument JSON file](#updating-the-instrument-json-file).
11. **Geant4 calibration file** (`--geant4-calibration-file`), see
    [Bank calibration for the Geant4 geometry](#bank-calibration-for-the-geant4-geometry).

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
y up, z along the beam, in mm) and RELATIVE TO THE SAMPLE POSITION.** If the survey is in another coordinate system, give
a `frame`.

The new file and entry are part of the package, so the installed package must include them: reinstall it with
`pip install .` after adding them (and after every later change), or install it once with `pip install -e .`
(editable), so that changes take effect immediately. Without installing, run the commands as
`PYTHONPATH=src python3 -m loki_geometry.<module> …` from the repository root.

1. **The file.** Put it in `src/loki_geometry/data/` with a new name ending in `.txt` (installed as package data). Format:
   one `label, X, Y, Z` line per point, in mm. The label is the surveyors' tube label followed by `I` or `II` for the two
   ends. Each point must be on the tube axis at the tube end. Every bank needs exactly 2 tubes of the FRONT layer
   (best: far apart), both ends of each. Banks that are not in the file are simply missing from the result.
2. **The entry.** Add a `SurveyFile` next to `SAM606` and add it to the tuple of `SURVEY_FILES`. Required:
   - `file_name`: the name of the file.
   - `label_map`: {tube label without `I`/`II`: (bank id, local tube id)} (see [Conventions](#conventions)). To find the
     ids, start with guessed (but different) ids and run step 3: the check "Tube id estimated from the
     nominal geometry" prints each tube's id estimated from its position alone.

   Optional:
   - `frame`: a function `frame(point)` that converts a point of the file (`np.array([X, Y, Z])`) to the NeXus frame
     relative to the sample (default: none, see above). Everything the conversion needs, like the position of the
     survey's origin relative to the sample, is part of the function. For a file in the TANK WINDOW coordinate system
     of the survey (X along the beam, Y left, Z up, origin at the tank window) use `tank_window_frame`, which uses
     `SAMPLE_TO_TANK_WINDOW_DISTANCE`; for any other system write a new function, a proper rotation (determinant +1)
     plus a shift.
   - `checksum`: the checksum of the file content, from
     `python3 -c "from loki_geometry.survey_files import content_checksum; print(content_checksum('src/loki_geometry/data/NEW.txt'))"`.
     With a checksum, a file whose content differs from the entry (e.g. a corrected version under the same name) is
     rejected. Without one (default), a changed file is used with the entry unnoticed, so adding it is recommended.
   - `description`: where the file comes from, printed with the checks.
   - `fixes`: a function for each mistake in the file (a mislabelled point, a missing end, …), called as
     `fix(points, verbose)` with `{label: np.array([X, Y, Z])}` in the file's frame, returning the corrected points; it
     should check its assumption and print a `WARNING` if it fails. See `sam606_fix_bank5_mislabelled_tube`.
   - `bank_notes`: notes printed with the checks of a bank (e.g. a tube made parallel by a fix).
   - `carriage_position`: the position of the detector carriage during the survey of the rear bank (0), in mm from
     its zero position (the position of `detector_carriage_zero`, the 5 m position of the CAD geometry). The bank 0
     points are moved back along the beam by it. SAM-606: 5000 mm (the rear bank at 10 m). Default 0; an error if the
     label map has no bank 0.
   - `excess_length_range`: (min, max) of the length removed from each tube by the trimming (step 6 of the chain above)
     [mm]; default 20–40 mm, for points at the physical tube ends.
   - `calibration_name`: the name of the calibration made from this file. It is needed to use the survey in
     `loki-pixel-json` and `loki-info` (`-c NAME`), and it is the default name of the Geant4 calibration file. A name
     that says when and where the survey was made helps, like `2026-September-SAM-606`.

   To make the new survey the default (of `loki-calibrate`, `loki-survey`, and, through its `calibration_name`, of
   `loki-pixel-json` and `loki-info`), set `DEFAULT_SURVEY_FILE = NEW_SURVEY.path`. The tests use the SAM-606 file
   explicitly; the tests of the defaults then need the new values.
3. **Check it.** `loki-survey --survey-file NEW.txt` prints the checks: parallel tubes, the tube spacing in
   pitches, the tube ids estimated from the nominal geometry, and the removed end-electronics length per tube (a packaged
   file can be given by its name; any other file by its path). There should be no `WARNING`.
4. **Calibrate.** `loki-calibrate --survey-file NEW.txt --verbose --compare-to-intended --print` (see the steps above for
   the quality report). The `Warning: Bank … differs from intended …` lines of the calibration are expected: they report
   the measured deviations of the tilts (> 0.1°) and distances (> 1 mm) from the intended geometry. A `WARNING` of the
   survey checks (step 3) is not expected.
5. **Pixel geometry.** `loki-pixel-json -c NEW-NAME -o loki_detector_pixels.json --pixels-per-straw 512` and
   `loki-verify-json loki_detector_pixels.json`, then copy it into the template (see
   [Updating the instrument JSON file](#updating-the-instrument-json-file)).
6. **Geant4.** `loki-calibrate --survey-file NEW.txt --geant4-calibration-file <dgcode_projects>/LOKI/G4GeoLoki/data/`
   writes the calibration file for the Geant4 geometry (see
   [Bank calibration for the Geant4 geometry](#bank-calibration-for-the-geant4-geometry)). Its bank volumes may then
   need a check for overlaps in Geant4.
7. **Tests.** Add tests for the entry to `tests/test_survey_files.py`; the SAM-606 tests there (label map against the
   nominal tube positions, the fix, the excess lengths) are the template. `test_a_minimal_entry_uses_the_coordinates_as_given`
   and `test_a_new_survey_file_with_its_own_entry` go through these steps with synthetic files.

## Geometry information

`loki-info` (module `geometry_info`, class `LokiGeometry`) gives the positions and ids of the banks, tubes, straws and
pixels for a bank calibration, selected by name (the same calibrations and positions as `loki-pixel-json`):

| Calibration | What it is |
| --- | --- |
| `nominal-cad-geometry` | The intended geometry: the corners of `REFERENCE_BANKS_DATA` (exact CAD values). |
| `nominal-geant4-geometry` | The nominal geometry of the Geant4 model (its built-in tables, via `geant4`); in Geant4 `bank_calibration=nominal-geant4-geometry`. It differs from the CAD values by up to ~1 mm. |
| a survey calibration | The `calibration_name` of a `survey_files` entry (e.g. `2026-September-SAM-606`); banks that are not in the survey are taken from `nominal-cad-geometry`. |

The default is the calibration of the default survey file (`DEFAULT_SURVEY_FILE` in `survey_files`, currently
`2026-September-SAM-606`), the same as the default of the Geant4 geometry.

```
loki-info                                         # summary of all banks: pixel id ranges, front face centres, distances
loki-info --list-calibrations                     # the calibration names
loki-info -b 5                                    # one bank: front face centre, axes u / w / n, corners
loki-info -c nominal-cad-geometry -b 5            # the same bank in the intended (CAD) geometry
loki-info -c nominal-geant4-geometry              # the summary for the nominal Geant4 geometry
loki-info -b 2 -t 0                               # a tube: its axis and the 7 straws (ends, pixel ids)
loki-info -b 5 -t 30                              # tube 30 of bank 5: layer 1, row 2
loki-info -b 2 -t 0 -s 3                          # one straw
loki-info -p 401408                               # a pixel by its 0-based pixel id (Geant4) ...
loki-info -d 401409                               # ... or by its detector number (pixel id + 1)
loki-info -p 401408 -c nominal-cad-geometry       # the same pixel in the CAD geometry
loki-info -p 401408 --unit m --precision 6        # positions in m with 6 decimals
loki-info -n 512 -d 802817                        # 512 pixels per straw (loki.json): the first pixel of bank 1
loki-info -b 0 -r 10000                           # the rear bank with its front face at 10 m
loki-info -n 8 -r 4000 -p 0                       # 8 pixels per straw, rear bank at 4 m
```

See [Conventions](#conventions) for the ids. The rear bank is placed as in the calibration, at the carriage zero position (CAD:
4999.8 mm, SAM-606: 4985.65 mm); with
`--rear-detector-distance` it is moved along z so that its front face centre is at that distance (as Geant4 does). The
pixel geometry is built from the bank corners by `bank_pixels`, so it is the same as in the NeXus-JSON output and as in
Geant4 for the same calibration (the tests compare it with the Geant4 pixel centres for the nominal and the SAM-606
placements, and the CAD geometry with `REFERENCE_BANKS_DATA` and `loki.json`). From Python:

```python
import numpy as np

from loki_geometry.geometry_info import LokiGeometry

geometry = LokiGeometry()  # the default calibration, 256 pixels per straw
pixel = geometry.pixel(401408)  # PixelInfo: ids, centre, straw ends, straw direction, ...
print(pixel.bank_id, pixel.tube_id, pixel.straw_id, pixel.pixel_in_straw, pixel.centre)
bank = geometry.bank(
    5
)  # BankGeometry: front_face_centre, along_tubes, across_tubes, layer_normal, distance, ...
first, last = geometry.straw_ends(5, 30, 3)  # the ends of a straw [mm]
pixel_id = geometry.pixel_id(
    5, 30, 3, 10
)  # and back: bank, tube, straw, pixel in the straw -> pixel id

# all pixel centres of a bank, with their pixel ids, to a text file
centres = geometry.bank_pixel_centres(5)  # shape (pixels, 3) [mm], in pixel id order
ids = geometry.first_pixel_id(5) + np.arange(len(centres))
np.savetxt(
    "bank5_pixel_centres.txt", np.column_stack([ids, centres]), fmt=["%d", "%.4f", "%.4f", "%.4f"]
)

# how far the survey calibration moves the pixels from the intended (CAD) positions, per bank
cad = LokiGeometry("nominal-cad-geometry")
for bank_id in range(9):
    shifts = np.linalg.norm(
        geometry.bank_pixel_centres(bank_id) - cad.bank_pixel_centres(bank_id), axis=1
    )
    print(f"bank {bank_id}: largest pixel shift {shifts.max():.2f} mm")
```

## Bank calibration for the Geant4 geometry

The Geant4 geometry ([G4GeoLoki](https://github.com/ess-dg/dgcode_projects/tree/main/LOKI/G4GeoLoki)) can place
the banks from a calibration instead of its nominal values. It reads one file per calibration, which `loki-calibrate`
writes:

```
loki-calibrate --geant4-calibration-file PATH [--geant4-calibration-name NAME]
```

- `PATH` is a file, or a directory: then the file `bank_calibration_<NAME>.txt` is written in it. Giving the data
  directory of G4GeoLoki in a dgcode_projects checkout (`LOKI/G4GeoLoki/data/`) puts it where Geant4 looks for it.
- `NAME` is the name of the calibration, i.e. the value of the Geant4 geometry parameter `bank_calibration`
  (letters, digits, `.`, `_`, `-`; `nominal-geant4-geometry` is reserved for the built-in nominal placement of Geant4). The default is
  the `calibration_name` of the survey file's entry in `survey_files` (`2026-September-SAM-606` for SAM-606). It is
  required for `--data reference` and `--data example`, which have no entry.
- The other options work as usual (`--survey-file`, `--data`, `--verbose`, …).

The file (`geant4_calibration.py` describes the format) has a header with the provenance (package version, survey file
and its checksum), then one line per bank with the front face centre F and the unit vectors u (along the tubes, from the
last to the first pixel end), w (across the tubes in the tube plane, towards tube 0) and n = u × w (the layer normal,
away from the sample), in the NeXus frame in mm, with full double precision: the `front_face_centre`,
`vector_along_tubes`, `vector_perpendicular_tubes` and `vector_perpendicular_layers` of `calibration.Bank`. These fix
the full rigid placement of each bank (3 rotations and 3 shifts); the tube grid inside the bank stays the one of
Geant4. All 9 banks are written; a bank that is not in the survey is written from
`REFERENCE_BANKS_DATA` (the intended geometry), marked with a comment in the file. The rear bank (0) moves along the
beam: Geant4 uses its axes and the x and y of its front face centre, while the z is set by the rear detector distance of
the simulation. Banks that are not in the file keep their nominal Geant4 placement.

In Geant4 the calibration is selected by name, e.g. `sb_loki_sim bank_calibration=2026-September-SAM-606 …`, after
rebuilding dgcode (`sb`) so that the new data file is installed; `bank_calibration=<path of a file>` uses a file
directly. With `--data example` (the Geant4 geometry as input) the file reproduces the nominal Geant4 placement.
Examples:

```
loki-calibrate --geant4-calibration-file DGCODE/LOKI/G4GeoLoki/data/      # the default survey (SAM-606)
loki-calibrate --survey-file NEW.txt --geant4-calibration-file DGCODE/LOKI/G4GeoLoki/data/
loki-calibrate --data reference --geant4-calibration-file DGCODE/LOKI/G4GeoLoki/data/ --geant4-calibration-name nominal-cad-geometry
loki-calibrate --geant4-calibration-file my_test.txt --geant4-calibration-name my-test   # bank_calibration=/path/to/my_test.txt
```

(`DGCODE` = a dgcode_projects checkout. The third one gives Geant4 the intended CAD geometry.)

## Other commands

```
# Self-test with the intended geometry: the corners must reproduce REFERENCE_BANKS_DATA (the largest difference is
# 0.066 mm, the rounding of the CAD values; exit code 1 above 0.1 mm). The quality report compares the geometry with
# itself (0.00).
loki-calibrate --data reference --verbose

# Test with the Geant4 geometry: reproduces the Geant4 tilts, section distances and offsets exactly
loki-calibrate --data example --verbose

# Only the conversion of the survey file (prints the checks and the calibration data)
loki-survey
loki-survey --survey-file NEW.txt             # another survey file (it needs a survey_files entry)

# Validate the Geant4 port against sb_loki_info values (exit code 1 if it deviates)
loki-geant4-validate

# Rebuild the intended corners from distances / angles / offsets and compare with REFERENCE_BANKS_DATA
loki-nominal-corners          # all banks, or only some: loki-nominal-corners 1 2

# Geometry information (positions and ids), see "Geometry information" above
loki-info -p 401408

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
    get_measured_calibration_data(),  # the default survey (SAM-606)
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

- **Tube end adjustment:** the difference between the measured and the effective tube length is assumed to be split
  equally between the two ends (step 6). If the end electronics are not equally long on both ends, the bank is shifted
  along its tubes by half the difference.
- **Geant4 geometry** ([G4GeoLoki](https://github.com/ess-dg/dgcode_projects/tree/main/LOKI/G4GeoLoki)): it uses rounded CAD values, so `REFERENCE_BANKS_DATA` differs slightly from it (most in bank 8:
  corners up to 0.96 mm, front face centre 0.59 mm, bank face 0.05°).

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
