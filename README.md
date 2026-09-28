# lokiJSON – LOKI detector bank calibration

Tools to derive the 8 corner positions of each LOKI detector bank from the as-built survey of the
detector tubes (SAM-606), and to turn corner positions into the pixel geometry (bank positions,
detector numbers and pixel offsets of all 1,605,632 pixels) for the NeXus/JSON instrument file.

All scripts need only Python 3 and numpy, and are run from this directory. The tests need pytest;
the JSON verification streams large files if `ijson` is installed (optional).

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
  `REFERENCE_BANKS_DATA` and `det_banks_data`.
- **Corners:** for each bank, 'A' and 'B' each list 4 points: [front layer top tube, back layer top tube,
  front layer bottom tube, back layer bottom tube] ("top/bottom" = lowest/highest tube id).
- **Bank distance** (drawing SI-7615-463, Geant4 `BcsBanks.cc`): distance from the sample to the centre of the
  bank's front face, measured in the section plane perpendicular to the tubes, along the bank position angle.
  The front face centre is the centre of the front-layer tube centres, moved one tube radius (12.7 mm) towards
  the sample along the bank normal. The panel offset (along the tubes: +100 / −15 / −100 / +50 mm for banks 5–8)
  comes on top, so the 3D distance is √(distance² + offset²).

## Files

| File | What it is |
| --- | --- |
| `data/ESS.NSS.D03.LOKI.OB.DETECTOR_PIPE_CENTRES.TANK_WINDOW_CS.txt` | SAM-606 survey of the tube ends: 32 points (2 tubes × 2 ends for banks 1–8) as `label, X, Y, Z` [mm] in the TANK WINDOW coordinate system (X along the beam, Y left, Z up). The points are on the tube axis at the physical tube ends, including ~15 mm of end electronics on each end. Bank 0 was not surveyed. |
| `measured_calibration_data.py` | Converts the survey file into calibration data: label → (bank, tube id) map, bank 5 fix, frame conversion, sanity checks, trimming of the end electronics. |
| `loki_json_calibration.py` | The calibration: calibration data → 8 corner positions per bank, quality report, comparison with the intended geometry, output in the `det_banks_data` format. |
| `reference_banks_data.py` | `REFERENCE_BANKS_DATA`: the intended corner positions (exact CAD values, a copy of `det_banks_data` in generate-nexus-files) and a converter to the calibration data format. This is the "intended" geometry of the comparisons. |
| `example_calibration_data.py` | Calibration data from the Geant4 geometry (first and last pixel centres of the centre straw of two front-layer tubes per bank, all 9 banks), generated with `geant4_pixel_positions.py`. Used to test the calibration. |
| `geant4_pixel_positions.py` | Python port of the Geant4 pixel centre calculation (`AimHelper::getPixelCentreCoordinates` in dgcode_projects/LOKI/G4GeoLoki). It matches `sb_loki_info` to 1e-6 mm and all 12,544 pixels of `G4GeoLoki/scripts/test_pixel_position.log` within that file's 0.05 mm rounding. No dgcode environment is needed. |
| `loki_bank_pixels.py` | Pixel geometry core: from the 8 corners of a bank to the detector numbers and the pixel centre offsets of all its pixels (vector approach of generate-nexus-files, straw layout of Geant4, mirrored straw pattern for the upside-down banks 1, 2, 5, 6 derived from the corners). |
| `loki_nexus_json.py` | Writes the pixel geometry as NeXus-JSON nodes (the geometry parts of the `loki_detector_<id>` groups, plus `detector_carriage_zero` for the rear bank), ready to be copied into the instrument JSON file. Uses `REFERENCE_BANKS_DATA` for now. |
| `verify_nexus_json.py` | Compares the pixel positions of a NeXus-JSON file (the output above, or a full file like `loki.json`) with the Geant4 geometry. Python counterpart of `verifyJSON` in dgcode_projects. |
| `tests/` | pytest tests of the pixel geometry, the JSON export, the verification and the Geant4 port. |
| `loki_json_detector_offsets.py` | Rebuilds the intended corner positions from the nominal bank distances, angles and offsets, and compares them with `REFERENCE_BANKS_DATA`. |

Directories:
- `data/`: measurement data used by the scripts (version controlled).
- `tests/`: the pytest tests.
- `*_local/` (e.g. `sources_local/`): local material that is not version controlled, like documents, drawings, raw survey
  files and large JSON files. The tests that compare with the current instrument file look for `sources_local/loki.json`,
  or the path in the `LOKI_JSON` environment variable, and are skipped if it is missing.

## From the measured points to the 8 corner positions

```
python3 loki_json_calibration.py --data measured --verbose --compare-to-intended --print
```

1. **Read the survey** (`measured_calibration_data.read_measured_points`): `label, X, Y, Z` lines.
   Labels: `C1`/`C2` = front/middle carriage, `U/D/L/R` = up/down/left/right bank, `P<n>` = the surveyors' pipe number,
   `I`/`II` = the two ends of the tube.
2. **Fix bank 5** (`fix_bank5_mislabelled_tube`): `C1UP8I` and `C1UP8II` are on two different tubes (UP9 and UP8).
   Assuming parallel tubes, the missing end of tube UP9 is rebuilt as `C1UP8I + (C1UP3II − C1UP3I)`, and `C1UP8II` is dropped.
   A check confirms that the dropped point is one tube pitch (28.4 mm) from the rebuilt one.
3. **Convert the frame** (`survey_to_nexus_frame`): x = Y, y = Z, z = X + D, where D = `SAMPLE_TO_TANK_WINDOW_DISTANCE` is
   the sample-to-window distance. **D is provisional (100 mm)**; override it with `--sample-to-window-distance`.
   D shifts every bank equally along z and doesn't affect any of the quality numbers.
4. **Identify the tubes** (`TUBE_LABEL_MAP`, `group_points_into_tubes`): each label is mapped to (bank, local tube id).
   The surveyors' P numbers do not follow our numbering. `C1DP31` is really tube 18, for example.
   The first pixel end of each tube is chosen from the coordinates (+x / −y).
5. **Sanity checks** (with `--verbose`): the two tubes of a bank must be parallel; their distance divided by 28.4 mm must equal
   the tube id difference; each tube's position across the bank must match its assigned id in the nominal geometry.
6. **Remove the end electronics** (`find_effective_tube_ends`): the measured tubes are ~30 mm longer than the effective straw
   length. Both ends are moved inwards by half the excess, along the line through the two points.
   The same function stretches Geant4 end pixel centres to the straw ends.
7. **Build the bank** (`loki_json_calibration.Bank`, full description in the module docstring):
   - average direction along the tubes
   - least-squares plane through the 4 points (numpy SVD)
   - orthonormal basis: along the tubes, across the tubes in the plane, normal
   - tube 0 position averaged from both tubes' midpoints; midpoints don't depend on symmetric errors of the ends
   - front layer corners from the effective straw length and the 28.40 mm tube pitch
   - back layer corners from +78.78 mm along the normal and ±18.84 mm in the plane
8. **Quality report** (`--verbose`, `Detectors.print_quality_report`), per bank:
   - plane fit residual (with 4 points on 2 lines all 4 residuals are equal; it measures the twist of the two tube axes)
   - angle between the two tubes
   - spacing in tube pitches
   - disagreement of the two tubes' predictions of tube 0
   - the input tube midpoints recomputed from the final corners (end-to-end check)
   - tilt angle, front face distance and front face centre shift vs the intended geometry
   - the placement in the drawing's terms (section distance, position angle, offset along the tubes)
9. **Output** (`--print`, `Detectors.formatted_data`): the corners as straw ends in the `REFERENCE_BANKS_DATA` format,
   the input of `loki_bank_pixels.py` / `loki_nexus_json.py`. Bank 0 is missing because it was not measured, so take it
   from `REFERENCE_BANKS_DATA` or measure it. (Feeding the measured corners into `loki_nexus_json.py` is not wired up yet.)

## Pixel geometry for the instrument JSON file

The people producing the instrument JSON file need, for each bank, the bank translation (from the sample to the
bank position) and the detector numbers with the local pixel offsets. `loki_nexus_json.py` writes exactly these, as
NeXus-JSON nodes in the structure of `loki.json`:

```
python3 loki_nexus_json.py -o loki_detector_pixels.json               # all banks, 256 pixels per straw
python3 loki_nexus_json.py -o test.json --pixels-per-straw 2 --banks 1 2
python3 loki_nexus_json.py -o base.json --pixel-reference pixel_base  # offsets to the pixel bases
python3 verify_nexus_json.py loki_detector_pixels.json               # compare with Geant4
python3 verify_nexus_json.py path/to/loki.json                       # works on the full instrument file too
python3 verify_nexus_json.py path/to/loki.json --ignore-pixel-shape  # offsets only, without the pixel shape
```

- Output: `{"info": {...}, "loki_detector_0": <group>, …, "loki_detector_8": <group>, "detector_carriage_zero": <translation>}`.
  Each group holds `detector_number`, `pixel_shape`, `x/y/z_pixel_offset` [m], `transformations` (one translation
  `trans_<id+1>`) and `depends_on`.
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
- **Known issue in the current `loki.json`:** its offsets point to the pixel centres (since the generate-nexus-files fix
  b7269dc), but its pixel shape still starts at the offset point, as if the offsets pointed to the pixel bases. So its pixel
  volumes are shifted by half a pixel along the straws (0.98 / 1.17 mm with 512 pixels per straw). `verify_nexus_json.py`
  shows this unless `--ignore-pixel-shape` is given.
- `verify_nexus_json.py` computes the pixel centres as translation + offset + centre of the pixel shape cylinder, so it
  checks either convention and catches an inconsistent shape.
- Pixel ids: `bank first id + (tube · 7 + straw) · N + pixel`, with tube = layer · rows + row, as in Geant4 and the ICD.
- Checks (all in `tests/`):
  - With corners taken from the Geant4 geometry, all 1,605,632 pixels agree with the Geant4 port within 1e-6 mm.
  - With `REFERENCE_BANKS_DATA` they agree within 1 mm, which is the rounding of the CAD values in Geant4.
  - With 512 pixels per straw, the current `loki.json` offsets are reproduced within 0.011 mm, which is its 5-decimal rounding.
  - Both pixel reference conventions give the same pixel centres, and an inconsistent pixel shape is detected.

## Other commands

```
# Self-test with the intended geometry: must reproduce REFERENCE_BANKS_DATA (differences 0.00)
python3 loki_json_calibration.py --data reference --verbose

# Test with the Geant4 geometry: reproduces the Geant4 tilts, section distances and offsets exactly
python3 loki_json_calibration.py --data example --verbose

# Measured data with another sample-to-window distance
python3 loki_json_calibration.py --sample-to-window-distance 119 --verbose

# Only the conversion of the survey file (prints the checks and the calibration data)
python3 measured_calibration_data.py

# Validate the Geant4 port against sb_loki_info values
python3 geant4_pixel_positions.py

# Rebuild the intended corners from distances / angles / offsets and compare with REFERENCE_BANKS_DATA
python3 loki_json_detector_offsets.py

# Run the tests
python3 -m pytest tests
```

Using the pieces from Python:

```python
from measured_calibration_data import get_measured_calibration_data
from loki_json_calibration import Detectors
from geant4_pixel_positions import pixel_centre, pixel_id

measured = Detectors(get_measured_calibration_data(sample_to_window_distance=100.0), use_pixel_centre_position=True, verbose=True)
measured.compare_to(measured.intended, label="intended")      # corner shifts vs REFERENCE_BANKS_DATA
corners = measured.formatted_data()                            # det_banks_data format (straw ends)
x, y, z = pixel_centre(pixel_id(bank_id=1, tube_id=13, straw_id=3, pixel_in_straw=0))  # Geant4 pixel centre [mm]
```

## Open points / known issues

- **Sample-to-window distance D:** 119 mm (flange face) or 108 mm (sapphire window) from drawing SI-7615-463, depending on
  which surface is the survey origin; provisional value 100 mm.
- **End electronics:** assumed equally long on both ends. This matters for bank 7, where the two measured tubes are shifted
  2.8 mm relative to each other along the tubes, and which comes out 8 mm off its intended offset.
- **Bank 0:** not surveyed.
- **Geant4 geometry:** it uses rounded CAD values, so `REFERENCE_BANKS_DATA` differs slightly from it (up to 0.56 mm and
  0.05° in bank 8). The geometry building (`pycpp_GeoBCSBanks/geometry_module.cc`) and the pixel position code
  (`libsrc/AimHelper.cc`) both take all positions from `BcsBanks.cc`, but apply the rotations separately.
