# Changelog

Changes between the versions of loki-geometry (`__version__` in `src/loki_geometry/__init__.py`, written into the
output files).

## 0.4.0 (2026-11-17)

**The straw ids follow the real detectors.** The readout of the LoKI detectors as built numbers the 7 straws of a tube
in the opposite order to the originally intended numbering of the tube design: straw id s is where the intended
numbering puts straw 6 − s (the straw hexagon turned by 180° about the tube axis), in every tube of every bank. This
was found in the AmBe source tests of October 2026. The pixel positions now follow the real numbering
(`constants.LOKI_STRAW_POSITIONS_IN_TUBE`):

- Changed: the position of every pixel of the 6 outer straws of each tube, by detector number (pixel id), in all
  outputs: `loki-pixel-json`, `loki-info`, `loki-verify-json` and the Python API. A pixel moves by up to 15.5 mm
  (twice the straw-to-straw distance); the centre straw (3), the bank corners, the bank positions and the
  calibrations are unchanged.
- Instrument files made with the intended numbering (earlier versions, the ESS instrument file `loki.json`) have the
  straws in the other order: `loki-verify-json` reports deviations of about 15.5 mm for them.
- Tests: the comparisons with `loki.json` map straw s to 6 − s; the reference pixels of the Geant4 port were
  regenerated (the earlier values are reproduced at straw 6 − s).

Also since 0.3.0 (2026-10-16, "Minor fixes"):

- survey: a `carriage_position` without bank 0 in the label map is an error; the expected removed length of the end
  electronics is a survey file field (`excess_length_range`, default 20-40 mm).
- calibration: `tilt_angle` is the tilt in the section plane only; new `roll_angle` (rotation of the tubes in the
  bank plane), in the quality report.
- loki-verify-json: a relative `depends_on` of a bank group is resolved in the group.
- loki-info: the corner sources come from the label map.
- README: the SAM-606 bank 0 survey (detector carriage at 5.0 m), labels, survey file fields.

## 0.3.0 (2026-10-16)

- SAM-606: the rear bank (0) from the Rev2 measurement (tubes P1 and P56, detector carriage at 5.0 m); the SAM-606
  calibration now has all 9 banks (banks 1-8 unchanged). New `SurveyFile.carriage_position`.
- loki-pixel-json `--visualize`: bank 0 placed by its own translation.

## 0.2.0 (2026-10-09)

- loki-pixel-json writes the calibrated (surveyed) pixel geometry, as the instrument tree (readable like `loki.json`);
  optional centre of mass as the bank origin.
- loki-info: geometry information for named bank calibrations (SAM-606 by default).
- Bank calibration file export for the Geant4 geometry; registry of survey files.
- SAM-606 sample-to-window distance: the measured 83.11 mm.
- Stricter input checks (corners, straw length, survey checks) and fixes from the code review; loki-verify-json
  follows the full `depends_on` chain of each bank.

## 0.1.0 (2026-09-28)

- First version of the package: conversion of the SAM-606 survey to calibration data, bank corners and quality
  report, pixel geometry export for the instrument NeXus-JSON file, Python port of the Geant4 pixel centre
  calculation, shared geometry constants.
