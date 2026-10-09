"""Tests of the geometry information tool (geometry_info.py, loki-info): the named
calibrations, the ids, and the positions compared with the reference values:
REFERENCE_BANKS_DATA and loki.json (nominal-cad-geometry), the Geant4 model
(nominal-geant4-geometry, the Python port and sb_loki_info values) and the Geant4
geometry with the SAM-606 bank calibration (hard-coded values of G4GeoLoki)."""

import os

import numpy as np
import pytest
from geant4_reference import GEANT4_SAM606_PIXELS
from loki_json_reference import LOKI_JSON_SAMPLE_OFFSETS_M, LOKI_JSON_TRANSLATIONS

from loki_geometry import bank_pixels as lbp
from loki_geometry import constants as c
from loki_geometry import geant4 as g4
from loki_geometry import geometry_info as gi
from loki_geometry.reference_data import REFERENCE_BANKS_DATA
from loki_geometry.survey_files import DEFAULT_SURVEY_FILE, SURVEY_FILES

SAM606 = "2026-September-SAM-606"
CORNER_TUBES = ("front top", "back top", "front bottom", "back bottom")


@pytest.fixture(scope="module")
def cad():
    return gi.LokiGeometry(gi.NOMINAL_CAD)


@pytest.fixture(scope="module")
def cad512():
    return gi.LokiGeometry(gi.NOMINAL_CAD, 512)


@pytest.fixture(scope="module")
def geant4_nominal():
    return gi.LokiGeometry(gi.NOMINAL_GEANT4)


def corner_tube_ids(bank_id):
    rows = c.TUBES_PER_LAYER[bank_id]
    return (0, 3 * rows, rows - 1, 4 * rows - 1)  # the order of the 'A' / 'B' corners


# --- calibrations ---------------------------------------------------------------


def test_the_calibrations():
    names = gi.available_calibrations()
    assert list(names)[:2] == ["nominal-cad-geometry", "nominal-geant4-geometry"]
    assert SAM606 in names
    # the default: the calibration of the default survey file (as the default of Geant4)
    assert gi.DEFAULT_CALIBRATION == SAM606
    assert SURVEY_FILES[os.path.basename(DEFAULT_SURVEY_FILE)].calibration_name == SAM606
    default = gi.LokiGeometry()
    assert default.calibration == SAM606
    assert np.array_equal(
        default.bank_pixel_centres(5), gi.LokiGeometry(SAM606).bank_pixel_centres(5)
    )
    with pytest.raises(ValueError, match="unknown calibration"):
        gi.LokiGeometry("nominal-geometry")


@pytest.mark.parametrize("pixels_per_straw", [0, -1, 2.5, "8", True])
def test_invalid_pixels_per_straw(pixels_per_straw):
    with pytest.raises(ValueError):
        gi.LokiGeometry(pixels_per_straw=pixels_per_straw)


# --- nominal-cad-geometry: the reference values ---------------------------------


@pytest.mark.parametrize("bank_id", c.BANK_IDS)
def test_cad_reproduces_the_reference_corners(cad, bank_id):
    """The straw ends of the corner tubes are the REFERENCE_BANKS_DATA corners: exactly for
    the 4 corners that define the bank (A[0], A[1], A[2], B[0]), within the rounding of
    the CAD values (0.01 mm) for the other 4."""
    reference = REFERENCE_BANKS_DATA[bank_id]
    worst = 0.0
    for index, tube in enumerate(corner_tube_ids(bank_id)):
        first, last = cad.tube_ends(bank_id, tube)
        for point, expected, defining in (
            (first, reference["A"][index], index < 3),
            (last, reference["B"][index], index == 0),
        ):
            deviation = np.abs(point - np.array(expected)).max()
            if defining:
                assert deviation < 1e-9, (bank_id, CORNER_TUBES[index])
            worst = max(worst, deviation)
    assert worst < 0.011


def test_cad_pixel_positions_agree_with_loki_json(cad512):
    """loki.json (512 pixels per straw) was made from the CAD corners: translation + pixel
    offset (5 decimals in m) agree within 0.011 mm."""
    for bank_id, samples in LOKI_JSON_SAMPLE_OFFSETS_M.items():
        magnitude, vector = LOKI_JSON_TRANSLATIONS[bank_id]
        translation = magnitude * np.array(vector) * 1000
        first = cad512.first_pixel_id(bank_id)
        for local, offset in samples.items():
            expected = translation + np.array(offset) * 1000
            assert np.abs(cad512.pixel_centre(first + local) - expected).max() < 0.011, (
                bank_id,
                local,
            )


def test_cad_pixel_centres_are_the_bank_pixels_of_the_reference_corners(cad):
    """Consistency of LokiGeometry with bank_pixels (the same code): the single-pixel lookup
    agrees with the bank arrays. The geometry itself is checked against loki.json above."""
    for bank_id in c.BANK_IDS:
        expected = lbp.compute_bank_pixels(bank_id, REFERENCE_BANKS_DATA[bank_id]).pixel_centres
        centres = cad.bank_pixel_centres(bank_id)
        assert np.array_equal(centres, expected)
        first = cad.first_pixel_id(bank_id)
        for local in (0, 1, 255, 256, len(centres) // 2 + 3, len(centres) - 1):
            assert np.allclose(cad.pixel_centre(first + local), centres[local], rtol=0, atol=1e-9)


def test_cad_front_face_centres_and_distances(cad):
    """The front face centre: the middle of the front layer tube axes, one tube radius
    towards the sample. Pinned: the CAD distances of the drawing (SI-7615-463), with the
    panel offsets of the front banks on top (e.g. bank 5: sqrt(1364.68^2 + 100^2))."""
    expected = (4999.80, 2949.84, 3349.84, 2949.76, 3349.76, 1368.12, 1749.91, 1343.93, 1750.15)
    for bank_id in c.BANK_IDS:
        bank = cad.bank(bank_id)
        assert bank.distance == pytest.approx(expected[bank_id], abs=0.005)
        corners = REFERENCE_BANKS_DATA[bank_id]
        mid = (np.array(corners["A"][0]) + corners["A"][2] + corners["B"][0] + corners["B"][2]) / 4
        assert np.allclose(
            bank.front_face_centre, mid - bank.layer_normal * c.TUBE_OUTER_RADIUS, atol=0.01
        )


@pytest.mark.parametrize("bank_id", c.BANK_IDS)
def test_bank_axes(cad, bank_id):
    """u, w, n: right-handed orthonormal, n away from the sample, u from the last to the
    first pixel end, w towards tube 0."""
    bank = cad.bank(bank_id)
    u, w, n = bank.along_tubes, bank.across_tubes, bank.layer_normal
    assert np.allclose(np.array([u, w, n]) @ np.array([u, w, n]).T, np.eye(3), atol=1e-12)
    assert np.allclose(np.cross(u, w), n, atol=1e-12)
    assert np.dot(n, bank.front_face_centre) > 0
    first, last = cad.tube_ends(bank_id, 0)
    assert np.allclose((first - last) / bank.straw_length, u, atol=1e-12)
    tube0 = cad.tube_ends(bank_id, 0)[0]
    tube1 = cad.tube_ends(bank_id, 1)[0]
    assert np.dot(tube0 - tube1, w) == pytest.approx(c.TUBE_GRID_SIDE, abs=0.01)


# --- nominal-geant4-geometry -----------------------------------------------------


def test_geant4_nominal_is_the_geant4_port(geant4_nominal):
    """The Python port of the Geant4 pixel centres (geant4.py), for every 97th pixel."""
    worst = max(
        np.abs(geant4_nominal.pixel_centre(p) - np.array(g4.pixel_centre(p))).max()
        for p in range(0, geant4_nominal.total_number_of_pixels, 97)
    )
    assert worst < 1e-9


def test_geant4_nominal_agrees_with_sb_loki_info(geant4_nominal):
    """The pixel centres printed by the Geant4 code (sb_loki_info, 6 decimals)."""
    for pixel, expected in g4._SB_LOKI_INFO_REFERENCE.items():
        assert np.abs(geant4_nominal.pixel_centre(pixel) - np.array(expected)).max() < 1e-6


def test_cad_and_geant4_nominal_differ_by_less_than_a_millimetre(cad, geant4_nominal):
    """Geant4 uses rounded CAD values: up to 0.96 mm (bank 8); the rear bank is 0.2 mm
    closer in CAD (4999.8 mm) than in Geant4 at 5 m."""
    worst = {}
    for bank_id in c.BANK_IDS:
        difference = cad.bank_pixel_centres(bank_id) - geant4_nominal.bank_pixel_centres(bank_id)
        worst[bank_id] = np.abs(difference).max()
    assert max(worst.values()) < 1.0
    assert worst[8] == max(worst.values())
    assert worst[0] == pytest.approx(0.2, abs=1e-6)


# --- the rear detector distance --------------------------------------------------


@pytest.mark.parametrize("calibration", [gi.NOMINAL_CAD, gi.NOMINAL_GEANT4, SAM606])
@pytest.mark.parametrize("distance", [4000.0, 10000.0])
def test_rear_detector_distance_moves_only_bank_0_along_z(calibration, distance):
    base = gi.LokiGeometry(calibration, 2)
    moved = gi.LokiGeometry(calibration, 2, rear_detector_distance=distance)
    assert moved.bank(0).front_face_centre[2] == pytest.approx(distance, abs=1e-9)
    shift = moved.bank_pixel_centres(0) - base.bank_pixel_centres(0)
    assert np.allclose(shift[:, :2], 0, atol=1e-9)
    assert np.allclose(shift[:, 2], shift[0, 2], atol=1e-9)
    for bank_id in range(1, 9):
        assert np.array_equal(moved.bank_pixel_centres(bank_id), base.bank_pixel_centres(bank_id))


# --- the SAM-606 calibration vs the Geant4 geometry with it ----------------------


def test_sam606_agrees_with_the_geant4_geometry():
    """The same pixel centres as Geant4 with the same calibration (all 9 banks; the rear bank
    of both at 5 m). Banks 1-8 within 1e-6 mm. Bank 0 (not surveyed, from the CAD geometry):
    within 0.002 mm, as its tube grid is built here from the REFERENCE_BANKS_DATA corners
    (rounded to 0.01 mm), but in Geant4 from the exact tube pitches (the same at pixel 0, the
    corner A[0])."""
    assert len(GEANT4_SAM606_PIXELS) == 27
    geometry = gi.LokiGeometry(SAM606, 256, rear_detector_distance=5000.0)
    for pixel, expected in GEANT4_SAM606_PIXELS.items():
        tolerance = 0.002 if geometry.decode(pixel)[0] == 0 else 1e-6
        assert np.abs(geometry.pixel_centre(pixel) - np.array(expected)).max() < tolerance, pixel
    assert np.abs(geometry.pixel_centre(0) - np.array(GEANT4_SAM606_PIXELS[0])).max() < 1e-6


def test_sam606_takes_bank_0_from_the_cad_geometry(cad):
    geometry = gi.LokiGeometry(SAM606)
    assert np.array_equal(geometry.bank_pixel_centres(0), cad.bank_pixel_centres(0))
    assert not np.array_equal(geometry.bank_pixel_centres(5), cad.bank_pixel_centres(5))


# --- ids -------------------------------------------------------------------------


@pytest.mark.parametrize("pixels_per_straw", [1, 7, 256, 512])
def test_pixel_ids(pixels_per_straw):
    geometry = gi.LokiGeometry(pixels_per_straw=pixels_per_straw)
    n = pixels_per_straw
    for bank_id in c.BANK_IDS:
        assert geometry.first_pixel_id(bank_id) == lbp.bank_first_pixel_id(bank_id, n) - 1
        assert geometry.number_of_pixels(bank_id) == lbp.pixels_in_bank(bank_id, n)
    assert geometry.total_number_of_pixels == sum(c.TUBES_PER_BANK) * 7 * n == 896 * 7 * n
    step = max(1, geometry.total_number_of_pixels // 4999)
    for pixel in list(range(0, geometry.total_number_of_pixels, step)) + [
        geometry.total_number_of_pixels - 1
    ]:
        bank_id, tube, straw, k = geometry.decode(pixel)
        assert geometry.pixel_id(bank_id, tube, straw, k) == pixel
        assert g4.pixel_id(bank_id, tube, straw, k, n) == pixel  # the Geant4 numbering


def test_pixel_info(cad):
    p = cad.pixel(802816 + (30 * 7 + 3) * 256 + 10)  # bank 5, tube 30 (layer 1, row 2), straw 3
    assert (p.bank_id, p.tube_id, p.straw_id, p.pixel_in_straw, p.layer, p.row) == (
        5,
        30,
        3,
        10,
        1,
        2,
    )
    assert p.detector_number == p.pixel_id + 1
    assert p.pixel_length == pytest.approx(1200 / 256)
    first, last = cad.straw_ends(5, 30, 3)
    assert np.allclose(p.centre, first + (last - first) * 10.5 / 256)
    assert np.allclose(p.straw_direction, -cad.bank(5).along_tubes)


@pytest.mark.parametrize(
    "call",
    [
        lambda g: g.decode(-1),
        lambda g: g.decode(g.total_number_of_pixels),
        lambda g: g.pixel_id(9, 0, 0, 0),
        lambda g: g.pixel_id(1, 64, 0, 0),
        lambda g: g.pixel_id(1, 0, 7, 0),
        lambda g: g.pixel_id(1, 0, 0, 256),
        lambda g: g.straw_ends(0, 224, 0),
        lambda g: g.bank(-1),
        lambda g: g.pixel(True),
    ],
)
def test_invalid_ids(cad, call):
    with pytest.raises(ValueError):
        call(cad)


# --- command line ------------------------------------------------------------------


def run(argv, capsys):
    code = gi.main(argv)
    return code, capsys.readouterr().out


def test_cli_summary(capsys):
    code, out = run([], capsys)
    assert code == 0
    assert f"Calibration: {SAM606}, 256 pixels per straw" in out  # the default
    assert "(102.047, 723.057, 1141.389)  1354.988" in out  # bank 5
    assert "-0.000" not in out
    _, cad = run(["-c", "nominal-cad-geometry"], capsys)
    assert "Calibration: nominal-cad-geometry, 256 pixels per straw" in cad
    assert "(100.000, 727.187, 1154.532)  1368.118" in cad


def test_cli_list_calibrations(capsys):
    code, out = run(["--list-calibrations"], capsys)
    assert (
        f"{SAM606} (default)" in out
        and "nominal-cad-geometry:" in out
        and "nominal-geant4-geometry:" in out
    )


def test_cli_pixel_and_detector_number(capsys):
    _, by_id = run(["-p", "401408"], capsys)
    _, by_number = run(["-d", "401409"], capsys)
    assert by_id == by_number
    assert "Pixel id 401408 (detector number 401409)" in by_id
    assert "bank 1, tube 0 (layer 0, row 0), straw 0, pixel 0 in the straw" in by_id
    _, in_m = run(
        ["-c", "nominal-cad-geometry", "-p", "0", "--unit", "m", "--precision", "6"], capsys
    )
    assert "centre [m]: (0.498047, 0.779861, 5.004834)" in in_m


def test_cli_bank_tube_straw(capsys):
    _, bank = run(["-c", SAM606, "-b", "2"], capsys)
    assert (
        "Bank 2 (mid left)" in bank
        and "front face centre F [mm]: (379.498, -1.179, 3360.794)" in bank
    )
    _, tube = run(["-b", "2", "-t", "0"], capsys)
    assert tube.count("  straw ") == 7 and "straw 3: pixel ids 516864..517119" in tube
    _, straw = run(["-b", "2", "-t", "0", "-s", "3"], capsys)
    assert "Bank 2, tube 0, straw 3" in straw and straw.count("  straw ") == 1


@pytest.mark.parametrize(
    "argv,message",
    [
        (["-c", "unknown"], "unknown calibration"),
        (["-t", "1"], "--tube needs --bank"),
        (["-b", "1", "-s", "1"], "--straw needs --tube"),
        (["-p", "-1"], "pixel id must be"),
        (["-b", "1", "-t", "64"], "tube id must be"),
        (["-n", "0"], "pixels_per_straw"),
        (["--precision", "-1"], "--precision"),
    ],
)
def test_cli_errors(argv, message, capsys):
    with pytest.raises(SystemExit) as error:
        gi.main(argv)
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_the_corners_can_not_change_the_package_constants(cad):
    """The corners of a LokiGeometry are an immutable copy (REFERENCE_BANKS_DATA has lists)."""
    corners = cad.corners[1]
    assert corners["A"][0] == tuple(REFERENCE_BANKS_DATA[1]["A"][0])
    assert corners is not REFERENCE_BANKS_DATA[1]
    with pytest.raises(TypeError):
        corners["A"][0] = (0.0, 0.0, 0.0)


def test_a_new_survey_calibration_is_available(monkeypatch, tmp_path):
    """A survey_files entry with a calibration_name is a calibration of the tool (here a copy
    of the SAM-606 entry under another name: the same geometry)."""
    import shutil

    from loki_geometry import survey_files as sf

    shutil.copy(sf.SAM606.path, tmp_path / sf.SAM606.file_name)
    shutil.copy(sf.SAM606.path, tmp_path / "copy.txt")
    monkeypatch.setattr(sf, "DATA_DIR", str(tmp_path))
    entry = sf.SurveyFile(
        "copy.txt",
        sf.SAM606.label_map,
        frame=sf.SAM606.frame,
        fixes=sf.SAM606.fixes,
        calibration_name="2027-January-test",
    )
    monkeypatch.setitem(sf.SURVEY_FILES, entry.file_name, entry)
    assert "2027-January-test" in gi.available_calibrations()
    copy = gi.LokiGeometry("2027-January-test", 2)
    original = gi.LokiGeometry(SAM606, 2)
    for bank_id in c.BANK_IDS:
        assert np.array_equal(
            copy.bank_pixel_centres(bank_id), original.bank_pixel_centres(bank_id)
        )


def test_cli_values(capsys):
    _, rear = run(["-b", "0", "-r", "10000"], capsys)
    assert "rear detector distance 10000.0 mm" in rear
    assert "front face centre F [mm]: (0.000, 0.000, 10000.000)" in rear
    _, geant4 = run(["-c", "nominal-geant4-geometry"], capsys)
    assert "Calibration: nominal-geant4-geometry" in geant4
    assert "(100.000, 727.206, 1154.783)  1368.339" in geant4  # bank 5 in the Geant4 tables
    _, straw = run(["-n", "4", "-b", "1", "-t", "0", "-s", "3"], capsys)
    first = 401408 // 256 * 4 + 3 * 4  # bank 1 with 4 pixels per straw, tube 0, straw 3
    assert f"straw 3: pixel ids {first}..{first + 3}" in straw


@pytest.mark.parametrize("number", [0, 1605633])
def test_cli_detector_number_out_of_range(number, capsys):
    with pytest.raises(SystemExit) as error:
        gi.main(["-d", str(number)])
    assert error.value.code == 2
    assert "detector number must be in 1 .. 1605632" in capsys.readouterr().err


def test_failed_survey_checks_are_reported(monkeypatch, capsys):
    """The survey checks only print warnings (in the survey output, which loki-pixel-json /
    loki-info do not show): a failed check must be reported on stderr. A label map with a
    tube one row off (C2UP3 is tube 13 of bank 1) fails the tube id check."""
    from loki_geometry import survey_files

    monkeypatch.setattr(gi, "_reported_survey_warnings", set())
    gi.calibration_corners("2026-September-SAM-606")
    assert capsys.readouterr().err == ""  # the real survey passes all checks
    monkeypatch.setitem(survey_files.SAM606_LABEL_MAP, "C2UP3", (1, 12))
    gi.calibration_corners("2026-September-SAM-606")
    gi.calibration_corners("2026-September-SAM-606")  # reported once
    err = capsys.readouterr().err
    assert err.count("survey check(s) of the calibration 2026-September-SAM-606 failed") == 1
    assert "mismatch" in err
