"""Tests of the bank calibration (calibration.py): 8 corners from 2 tubes per bank,
its invariances, input checks, quality report, output format and command line."""

import contextlib
import copy
import dataclasses
import io
import re

import numpy as np
import pytest

from loki_geometry import calibration, survey, survey_files
from loki_geometry import constants as c
from loki_geometry import geant4 as g4
from loki_geometry import survey_files as sf
from loki_geometry.calibration import Detectors
from loki_geometry.geant4_example_data import example_calibration_data
from loki_geometry.reference_data import (
    REFERENCE_BANKS_DATA,
    convert_reference_data_to_calibration_data,
)
from loki_geometry.survey_files import SAM606


def _quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def _detectors(data, use_pixel_centre_position=True, **kwargs):
    kwargs.setdefault("intended", None)
    kwargs.setdefault("run_checks", False)
    return _quiet(Detectors, data, use_pixel_centre_position, **kwargs)


def _geant4_tube(bank_id, tube_id):
    """First / last pixel centre of the centre straw of a Geant4 tube."""
    n = c.DEFAULT_PIXELS_PER_STRAW
    first = g4.pixel_centre(g4.pixel_id(bank_id, tube_id, c.CENTRE_STRAW, 0))
    last = g4.pixel_centre(g4.pixel_id(bank_id, tube_id, c.CENTRE_STRAW, n - 1))
    return {"tube_id": tube_id, "first_pixel_end": list(first), "last_pixel_end": list(last)}


def _geant4_bank_data(bank_id, tube_ids):
    return {"bank_id": bank_id, "positions": [_geant4_tube(bank_id, t) for t in tube_ids]}


def _corners(detectors, bank_id):
    bank = detectors.banks[bank_id]
    return np.concatenate([bank.positionA, bank.positionB])


@pytest.fixture
def register(monkeypatch):
    def add(entry):
        monkeypatch.setitem(sf.SURVEY_FILES, entry.file_name, entry)
        return entry

    return add


def _fixed_sam606_points():
    return _quiet(sf.sam606_fix_bank5_mislabelled_tube, survey.read_measured_points(sf.SAM606.path))


def _write(path, points):
    path.write_text(
        "".join(
            f"{k}, {float(x)!r}, {float(y)!r}, {float(z)!r}\n" for k, (x, y, z) in points.items()
        )
    )
    return path


@pytest.fixture(scope="module")
def reference_detectors():
    return Detectors(
        convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA), True, run_checks=False
    )


@pytest.fixture(scope="module")
def measured_detectors():
    return Detectors(
        survey.get_measured_calibration_data(SAM606.path, verbose=False), True, run_checks=False
    )


def test_reference_data_reproduces_itself(reference_detectors):
    corners = reference_detectors.formatted_data()
    for bank_id in c.BANK_IDS:
        for end in "AB":
            assert np.allclose(corners[bank_id][end], REFERENCE_BANKS_DATA[bank_id][end], atol=0.07)


def test_straw_end_output_does_not_depend_on_the_pixel_centre_option(reference_detectors):
    # reference_detectors uses pixel centres; this one the straw ends
    detectors = Detectors(
        convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA),
        False,
        run_checks=False,
    )
    assert detectors.formatted_data() == reference_detectors.formatted_data()
    for bank_id in c.BANK_IDS:
        assert detectors.banks[bank_id].sample_bank_distance == pytest.approx(
            reference_detectors.banks[bank_id].sample_bank_distance
        )


def test_geant4_example_data_reproduces_the_nominal_placement():
    detectors = Detectors(example_calibration_data, True, intended=None, run_checks=False)
    for bank_id in c.BANK_IDS:
        assert detectors.tilt_angle(bank_id) == pytest.approx(c.BANK_TILT_ANGLES[bank_id], abs=1e-6)
        distance, angle, offset = detectors.section_parameters(bank_id)
        nominal = detectors.nominal_section_parameters(bank_id)
        assert (distance, angle, offset) == pytest.approx(nominal, abs=1e-5)  # data has 6 decimals


def _rotated(data, bank_id, axis, angle_deg, centre):
    """Calibration data with the tubes of a bank rotated about axis (through centre)."""
    k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    a = np.radians(angle_deg)
    matrix = (
        np.cos(a) * np.eye(3)
        + np.sin(a) * np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        + (1 - np.cos(a)) * np.outer(k, k)
    )
    data = copy.deepcopy(data)
    for bank in data:
        if bank["bank_id"] == bank_id:
            for tube in bank["positions"]:
                for end in ("first_pixel_end", "last_pixel_end"):
                    tube[end] = list(
                        matrix @ (np.asarray(tube[end], dtype=float) - centre) + centre
                    )
    return data


@pytest.mark.parametrize("bank_id", c.BANK_IDS)
def test_tilt_and_roll_angles_are_separated(bank_id):
    """A rotation of the tubes in the bank plane (about the normal) is the roll angle and does
    not change the tilt angle; a rotation about the tubes changes only the tilt angle."""
    data = convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA)
    reference = _detectors(data)
    bank = reference.banks[bank_id]
    centre = bank.front_face_centre
    assert reference.roll_angle(bank_id) == pytest.approx(0.0, abs=1e-3)  # CAD rounding
    rolled = _detectors(_rotated(data, bank_id, bank.vector_perpendicular_layers, 0.3, centre))
    assert rolled.roll_angle(bank_id) == pytest.approx(0.3, abs=2e-3)
    assert rolled.tilt_angle(bank_id) == pytest.approx(reference.tilt_angle(bank_id), abs=1e-3)
    tilted = _detectors(_rotated(data, bank_id, bank.vector_along_tubes, 0.3, centre))
    assert abs(tilted.tilt_angle(bank_id) - reference.tilt_angle(bank_id)) == pytest.approx(
        0.3, abs=1e-3
    )
    assert tilted.roll_angle(bank_id) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("bank_id", c.BANK_IDS)
def test_any_pair_of_front_tubes_recovers_the_geant4_corners(bank_id):
    """Two arbitrary front layer tubes (not only 0 and N-1, in any order, also
    neighbours) give the Geant4 corners exactly, including the back layer and
    its in-plane shift (sign differs for the upside-down banks)."""
    rows = c.TUBES_PER_LAYER[bank_id]
    expected = g4.bank_corners(bank_id)
    for tube_ids in [(0, rows - 1), (3, 9), (rows - 2, 1), (5, 6)]:
        detectors = _detectors([_geant4_bank_data(bank_id, tube_ids)])
        corners = detectors.formatted_data(decimals=9)[bank_id]
        for end in "AB":
            assert np.allclose(corners[end], expected[end], atol=1e-6), (tube_ids, end)


@pytest.mark.parametrize("bank_id", c.BANK_IDS)
def test_corners_follow_a_rigid_motion_of_the_input(bank_id):
    """Rotating and translating the input moves the corners in the same way."""
    rng = np.random.default_rng(1000 + bank_id)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    translation = rng.normal(size=3) * 1000.0
    data = _geant4_bank_data(bank_id, (2, 7))
    moved = copy.deepcopy(data)
    for tube in moved["positions"]:
        for key in ("first_pixel_end", "last_pixel_end"):
            tube[key] = list(rotation @ np.array(tube[key]) + translation)
    original = _corners(_detectors([data]), bank_id)
    result = _corners(_detectors([moved]), bank_id)
    assert np.allclose(result, original @ rotation.T + translation, atol=1e-8)


def test_order_of_the_input_tubes_and_banks_does_not_matter():
    reference = _detectors(example_calibration_data)
    shuffled = copy.deepcopy(example_calibration_data)[::-1]
    for bank in shuffled:
        bank["positions"] = bank["positions"][::-1]
    result = _detectors(shuffled)
    for bank_id in c.BANK_IDS:
        assert np.allclose(_corners(result, bank_id), _corners(reference, bank_id), atol=1e-9)


def test_symmetric_end_errors_do_not_change_the_corners():
    """Only the midpoints and directions are used: moving both ends of a tube
    symmetrically (end electronics, half pixels) changes nothing."""
    reference = _detectors(example_calibration_data)
    changed = copy.deepcopy(example_calibration_data)
    for bank in changed:
        for tube, delta in zip(bank["positions"], (17.0, -3.0)):
            first = np.array(tube["first_pixel_end"])
            last = np.array(tube["last_pixel_end"])
            u = (first - last) / np.linalg.norm(first - last)
            tube["first_pixel_end"] = list(first + u * delta)
            tube["last_pixel_end"] = list(last - u * delta)
    result = _detectors(changed)
    for bank_id in c.BANK_IDS:
        assert np.allclose(_corners(result, bank_id), _corners(reference, bank_id), atol=1e-9)


_INPUTS = {  # (functions: the survey is read when the test runs, not when it is collected)
    "example": lambda: example_calibration_data,
    "reference": lambda: convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA),
    "measured": lambda: _quiet(survey.get_measured_calibration_data, SAM606.path, verbose=False),
}


@pytest.mark.parametrize("data", list(_INPUTS))
def test_normal_points_away_from_the_sample_and_the_back_layer_is_behind(data):
    detectors = _detectors(_INPUTS[data]())
    for bank in detectors.banks.values():
        n = bank.vector_perpendicular_layers
        assert np.dot(n, bank.front_face_centre) > 0.9 * np.linalg.norm(bank.front_face_centre)
        # back layer (index 1, 3) farther from the sample than the front layer (0, 2)
        for front, back in ((0, 1), (2, 3)):
            depth = np.dot(bank.positionA[back] - bank.positionA[front], n)
            assert depth == pytest.approx(c.FRONT_TO_BACK_NORMAL, abs=1e-9)
        # front face 1 tube radius in front of the front layer tube centres
        front_centre = (
            bank.positionA[0] + bank.positionA[2] + bank.positionB[0] + bank.positionB[2]
        ) / 4
        assert np.dot(front_centre - bank.front_face_centre, n) == pytest.approx(
            c.TUBE_OUTER_RADIUS
        )


@pytest.mark.parametrize("bank_id", c.BANK_IDS)
def test_back_layer_in_plane_shift_sign_matches_reference_and_geant4(bank_id):
    """Front -> back layer shift in the tube plane: towards tube 0 for normal
    banks, towards the highest tube id for the upside-down banks 1, 2, 5, 6."""
    sign = -1 if bank_id in c.UPSIDE_DOWN_BANK_IDS else 1
    for corners in (REFERENCE_BANKS_DATA[bank_id], g4.bank_corners(bank_id)):
        a = np.array(corners["A"], dtype=float)
        towards_tube0 = (a[0] - a[2]) / np.linalg.norm(a[0] - a[2])
        shift = np.dot(a[1] - a[0], towards_tube0)
        assert shift == pytest.approx(sign * c.FRONT_TO_BACK_IN_PLANE, abs=0.1)


def test_formatted_data_has_the_reference_format():
    detectors = _detectors(example_calibration_data)
    data = detectors.formatted_data()
    assert list(data) == list(REFERENCE_BANKS_DATA)
    for bank_id, bank in data.items():
        reference = REFERENCE_BANKS_DATA[bank_id]
        assert list(bank) == list(reference)
        for key in reference:
            if key in ("A", "B"):
                assert len(bank[key]) == 4
                for point in bank[key]:
                    assert isinstance(point, tuple) and len(point) == 3
                    assert all(type(v) is float for v in point)
                assert np.allclose(bank[key], g4.bank_corners(bank_id)[key], atol=0.006)
            else:
                assert bank[key] == reference[key]
    # the template is not modified
    assert REFERENCE_BANKS_DATA[1]["A"][1] == (500, 699.24, 2979.43)


def test_formatted_data_straw_ends_are_one_straw_length_apart():
    data = _detectors(example_calibration_data).formatted_data(decimals=9)
    for bank_id, bank in data.items():
        lengths = np.linalg.norm(np.subtract(bank["B"], bank["A"]), axis=1)
        assert np.allclose(lengths, c.STRAW_LENGTHS[bank_id], atol=1e-6)


# SAM-606: the largest distance of the model tubes from the input tubes [mm] (calibrate
# --verbose: 0.75, 0.08, 0.12, 0.25, 0.71, 0.45, 0.37, 1.44, 0.15 for banks 0-8), with a 0.1 mm margin
SAM606_RESIDUAL_LIMITS = {
    0: 0.85,
    1: 0.2,
    2: 0.25,
    3: 0.35,
    4: 0.85,
    5: 0.55,
    6: 0.5,
    7: 1.55,
    8: 0.25,
}


def test_measured_banks_reproduce_their_input_tubes(measured_detectors):
    """Regression bounds per bank (the measured values plus a small margin): a change of the
    calibration that moves the model tubes by a few tenths of a mm in any bank shows up."""
    for bank_id, bank in measured_detectors.banks.items():
        worst = max(
            np.linalg.norm(residual) for _, residual, _ in bank.quality["reproduction_residuals"]
        )
        assert worst < SAM606_RESIDUAL_LIMITS[bank_id], (bank_id, worst)


def test_measured_banks_are_on_the_expected_side_of_the_beam():
    """Labels U/D/L/R vs the frame: y > 0 up, x > 0 left (beam's view), and the
    front bank offsets along the tubes have the nominal signs."""
    detectors = _detectors(_quiet(survey.get_measured_calibration_data, SAM606.path, verbose=False))
    expected_side = {1: (1, +1), 2: (0, +1), 3: (1, -1), 4: (0, -1),
                     5: (1, +1), 6: (0, +1), 7: (1, -1), 8: (0, -1)}  # fmt: skip
    for bank_id, (axis, sign) in expected_side.items():
        centre = detectors.banks[bank_id].front_face_centre
        assert np.sign(centre[axis]) == sign
        assert centre[2] > 0
    # the measured panel offsets differ from the nominal ones by +2.05, -2.30, +8.03, -4.06 mm
    # (banks 5-8): within 0.5 mm of these (a sign or axis error would be far off)
    measured_differences = {5: 2.05, 6: -2.30, 7: 8.03, 8: -4.06}
    for bank_id, difference in measured_differences.items():
        _, _, offset = detectors.section_parameters(bank_id)
        _, _, nominal = detectors.nominal_section_parameters(bank_id)
        assert offset - nominal == pytest.approx(difference, abs=0.5), bank_id


def test_measured_front_face_distance_shifts_with_the_sample_to_window_distance(
    measured_detectors, monkeypatch
):
    """The sample-to-window distance of the SAM-606 frame shifts every bank along z."""
    distance = survey_files.SAMPLE_TO_TANK_WINDOW_DISTANCE
    monkeypatch.setattr(survey_files, "SAMPLE_TO_TANK_WINDOW_DISTANCE", distance + 10.0)
    shifted = Detectors(
        survey.get_measured_calibration_data(SAM606.path, verbose=False), True, run_checks=False
    )
    for bank_id, bank in measured_detectors.banks.items():
        shift = shifted.banks[bank_id].front_face_centre - bank.front_face_centre
        assert np.allclose(shift, (0.0, 0.0, 10.0))


def test_sample_to_window_distance_changes_only_the_distance_numbers(monkeypatch):
    """README step 3: 'D shifts every bank equally along z and doesn't affect any of
    the quality numbers'. True for the fit quality numbers, NOT for the
    distance / position angle comparisons of the quality report."""
    before = _quiet(
        Detectors, survey.get_measured_calibration_data(SAM606.path, verbose=False), True
    )
    monkeypatch.setattr(sf, "SAMPLE_TO_TANK_WINDOW_DISTANCE", 150.0)
    after = _quiet(
        Detectors, survey.get_measured_calibration_data(SAM606.path, verbose=False), True
    )
    for bank_id in before.bank_ids:
        q0, q1 = before.banks[bank_id].quality, after.banks[bank_id].quality
        assert np.allclose(q0["plane_residuals"], q1["plane_residuals"], atol=1e-9)
        assert q0["tube_angle_deg"] == pytest.approx(q1["tube_angle_deg"], abs=1e-9)
        assert q0["spacing_in_pitches"] == pytest.approx(q1["spacing_in_pitches"], abs=1e-9)
        assert before.tilt_angle(bank_id) == pytest.approx(after.tilt_angle(bank_id), abs=1e-9)
        # ...but the section distance and the position angle do change
        d0, a0, _ = before.section_parameters(bank_id)
        d1, a1, _ = after.section_parameters(bank_id)
        assert d1 - d0 > 40.0
        if bank_id != 0:  # (the rear bank is on the beam axis: its position angle stays ~0)
            assert abs(a1 - a0) > 1e-3


def test_calibration_checks_pass_for_the_reference_data(capsys):
    calibration.Detectors(
        calibration.convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA), True
    )
    out = capsys.readouterr().out
    assert "All bank base vectors are orthogonal." in out
    assert "All bank tilt angles agree" in out
    assert "All bank sample-bank distances agree" in out
    assert "Warning" not in out


def test_quality_report_against_the_nominal_geant4_values(capsys):
    """intended=None: the report falls back to the Geant4 table (never printed in the tests)."""
    detectors = Detectors(example_calibration_data, True, verbose=True, intended=None)
    out = capsys.readouterr().out
    assert "intended = Geant4 nominal values" in out
    for bank_id in c.BANK_IDS:
        assert detectors.intended_tilt(bank_id) == c.BANK_TILT_ANGLES[bank_id]
        assert detectors.intended_section_parameters(
            bank_id
        ) == detectors.nominal_section_parameters(bank_id)


def test_compare_to_skips_banks_missing_in_the_other_geometry(capsys):
    full = _quiet(Detectors, example_calibration_data, True, run_checks=False)
    partial = _quiet(Detectors, example_calibration_data[3:5], True, run_checks=False)
    full.compare_to(partial, label="partial")
    out = capsys.readouterr().out
    assert re.findall(r"bank (\d):", out) == ["3", "4"]


def test_compare_arrays_reports_differences(capsys):
    Detectors.compare_arrays([0.0, 0.0], [0.0, 2.0], tolerance=1.0)
    out = capsys.readouterr().out
    assert "Arrays differ! Max difference: 2.0" in out and "[[1]]" in out


def test_verify_positions_against_reference_data_accepts_the_reference_itself():
    detectors = _detectors(convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        worst = detectors.verify_positions_against_reference_data(tolerance=0.07)
    assert "Arrays differ" not in out.getvalue()
    assert worst < 0.07


def test_verify_positions_against_reference_data_reports_the_largest_deviation():
    data = copy.deepcopy(example_calibration_data)
    detectors = _quiet(calibration.Detectors, data, True, intended=None, run_checks=False)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        worst = detectors.verify_positions_against_reference_data(tolerance=0.1)
    assert 0.1 < worst < 1.0  # Geant4 (rounded CAD values) vs CAD: up to 0.56 mm (README)
    assert "Arrays differ" in out.getvalue()


def _swap_ends(tube):
    tube["first_pixel_end"], tube["last_pixel_end"] = (
        tube["last_pixel_end"],
        tube["first_pixel_end"],
    )


@pytest.mark.parametrize("bank_index", range(len(example_calibration_data)))
def test_swapped_pixel_ends_are_detected(bank_index):
    """Swapped ends in both tubes would flip the layer normal (back layers in
    front of the bank), swapped ends in one tube give opposite tube directions."""
    both = copy.deepcopy(example_calibration_data[bank_index])
    for tube in both["positions"]:
        _swap_ends(tube)
    with pytest.raises(ValueError, match="normal points towards the sample"):
        _detectors([both])
    one = copy.deepcopy(example_calibration_data[bank_index])
    _swap_ends(one["positions"][1])
    with pytest.raises(ValueError, match="opposite directions"):
        _detectors([one])


def test_input_tubes_with_the_same_id_are_rejected():
    data = copy.deepcopy(example_calibration_data[3])
    data["positions"][1]["tube_id"] = data["positions"][0]["tube_id"]
    with pytest.raises(ValueError, match="same id"):
        _detectors([data])


def test_more_than_two_input_tubes_give_a_clear_error():
    bank = copy.deepcopy(example_calibration_data[1])
    extra = copy.deepcopy(bank["positions"][0])
    extra["tube_id"] = 7
    bank["positions"].append(extra)
    with pytest.raises(ValueError, match="exactly 2 input tubes"):
        try:
            _quiet(calibration.Detectors, [bank], True, intended=None, run_checks=False)
        except ValueError as error:
            assert "unpack" not in str(error)
            raise


def test_calibrate_cli_reference_self_test_compares_with_reference_data(capsys):
    from loki_geometry import calibration

    assert calibration.main(["--data", "reference"]) == 0
    out = capsys.readouterr().out
    assert "Arrays differ" not in out
    worst = float(re.search(r"largest difference from REFERENCE_BANKS_DATA ([\d.]+) mm", out)[1])
    assert 0.0 < worst < 0.07  # the rounding of the CAD values, not 0 by construction


def test_calibrate_reference_self_test_fails_with_exit_code_1(monkeypatch):
    """The failure path of `loki-calibrate --data reference` (only success is tested)."""
    shifted = copy.deepcopy(convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA))
    for tube in shifted[3]["positions"]:
        for end in ("first_pixel_end", "last_pixel_end"):
            tube[end] = list(np.array(tube[end]) + [0.0, 0.0, 0.5])
    monkeypatch.setattr(
        calibration, "convert_reference_data_to_calibration_data", lambda data: shifted
    )
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert calibration.main(["--data", "reference"]) == 1
    assert "Arrays differ" in out.getvalue()


def test_calibrate_cli_rejects_an_unknown_survey_file_with_exit_code_2(tmp_path, capsys):
    path = tmp_path / "unknown.txt"
    path.write_text("C1UP3I, 1, 2, 3\n")
    with pytest.raises(SystemExit) as error:
        _quiet(calibration.main, ["--survey-file", str(path)])
    assert error.value.code == 2
    assert "unknown survey file 'unknown.txt'" in capsys.readouterr().err


def test_calibrate_cli_reports_bad_survey_geometry_as_a_usage_error(register, tmp_path, capsys):
    """A mirrored frame (determinant -1) flips the layer normal of bank 2:
    loki-calibrate should exit with a message (code 2), not a traceback."""
    path = _write(tmp_path / "mirror.txt", _fixed_sam606_points())
    register(
        dataclasses.replace(
            sf.SAM606,
            file_name="mirror.txt",
            checksum=None,
            fixes=(),
            frame=lambda p: np.array([-p[1], p[2], p[0] + 100.0]),
        )
    )
    with pytest.raises(SystemExit) as error:
        _quiet(calibration.main, ["--survey-file", str(path)])
    assert error.value.code == 2
    assert "normal points towards the sample" in capsys.readouterr().err


def test_a_bank_given_twice_is_rejected():
    data = copy.deepcopy(example_calibration_data[:2]) + [
        copy.deepcopy(example_calibration_data[1])
    ]
    with pytest.raises(ValueError, match="bank 1 appears twice"):
        _detectors(data)


def test_formatted_data_without_rounding():
    """decimals=None: the straw ends exactly (positionA / B are pixel centres, half a pixel
    inside); the default rounds them to 2 decimals."""
    detectors = _detectors(_INPUTS["measured"]())
    exact = detectors.formatted_data(decimals=None)
    rounded = detectors.formatted_data()
    for bank_id, bank in detectors.banks.items():
        half = bank.vector_along_tubes * bank.pixel_size / 2
        assert np.array_equal(np.array(exact[bank_id]["A"]), bank.positionA + half)
        assert np.array_equal(np.array(exact[bank_id]["B"]), bank.positionB - half)
        for key in "AB":
            assert rounded[bank_id][key] == [
                tuple(round(v, 2) for v in p) for p in exact[bank_id][key]
            ]


def test_the_checks_warn_about_a_moved_and_tilted_bank(capsys):
    """The tilt and distance checks against the intended geometry print warnings for a bank
    that is moved by 10 mm along the beam and tilted (the unchanged banks pass)."""
    data = copy.deepcopy(convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA))
    bank = next(b for b in data if b["bank_id"] == 3)
    keys = ("first_pixel_end", "last_pixel_end")
    y_centre = np.mean([p[k][1] for p in bank["positions"] for k in keys])
    for position in bank["positions"]:
        for key in keys:
            point = np.array(position[key], dtype=float)
            point[2] += 10.0 + 0.01 * (point[1] - y_centre)  # 10 mm farther, tilted by ~0.6 deg
            position[key] = tuple(point)
    calibration.Detectors(data, True)
    out = capsys.readouterr().out
    assert "Warning: Bank 3" in out and "tilt" in out.lower()
    assert "sample bank distance" in out
    assert all(f"Warning: Bank {b} " not in out for b in (0, 1, 2, 4, 5, 6, 7, 8))


def test_calibrate_cli_reports_a_missing_survey_file(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        _quiet(calibration.main, ["--survey-file", str(tmp_path / sf.SAM606_FILE_NAME)])
    assert error.value.code == 2
    assert "cannot read" in capsys.readouterr().err


@pytest.mark.parametrize("bank_id", [-1, 9, True, 2.0])
def test_an_unknown_bank_id_is_rejected(bank_id):
    data = _geant4_bank_data(3, (0, 15))
    data["bank_id"] = bank_id
    with pytest.raises(ValueError, match="unknown bank id"):
        _detectors([data])


@pytest.mark.parametrize("tube_id", [-1, 16, 40])
def test_a_tube_id_outside_the_front_layer_is_rejected(tube_id):
    """Bank 1 has 16 tubes per layer (ids 0-15)."""
    data = _geant4_bank_data(1, (0, 5))
    data["positions"][1]["tube_id"] = tube_id
    with pytest.raises(ValueError, match="not a front layer tube"):
        _detectors([data])


def _rotated_about(point, centre, axis, angle):
    """point rotated by angle about the line through centre along the unit vector axis."""
    v = np.asarray(point, dtype=float) - centre
    rotated = (
        v * np.cos(angle)
        + np.cross(axis, v) * np.sin(angle)
        + axis * np.dot(axis, v) * (1 - np.cos(angle))
    )
    return centre + rotated


@pytest.mark.parametrize("bank_id", [1, 2, 5, 7])
def test_imperfect_input_known_answer(bank_id):
    """Known answer for imperfect input, where the averaging steps of the algorithm matter
    (with perfect input every step is exact): the two input tubes are turned by +/- 0.5 deg
    in the bank plane (about the layer normal) and moved by +/- 1.5 mm across the tubes. The
    direction along the tubes is the bisector of the two tube directions and tube 0 the mean
    of the two predictions, so the corners are those of the unperturbed bank."""
    rows = c.TUBES_PER_LAYER[bank_id]
    expected = g4.bank_corners(bank_id)
    a0, a2, b0 = (
        np.array(p, dtype=float) for p in (expected["A"][0], expected["A"][2], expected["B"][0])
    )
    along = (a0 - b0) / np.linalg.norm(a0 - b0)  # last -> first pixel end
    across = (a2 - a0) - np.dot(a2 - a0, along) * along
    across /= np.linalg.norm(across)
    normal = np.cross(along, across)
    data = _geant4_bank_data(bank_id, (2, rows - 3))
    for tube, sign in zip(data["positions"], (1, -1)):
        first, last = np.array(tube["first_pixel_end"]), np.array(tube["last_pixel_end"])
        middle = (first + last) / 2
        for end, point in (("first_pixel_end", first), ("last_pixel_end", last)):
            moved = _rotated_about(point, middle, normal, sign * np.radians(0.5))
            tube[end] = list(moved + sign * 1.5 * across)
    corners = _detectors([data]).formatted_data(decimals=9)[bank_id]
    for end in "AB":
        assert np.allclose(corners[end], expected[end], atol=1e-6), end


def _bank_axes(bank_id):
    """(along: last -> first pixel end, across: towards the higher tube ids) of the nominal bank."""
    expected = g4.bank_corners(bank_id)
    a0, a2, b0 = (
        np.array(p, dtype=float) for p in (expected["A"][0], expected["A"][2], expected["B"][0])
    )
    along = (a0 - b0) / np.linalg.norm(a0 - b0)
    across = (a2 - a0) - np.dot(a2 - a0, along) * along
    return along, across / np.linalg.norm(across)


@pytest.mark.parametrize("bank_id", [3, 6])
def test_quality_numbers_decompose_a_known_offset(bank_id):
    """Tube b moved 1 mm along the tubes and 0.5 pitch across them (away from tube a): the
    difference of the two tube 0 predictions (a - b, in the calibration frame u, w, n; w points
    towards tube 0) is (-1 mm, +0.5 pitch, 0), and
    the measured tube spacing (perpendicular to the tubes) is the id difference + 0.5 pitch."""
    along, across = _bank_axes(bank_id)
    data = _geant4_bank_data(bank_id, (2, 9))
    tube_b = data["positions"][1]
    for end in ("first_pixel_end", "last_pixel_end"):
        tube_b[end] = list(np.array(tube_b[end]) + 1.0 * along + 0.5 * c.TUBE_GRID_SIDE * across)
    quality = _detectors([data]).banks[bank_id].quality
    assert quality["tube0_prediction_difference"] == pytest.approx(
        [-1.0, 0.5 * c.TUBE_GRID_SIDE, 0.0], abs=1e-6
    )
    assert quality["spacing_in_pitches"] == pytest.approx(7.5, abs=1e-9)


def _moved_bank_3(dz, tilt_deg):
    """REFERENCE_BANKS_DATA calibration data with bank 3 moved by dz along z and tilted by
    about tilt_deg (z grows with y)."""
    data = copy.deepcopy(convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA))
    bank = next(b for b in data if b["bank_id"] == 3)
    keys = ("first_pixel_end", "last_pixel_end")
    y_centre = np.mean([p[k][1] for p in bank["positions"] for k in keys])
    for position in bank["positions"]:
        for key in keys:
            point = np.array(position[key], dtype=float)
            point[2] += dz + np.tan(np.radians(tilt_deg)) * (point[1] - y_centre)
            position[key] = tuple(point)
    return data


@pytest.mark.parametrize(
    "dz, tilt_deg, distance_warning, tilt_warning",
    [
        (0.5, 0.0, False, False),
        (1.5, 0.0, True, False),
        (0.0, 0.05, False, False),
        (0.0, 0.15, False, True),
    ],
)
def test_the_check_tolerances(capsys, dz, tilt_deg, distance_warning, tilt_warning):
    """At the limits of the checks against the intended geometry: 1 mm for the sample - bank
    distance and 0.1 deg for the tilt."""
    calibration.Detectors(_moved_bank_3(dz, tilt_deg), True)
    out = capsys.readouterr().out
    assert ("Warning: Bank 3 sample bank distance" in out) == distance_warning
    assert ("Warning: Bank 3 tilt angle" in out) == tilt_warning
