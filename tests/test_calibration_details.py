"""Detailed tests of the calibration chain (survey -> calibration data -> 8 corners)
and of the nominal corner reconstruction.

Covers: invariances of the algorithm (rigid motions, input order, I/II labels),
exact recovery of the Geant4 corners from arbitrary tube pairs, frame conversion
and handedness, the survey tube identification, the sign of the back layer shift,
the output format, the input checks and the command line tools."""

import contextlib
import copy
import io
import re

import numpy as np
import pytest

from loki_geometry import constants as c
from loki_geometry import geant4 as g4
from loki_geometry import nominal_corners, survey
from loki_geometry.calibration import Detectors
from loki_geometry.geant4_example_data import example_calibration_data
from loki_geometry.reference_data import (
    REFERENCE_BANKS_DATA,
    convert_reference_data_to_calibration_data,
)


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


# --- exactness of the algorithm ---------------------------------------------------


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


@pytest.mark.parametrize(
    "data",
    [
        example_calibration_data,
        convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA),
        _quiet(survey.get_measured_calibration_data, verbose=False),
    ],
    ids=["example", "reference", "measured"],
)
def test_normal_points_away_from_the_sample_and_the_back_layer_is_behind(data):
    detectors = _detectors(data)
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


# --- output format -----------------------------------------------------------------


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


# --- survey conversion ----------------------------------------------------------------


def test_measured_banks_are_on_the_expected_side_of_the_beam():
    """Labels U/D/L/R vs the frame: y > 0 up, x > 0 left (beam's view), and the
    front bank offsets along the tubes have the nominal signs."""
    detectors = _detectors(_quiet(survey.get_measured_calibration_data, verbose=False))
    expected_side = {1: (1, +1), 2: (0, +1), 3: (1, -1), 4: (0, -1),
                     5: (1, +1), 6: (0, +1), 7: (1, -1), 8: (0, -1)}  # fmt: skip
    for bank_id, (axis, sign) in expected_side.items():
        centre = detectors.banks[bank_id].front_face_centre
        assert np.sign(centre[axis]) == sign
        assert centre[2] > 0
    for bank_id in (5, 6, 7, 8):
        _, _, offset = detectors.section_parameters(bank_id)
        _, _, nominal = detectors.nominal_section_parameters(bank_id)
        assert offset == pytest.approx(nominal, abs=10.0)


def test_measured_first_pixel_ends_are_at_plus_x_or_minus_y():
    data = _quiet(survey.get_measured_calibration_data, verbose=False)
    for bank in data:
        for tube in bank["positions"]:
            first, last = np.array(tube["first_pixel_end"]), np.array(tube["last_pixel_end"])
            if bank["bank_id"] in c.VERTICAL_BANK_IDS:
                assert first[1] < last[1]
            else:
                assert first[0] > last[0]


# --- nominal corners and constants ------------------------------------------------------


@pytest.mark.parametrize("bank_id", c.BANK_IDS)
def test_nominal_corners_match_geant4(bank_id):
    """Much tighter than the 1 mm vs REFERENCE_BANKS_DATA: the same rounded
    Geant4 table values give the same corners (up to the 2-decimal rounding)."""
    bank = nominal_corners.Bank(bank_id)
    expected = g4.bank_corners(bank_id)
    assert np.allclose(bank.positionA, expected["A"], atol=0.006)
    assert np.allclose(bank.positionB, expected["B"], atol=0.006)


# --- CLI -------------------------------------------------------------------------------


@pytest.mark.parametrize("data", ["reference", "example"])
def test_calibrate_cli_self_tests_run(monkeypatch, capsys, data):
    from loki_geometry import calibration

    assert calibration.main(["--data", data, "--print"]) == 0
    out = capsys.readouterr().out
    assert "All bank tilt angles agree" in out
    assert "All bank sample-bank distances agree" in out
    assert "'loki_detector_8'" in out


def test_calibrate_cli_passes_the_sample_to_window_distance(monkeypatch, capsys):
    from loki_geometry import calibration

    calibration.main(["--sample-to-window-distance", "119", "--verbose"])
    assert "using sample to tank window distance 119.0 mm" in capsys.readouterr().out


@pytest.mark.parametrize("data", ["reference", "example"])
def test_calibrate_cli_rejects_the_distance_without_measured_data(capsys, data):
    from loki_geometry import calibration

    with pytest.raises(SystemExit) as error:
        calibration.main(["--data", data, "--sample-to-window-distance", "119"])
    assert error.value.code == 2
    assert "only applies to --data measured" in capsys.readouterr().err


def test_calibrate_cli_reference_self_test_compares_with_reference_data(capsys):
    from loki_geometry import calibration

    assert calibration.main(["--data", "reference"]) == 0
    out = capsys.readouterr().out
    assert "Arrays differ" not in out
    worst = float(re.search(r"largest difference from REFERENCE_BANKS_DATA ([\d.]+) mm", out)[1])
    assert 0.0 < worst < 0.07  # the rounding of the CAD values, not 0 by construction


# --- input checks and fixed problems ----------------------------------------------------


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


def test_verify_positions_against_reference_data_accepts_the_reference_itself():
    detectors = _detectors(convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        worst = detectors.verify_positions_against_reference_data(tolerance=0.07)
    assert "Arrays differ" not in out.getvalue()
    assert worst < 0.07


def test_nominal_corners_cli_honours_several_bank_ids(capsys):
    assert nominal_corners.cli(["1", "2"]) == 0
    assert capsys.readouterr().out.count("Comparing") == 4


def test_nominal_corners_cli_rejects_unknown_bank_ids(capsys):
    with pytest.raises(SystemExit) as error:
        nominal_corners.cli(["9"])
    assert error.value.code == 2


def test_survey_is_quiet_without_verbose(capsys):
    survey.get_measured_calibration_data(verbose=False)
    assert capsys.readouterr().out == ""
