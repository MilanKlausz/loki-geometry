"""Tests of the bank calibration (8 corners from 2 tubes per bank), the survey
conversion and the nominal corner reconstruction."""

import numpy as np
import pytest

from loki_geometry import constants as c
from loki_geometry import geant4 as g4
from loki_geometry import survey
from loki_geometry.calibration import Detectors
from loki_geometry.geant4_example_data import example_calibration_data
from loki_geometry.nominal_corners import Bank as NominalBank
from loki_geometry.reference_data import (
    REFERENCE_BANKS_DATA,
    convert_reference_data_to_calibration_data,
)


@pytest.fixture(scope="module")
def reference_detectors():
    return Detectors(
        convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA), True, run_checks=False
    )


@pytest.fixture(scope="module")
def measured_detectors():
    return Detectors(survey.get_measured_calibration_data(verbose=False), True, run_checks=False)


# --- self tests ----------------------------------------------------------------


def test_reference_data_reproduces_itself(reference_detectors):
    corners = reference_detectors.formatted_data()
    for bank_id in c.BANK_IDS:
        for end in "AB":
            assert np.allclose(corners[bank_id][end], REFERENCE_BANKS_DATA[bank_id][end], atol=0.1)


@pytest.mark.parametrize("use_pixel_centre_position", [True, False])
def test_straw_end_output_does_not_depend_on_the_pixel_centre_option(
    use_pixel_centre_position, reference_detectors
):
    detectors = Detectors(
        convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA),
        use_pixel_centre_position,
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


def test_geant4_example_data_is_up_to_date():
    """example_calibration_data = first/last pixel centres of the centre straw (Geant4 port)."""
    for bank in example_calibration_data:
        for tube in bank["positions"]:
            first = g4.pixel_centre(
                g4.pixel_id(bank["bank_id"], tube["tube_id"], c.CENTRE_STRAW, 0)
            )
            last = g4.pixel_centre(
                g4.pixel_id(
                    bank["bank_id"], tube["tube_id"], c.CENTRE_STRAW, c.DEFAULT_PIXELS_PER_STRAW - 1
                )
            )
            assert np.allclose(tube["first_pixel_end"], first, atol=1e-5)
            assert np.allclose(tube["last_pixel_end"], last, atol=1e-5)


def test_nominal_corners_match_the_reference_data():
    for bank_id in c.BANK_IDS:
        bank = NominalBank(bank_id)
        assert np.allclose(bank.positionA, REFERENCE_BANKS_DATA[bank_id]["A"], atol=1.0)
        assert np.allclose(bank.positionB, REFERENCE_BANKS_DATA[bank_id]["B"], atol=1.0)


# --- measured data -----------------------------------------------------------


def test_survey_file_covers_banks_1_to_8():
    data = survey.get_measured_calibration_data(verbose=False)
    assert [bank["bank_id"] for bank in data] == list(range(1, 9))
    for bank in data:
        for tube in bank["positions"]:
            assert tube["distance"] == pytest.approx(c.STRAW_LENGTHS[bank["bank_id"]])


def test_survey_tube_ids_agree_with_their_spacing():
    points = {
        label: survey.survey_to_nexus_frame(p, 0.0)
        for label, p in survey.read_measured_points().items()
    }
    points = survey.fix_bank5_mislabelled_tube(points, c.TUBE_GRID_SIDE)
    banks = survey.group_points_into_tubes(points, c.VERTICAL_BANK_IDS)
    for tube_a, tube_b in banks.values():
        mid_a = (tube_a["end_A"] + tube_a["end_B"]) / 2
        mid_b = (tube_b["end_A"] + tube_b["end_B"]) / 2
        along = (tube_a["end_B"] - tube_a["end_A"]) / np.linalg.norm(
            tube_a["end_B"] - tube_a["end_A"]
        )
        across = (mid_b - mid_a) - np.dot(mid_b - mid_a, along) * along
        steps = np.linalg.norm(across) / c.TUBE_GRID_SIDE
        assert steps == pytest.approx(
            tube_b["tube_id"] - tube_a["tube_id"], abs=survey.TUBE_ID_TOLERANCE
        )


def test_find_effective_tube_ends_shortens_and_stretches_symmetrically():
    first, last = np.array([10.0, 0, 0]), np.array([-10.0, 0, 0])
    for length in (5.0, 30.0):
        a, b, input_length, excess = survey.find_effective_tube_ends(first, last, length)
        assert np.linalg.norm(b - a) == pytest.approx(length)
        assert np.allclose((a + b) / 2, 0.0)
        assert input_length == pytest.approx(20.0)
        assert excess == pytest.approx(20.0 - length)


def test_measured_banks_reproduce_their_input_tubes(measured_detectors):
    for bank in measured_detectors.banks.values():
        worst = max(
            np.linalg.norm(residual) for _, residual, _ in bank.quality["reproduction_residuals"]
        )
        assert worst < 2.0


def test_measured_front_face_distance_shifts_with_the_sample_to_window_distance(measured_detectors):
    shifted = Detectors(
        survey.get_measured_calibration_data(sample_to_window_distance=110.0, verbose=False),
        True,
        run_checks=False,
    )
    for bank_id, bank in measured_detectors.banks.items():
        shift = shifted.banks[bank_id].front_face_centre - bank.front_face_centre
        assert np.allclose(shift, (0.0, 0.0, 110.0 - survey.SAMPLE_TO_TANK_WINDOW_DISTANCE))
