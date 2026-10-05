"""Tests of the intended corner positions (REFERENCE_BANKS_DATA) and their
conversion to calibration data."""

import numpy as np
import pytest

from loki_geometry import bank_pixels as lbp
from loki_geometry import constants as c
from loki_geometry.reference_data import (
    REFERENCE_BANKS_DATA,
    convert_reference_data_to_calibration_data,
)

ALL_BANKS = range(c.NUMBER_OF_BANKS)


@pytest.mark.parametrize("bank_id", ALL_BANKS)
def test_reference_corners_form_a_regular_bank(bank_id):
    """The model uses only A[0], A[1], A[2], B[0]; the other corners must agree with it."""
    a = np.array(REFERENCE_BANKS_DATA[bank_id]["A"], dtype=float)
    b = np.array(REFERENCE_BANKS_DATA[bank_id]["B"], dtype=float)
    assert np.allclose(a[3], a[1] + a[2] - a[0], atol=0.02)  # parallelogram
    assert np.allclose(b - a, b[0] - a[0], atol=1e-9)  # parallel tubes of equal length
    assert np.linalg.norm(b[0] - a[0]) == pytest.approx(c.STRAW_LENGTHS[bank_id])
    assert set(REFERENCE_BANKS_DATA[bank_id]) == {"A", "B", "num_tubes"}
    assert REFERENCE_BANKS_DATA[bank_id]["num_tubes"] == c.TUBES_PER_BANK[bank_id]
    frame = lbp.BankFrame.from_corners(REFERENCE_BANKS_DATA[bank_id], lbp.tubes_per_layer(bank_id))
    assert np.linalg.norm(frame.row_step) == pytest.approx(c.TUBE_GRID_SIDE, abs=0.01)
    assert np.linalg.norm(frame.layer_step) == pytest.approx(c.TUBE_GRID_BASE, abs=0.01)
    # the layer shift within the tube plane (13.45 deg parallelogram; the CAD
    # values of the tilted banks give 18.89-18.90 mm instead of 18.84 mm)
    in_plane = abs(np.dot(frame.layer_step, frame.row_axis)) * (c.LAYERS_PER_BANK - 1)
    assert in_plane == pytest.approx(c.FRONT_TO_BACK_IN_PLANE, abs=0.1)


def test_pixel_ends_point_in_the_icd_direction():
    """Pixel ids increase along -x (horizontal banks) and +y (vertical banks)."""
    for bank_id in ALL_BANKS:
        frame = lbp.BankFrame.from_corners(
            REFERENCE_BANKS_DATA[bank_id], lbp.tubes_per_layer(bank_id)
        )
        expected = (0, 1, 0) if bank_id in c.VERTICAL_BANK_IDS else (-1, 0, 0)
        assert np.allclose(frame.along, expected)


def test_conversion_to_calibration_data():
    calibration = convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA)
    assert [bank["bank_id"] for bank in calibration] == list(ALL_BANKS)
    for bank in calibration:
        bank_id = bank["bank_id"]
        reference = REFERENCE_BANKS_DATA[bank_id]
        first, last = bank["positions"]
        assert first["tube_id"] == 0
        assert last["tube_id"] == lbp.tubes_per_layer(bank_id) - 1
        assert first["first_pixel_end"] == list(reference["A"][0])
        assert first["last_pixel_end"] == list(reference["B"][0])
        assert last["first_pixel_end"] == list(reference["A"][2])
        assert last["last_pixel_end"] == list(reference["B"][2])
