"""Consistency of the shared geometry constants."""

import numpy as np
import pytest

from loki_geometry import constants as c
from loki_geometry.reference_data import REFERENCE_BANKS_DATA


def test_per_bank_tables_have_one_entry_per_bank():
    for table in (c.BANK_NAMES, c.PACKS_PER_BANK, c.STRAW_LENGTHS, c.BANK_DISTANCES,
                  c.BANK_POSITION_ANGLES, c.BANK_FACE_ANGLES, c.BANK_OFFSETS, c.BANK_TILT_ANGLES):
        assert len(table) == c.NUMBER_OF_BANKS


def test_total_number_of_pixels():
    total = sum(c.TUBES_PER_BANK) * c.STRAWS_PER_TUBE * c.DEFAULT_PIXELS_PER_STRAW
    assert total == 1_605_632


def test_tilt_angles():
    assert c.BANK_TILT_ANGLES == pytest.approx((0.0, 5.7, 4.5, 5.7, 4.5, 17.5, 10.6, 17.5, 10.6))


def test_position_sides():
    assert [c.bank_position_side(b) for b in c.BANK_IDS] == [1, 1, 1, -1, -1, 1, 1, -1, -1]


@pytest.mark.parametrize("bank_id", c.BANK_IDS)
def test_front_to_back_shift_matches_the_reference_corners(bank_id):
    """A[1] - A[0] (front to back layer) has the lengths derived from the tube grid
    (within 0.1 mm: the CAD values differ slightly from the nominal grid)."""
    corners = REFERENCE_BANKS_DATA[bank_id]
    shift = np.subtract(corners["A"][1], corners["A"][0])
    row = np.subtract(corners["A"][2], corners["A"][0])
    row /= np.linalg.norm(row)
    in_plane = abs(np.dot(shift, row))
    assert in_plane == pytest.approx(c.FRONT_TO_BACK_IN_PLANE, abs=0.1)
    assert np.sqrt(np.dot(shift, shift) - in_plane**2) == pytest.approx(c.FRONT_TO_BACK_NORMAL, abs=0.1)


@pytest.mark.parametrize("bank_id", c.BANK_IDS)
def test_straw_lengths_match_the_reference_corners(bank_id):
    corners = REFERENCE_BANKS_DATA[bank_id]
    length = np.linalg.norm(np.subtract(corners["B"][0], corners["A"][0]))
    assert length == pytest.approx(c.STRAW_LENGTHS[bank_id], abs=0.01)
