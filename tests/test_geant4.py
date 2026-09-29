"""Tests of the Python port of the Geant4 pixel positions."""

import numpy as np
import pytest

from loki_geometry import geant4 as g4


def test_port_matches_sb_loki_info_values():
    assert g4.validate() < 1e-6  # the reference values have 6 decimals


@pytest.mark.parametrize("bank_id", [0, 5])
def test_bank_pixel_centres_equals_scalar_function(bank_id):
    centres = g4.bank_pixel_centres(bank_id)
    offset = g4.bank_pixel_offset(bank_id)
    for local in (0, 1, 255, 256, 777, len(centres) - 1):
        assert np.allclose(centres[local], g4.pixel_centre(offset + local), atol=1e-9)


def test_bank_corners_are_straw_ends():
    corners = g4.bank_corners(1)
    length = np.linalg.norm(np.subtract(corners["B"][0], corners["A"][0]))
    assert length == pytest.approx(g4.STRAW_LENGTH[1])
    assert corners["num_tubes"] == 64
