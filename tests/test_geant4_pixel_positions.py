"""Tests of the Python port of the Geant4 pixel positions."""
import os

import numpy as np
import pytest

import geant4_pixel_positions as g4
from conftest import REPO_DIR

DGCODE_PIXEL_LOG = os.path.join(REPO_DIR, "..", "dgcode_projects", "LOKI", "G4GeoLoki", "scripts", "test_pixel_position.log")


def test_port_matches_sb_loki_info_values():
    assert g4.validate() < 1e-5


@pytest.mark.skipif(not os.path.exists(DGCODE_PIXEL_LOG), reason="dgcode_projects test log not available")
def test_port_matches_dgcode_pixel_file():
    """All straws of all banks (2 pixels per straw), file values rounded to 0.1 mm."""
    data = np.loadtxt(DGCODE_PIXEL_LOG, delimiter=",")
    positions = np.vstack([g4.bank_pixel_centres(bank_id, 2) for bank_id in range(9)])
    assert np.array_equal(data[:, 0].astype(int) - 1, np.arange(len(positions)))
    assert np.abs(positions / 1000 - data[:, 1:]).max() <= 0.5e-4 + 1e-9


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
