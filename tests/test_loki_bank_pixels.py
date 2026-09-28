"""Tests of the pixel geometry derived from the 8 bank corners."""
import numpy as np
import pytest

import geant4_pixel_positions as g4
import loki_bank_pixels as lbp
from reference_banks_data import REFERENCE_BANKS_DATA

ALL_BANKS = range(lbp.NUMBER_OF_BANKS)


@pytest.fixture(scope="module")
def geant4_corners():
    return {bank_id: g4.bank_corners(bank_id) for bank_id in ALL_BANKS}


# --- straw pattern ----------------------------------------------------------

def test_straw_pattern_is_a_hexagon_around_the_centre_straw():
    pattern = lbp.STRAW_POSITIONS_PACK_FRAME
    assert pattern.shape == (7, 2)
    assert np.allclose(pattern[lbp.CENTRE_STRAW], 0.0)
    outer = np.delete(pattern, lbp.CENTRE_STRAW, axis=0)
    assert np.allclose(np.linalg.norm(outer, axis=1), 7.75)
    angles = np.sort(np.degrees(np.arctan2(outer[:, 1], outer[:, 0])) % 360)
    assert np.allclose(np.diff(angles), 60.0)


@pytest.mark.parametrize("bank_id", ALL_BANKS)
def test_straw_offsets_match_geant4(bank_id, geant4_corners):
    frame = lbp.BankFrame.from_corners(geant4_corners[bank_id], lbp.tubes_per_layer(bank_id))
    geant4 = np.array([g4.pixel_centre(g4.pixel_id(bank_id, 0, straw, 0)) for straw in range(7)])
    assert np.allclose(lbp.straw_offsets(frame), geant4 - geant4[lbp.CENTRE_STRAW], atol=1e-9)


# --- bank frame ---------------------------------------------------------------

@pytest.mark.parametrize("corner_source", ["reference", "geant4"])
def test_mirrored_banks_are_derived_from_the_corners(corner_source, geant4_corners):
    corners = REFERENCE_BANKS_DATA if corner_source == "reference" else geant4_corners
    mirrored = {b for b in ALL_BANKS
                if lbp.BankFrame.from_corners(corners[b], lbp.tubes_per_layer(b)).mirrored}
    assert mirrored == set(lbp.MIRRORED_BANK_IDS)


def test_frame_axes_are_orthonormal():
    frame = lbp.BankFrame.from_corners(REFERENCE_BANKS_DATA[5], lbp.tubes_per_layer(5))
    basis = np.array([frame.along, frame.row_axis, frame.depth_axis])
    assert np.allclose(basis @ basis.T, np.eye(3))


def test_invalid_corners_raise():
    with pytest.raises(ValueError):
        lbp.BankFrame.from_corners({"A": [(0, 0, 0)] * 3, "B": [(1, 0, 0)] * 4}, 16)
    corners = dict(REFERENCE_BANKS_DATA[1], num_tubes=65)
    with pytest.raises(ValueError):
        lbp.compute_bank_pixels(1, corners)


# --- offsets ------------------------------------------------------------------

def test_tube_order_is_layer_major():
    frame = lbp.BankFrame.from_corners(REFERENCE_BANKS_DATA[3], lbp.tubes_per_layer(3))
    tubes = lbp.tube_offsets(frame)
    rows = frame.rows
    assert tubes.shape == (4 * rows, 3)
    assert np.allclose(tubes[0], 0.0)
    assert np.allclose(tubes[rows - 1], np.subtract(REFERENCE_BANKS_DATA[3]["A"][2], REFERENCE_BANKS_DATA[3]["A"][0]))
    assert np.allclose(tubes[3 * rows], np.subtract(REFERENCE_BANKS_DATA[3]["A"][1], REFERENCE_BANKS_DATA[3]["A"][0]))


def test_pixels_are_centred_in_equal_parts_of_the_straw():
    frame = lbp.BankFrame.from_corners(REFERENCE_BANKS_DATA[2], lbp.tubes_per_layer(2))
    pixels = lbp.pixel_offsets_along_straw(frame, 4)
    distances = pixels @ frame.along
    assert np.allclose(distances, frame.straw_length * np.array([0.125, 0.375, 0.625, 0.875]))


# --- whole banks ----------------------------------------------------------------

@pytest.mark.parametrize("bank_id", ALL_BANKS)
def test_all_pixels_reproduce_geant4_exactly(bank_id, geant4_corners):
    """With corners taken from the Geant4 geometry, every pixel must agree."""
    bank = lbp.compute_bank_pixels(bank_id, geant4_corners[bank_id])
    assert np.abs(bank.positions - g4.bank_pixel_centres(bank_id)).max() < 1e-6


@pytest.mark.parametrize("pixels_per_straw", [1, 2, 512])
def test_other_pixel_numbers_reproduce_geant4(pixels_per_straw, geant4_corners):
    bank = lbp.compute_bank_pixels(6, geant4_corners[6], pixels_per_straw)
    assert np.abs(bank.positions - g4.bank_pixel_centres(6, pixels_per_straw)).max() < 1e-6


def test_reference_corners_are_close_to_geant4():
    """REFERENCE_BANKS_DATA (exact CAD) vs Geant4 (rounded CAD values): < 1 mm."""
    for bank_id in ALL_BANKS:
        bank = lbp.compute_bank_pixels(bank_id, REFERENCE_BANKS_DATA[bank_id])
        assert np.abs(bank.positions - g4.bank_pixel_centres(bank_id)).max() < 1.0


# --- pixel ids -------------------------------------------------------------------

def test_pixel_ids_are_consecutive_over_the_instrument():
    ids = np.concatenate([lbp.bank_pixel_ids(b) for b in ALL_BANKS])
    assert ids[0] == 1
    assert np.array_equal(ids, np.arange(1, 1605632 + 1))


@pytest.mark.parametrize("pixels_per_straw", [256, 512])
def test_bank_first_pixel_ids_match_geant4(pixels_per_straw):
    for bank_id in ALL_BANKS:
        assert lbp.bank_first_pixel_id(bank_id, pixels_per_straw) == g4.bank_pixel_offset(bank_id, pixels_per_straw) + 1


def test_compute_instrument_pixels_returns_all_banks_in_order():
    pixels = lbp.compute_instrument_pixels({4: REFERENCE_BANKS_DATA[4], 1: REFERENCE_BANKS_DATA[1]}, 2)
    assert list(pixels) == [1, 4]
    assert len(pixels[4].offsets) == len(pixels[4].pixel_ids) == 48 * 7 * 2


# --- pixel reference point ---------------------------------------------------------

def test_pixel_base_offsets_are_half_a_pixel_before_the_centres():
    centre = lbp.compute_bank_pixels(7, REFERENCE_BANKS_DATA[7], 4, pixel_reference=lbp.PIXEL_CENTRE)
    base = lbp.compute_bank_pixels(7, REFERENCE_BANKS_DATA[7], 4, pixel_reference=lbp.PIXEL_BASE)
    half_pixel = centre.frame.along * centre.pixel_length / 2
    assert np.allclose(base.offsets, centre.offsets - half_pixel)
    assert np.allclose(base.pixel_centres, centre.pixel_centres)
    assert np.allclose(centre.pixel_centres, centre.positions)


def test_first_pixel_base_is_the_bank_origin_in_the_centre_straw():
    base = lbp.compute_bank_pixels(0, REFERENCE_BANKS_DATA[0], pixel_reference=lbp.PIXEL_BASE)
    first_pixel_of_centre_straw = lbp.CENTRE_STRAW * base.pixels_per_straw
    assert np.allclose(base.offsets[first_pixel_of_centre_straw], 0.0)


def test_unknown_pixel_reference_raises():
    with pytest.raises(ValueError):
        lbp.compute_bank_pixels(1, REFERENCE_BANKS_DATA[1], pixel_reference="pixel_top")
