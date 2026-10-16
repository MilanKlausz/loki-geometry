"""Tests of the pixel geometry core (bank_pixels.py): from the 8 corners of a bank
to the detector numbers and pixel offsets, and the checks of the corners."""

import contextlib
import io

import numpy as np
import pytest

from loki_geometry import bank_pixels as lbp
from loki_geometry import calibration, survey
from loki_geometry import constants as c
from loki_geometry import geant4 as g4
from loki_geometry.reference_data import (
    REFERENCE_BANKS_DATA,
    convert_reference_data_to_calibration_data,
)
from loki_geometry.survey_files import SAM606

ALL_BANKS = range(lbp.NUMBER_OF_BANKS)


@pytest.fixture(scope="module")
def geant4_corners():
    return {bank_id: g4.bank_corners(bank_id) for bank_id in ALL_BANKS}


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


@pytest.mark.parametrize("corner_source", ["reference", "geant4"])
def test_mirrored_banks_are_derived_from_the_corners(corner_source, geant4_corners):
    corners = REFERENCE_BANKS_DATA if corner_source == "reference" else geant4_corners
    mirrored = {
        b
        for b in ALL_BANKS
        if lbp.BankFrame.from_corners(corners[b], lbp.tubes_per_layer(b)).mirrored
    }
    assert mirrored == set(lbp.MIRRORED_BANK_IDS)


def test_frame_axes_are_orthonormal():
    frame = lbp.BankFrame.from_corners(REFERENCE_BANKS_DATA[5], lbp.tubes_per_layer(5))
    basis = np.array([frame.along, frame.row_axis, frame.depth_axis])
    assert np.allclose(basis @ basis.T, np.eye(3))


def test_tube_order_is_layer_major():
    frame = lbp.BankFrame.from_corners(REFERENCE_BANKS_DATA[3], lbp.tubes_per_layer(3))
    tubes = lbp.tube_offsets(frame)
    rows = frame.rows
    assert tubes.shape == (4 * rows, 3)
    assert np.allclose(tubes[0], 0.0)
    assert np.allclose(
        tubes[rows - 1],
        np.subtract(REFERENCE_BANKS_DATA[3]["A"][2], REFERENCE_BANKS_DATA[3]["A"][0]),
    )
    assert np.allclose(
        tubes[3 * rows],
        np.subtract(REFERENCE_BANKS_DATA[3]["A"][1], REFERENCE_BANKS_DATA[3]["A"][0]),
    )


def test_pixels_are_centred_in_equal_parts_of_the_straw():
    frame = lbp.BankFrame.from_corners(REFERENCE_BANKS_DATA[2], lbp.tubes_per_layer(2))
    pixels = lbp.pixel_offsets_along_straw(frame, 4)
    distances = pixels @ frame.along
    assert np.allclose(distances, frame.straw_length * np.array([0.125, 0.375, 0.625, 0.875]))


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


def test_pixel_ids_are_consecutive_over_the_instrument():
    ids = np.concatenate([lbp.bank_pixel_ids(b) for b in ALL_BANKS])
    assert ids[0] == 1
    assert np.array_equal(ids, np.arange(1, 1605632 + 1))


@pytest.mark.parametrize("pixels_per_straw", [256, 512])
def test_bank_first_pixel_ids_match_geant4(pixels_per_straw):
    for bank_id in ALL_BANKS:
        assert (
            lbp.bank_first_pixel_id(bank_id, pixels_per_straw)
            == g4.bank_pixel_offset(bank_id, pixels_per_straw) + 1
        )


def test_compute_instrument_pixels_returns_all_banks_in_order():
    pixels = lbp.compute_instrument_pixels(
        {4: REFERENCE_BANKS_DATA[4], 1: REFERENCE_BANKS_DATA[1]}, 2
    )
    assert list(pixels) == [1, 4]
    assert len(pixels[4].offsets) == len(pixels[4].pixel_ids) == 48 * 7 * 2


def test_pixel_base_offsets_are_half_a_pixel_before_the_centres():
    centre = lbp.compute_bank_pixels(
        7, REFERENCE_BANKS_DATA[7], 4, pixel_reference=lbp.PIXEL_CENTRE
    )
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


@pytest.mark.parametrize("n", [1, 3])
def test_geant4_corners_reproduce_all_banks_pixel_by_pixel(n):
    """bank_corners -> bank_pixels reproduces every Geant4 pixel centre and id of
    ALL banks (the existing tests only cover banks 0, 2, 6, 7 with 2 px), including
    the 1-pixel-per-straw special case of bank_corners."""
    corners = {b: g4.bank_corners(b, n) for b in c.BANK_IDS}
    pixels = lbp.compute_instrument_pixels(corners, n, pixel_reference=lbp.PIXEL_CENTRE)
    for b in c.BANK_IDS:
        first = g4.bank_pixel_offset(b, n) + 1
        assert np.array_equal(
            pixels[b].pixel_ids, np.arange(first, first + len(pixels[b].pixel_ids))
        )
        assert np.abs(pixels[b].positions - g4.bank_pixel_centres(b, n)).max() < 1e-6


def corner_tube_ids(bank_id):
    """Tube ids of the corners A[0..3] / B[0..3]: front top, back top, front bottom, back bottom."""
    rows = lbp.tubes_per_layer(bank_id)
    return (0, 3 * rows, rows - 1, 4 * rows - 1)


@pytest.mark.parametrize("bank_id", ALL_BANKS)
def test_calibration_data_tubes_are_the_centre_straws_of_the_pixel_model(bank_id):
    """The two tubes given to the calibration are the tubes with these ids in the pixel model."""
    bank = lbp.compute_bank_pixels(
        bank_id, REFERENCE_BANKS_DATA[bank_id], 4, pixel_reference=lbp.PIXEL_BASE
    )
    calibration = convert_reference_data_to_calibration_data(
        {bank_id: REFERENCE_BANKS_DATA[bank_id]}
    )
    offsets = bank.positions.reshape(-1, 7, 4, 3)
    for tube in calibration[0]["positions"]:
        centre_straw = offsets[tube["tube_id"], lbp.CENTRE_STRAW]
        assert np.allclose(centre_straw[0], tube["first_pixel_end"], atol=1e-9)
        last_end = centre_straw[-1] + bank.frame.along * bank.pixel_length
        assert np.allclose(last_end, tube["last_pixel_end"], atol=1e-9)


@pytest.mark.parametrize("bank_id", ALL_BANKS)
def test_corner_tubes_reproduce_all_eight_corners(bank_id):
    """Tube id = layer * rows + row: the corner tubes start at A[i] and end at B[i]
    (also A[3] and B[1..3], which the model does not use)."""
    n = 4
    bank = lbp.compute_bank_pixels(
        bank_id, REFERENCE_BANKS_DATA[bank_id], n, pixel_reference=lbp.PIXEL_BASE
    )
    positions = bank.positions.reshape(-1, 7, n, 3)
    corners = REFERENCE_BANKS_DATA[bank_id]
    for index, tube in enumerate(corner_tube_ids(bank_id)):
        centre_straw = positions[tube, lbp.CENTRE_STRAW]
        assert np.allclose(centre_straw[0], corners["A"][index], atol=0.02)
        assert np.allclose(
            centre_straw[-1] + bank.frame.along * bank.pixel_length, corners["B"][index], atol=0.02
        )


@pytest.mark.parametrize("pixels_per_straw", [3, 256, 512])
def test_pixel_ids_and_offsets_follow_the_icd_formula(pixels_per_straw):
    """pixel = 1 + pos + (straw_offset[bank] + tube * 7 + straw) * N, and the offset of
    that pixel is tube + straw + (pos + 0.5) * L / N."""
    rng = np.random.default_rng(1)
    straw_offset = 0
    for bank_id in ALL_BANKS:
        bank = lbp.compute_bank_pixels(bank_id, REFERENCE_BANKS_DATA[bank_id], pixels_per_straw)
        tubes = lbp.tube_offsets(bank.frame)
        straws = lbp.straw_offsets(bank.frame)
        for _ in range(20):
            tube = rng.integers(c.TUBES_PER_BANK[bank_id])
            straw = rng.integers(7)
            pos = rng.integers(pixels_per_straw)
            index = (tube * 7 + straw) * pixels_per_straw + pos
            expected_id = 1 + pos + (straw_offset + tube * 7 + straw) * pixels_per_straw
            assert bank.pixel_ids[index] == expected_id
            expected = (
                tubes[tube] + straws[straw] + bank.frame.along * (pos + 0.5) * bank.pixel_length
            )
            assert np.allclose(bank.offsets[index], expected, atol=1e-9)
        straw_offset += c.TUBES_PER_BANK[bank_id] * 7


@pytest.mark.parametrize("normal, mirrored", [(3, 1), (4, 2), (7, 5), (8, 6)])
def test_upside_down_banks_have_mirrored_straw_patterns(normal, mirrored):
    """In the (row, depth) frame the straw pattern of an upside down bank is the
    mirror image (row -> -row) of that of the corresponding normal bank."""

    def pattern(bank_id):
        frame = lbp.BankFrame.from_corners(
            REFERENCE_BANKS_DATA[bank_id], lbp.tubes_per_layer(bank_id)
        )
        straws = lbp.straw_offsets(frame)
        return np.column_stack([straws @ frame.row_axis, straws @ frame.depth_axis]), frame

    normal_pattern, normal_frame = pattern(normal)
    mirrored_pattern, mirrored_frame = pattern(mirrored)
    assert not normal_frame.mirrored and mirrored_frame.mirrored
    assert np.allclose(normal_pattern, lbp.STRAW_POSITIONS_PACK_FRAME)
    assert np.allclose(mirrored_pattern, normal_pattern * [-1, 1])
    # straws are perpendicular to the straw axis, and on a 7.75 mm hexagon
    for frame in (normal_frame, mirrored_frame):
        assert np.allclose(lbp.straw_offsets(frame) @ frame.along, 0.0)
    assert np.allclose(
        np.linalg.norm(np.delete(mirrored_pattern, lbp.CENTRE_STRAW, axis=0), axis=1), 7.75
    )


def test_depth_axis_points_away_from_the_sample():
    """Layer 0 (tube ids 0 .. rows-1) is the front layer, facing the sample."""
    for bank_id in ALL_BANKS:
        frame = lbp.BankFrame.from_corners(
            REFERENCE_BANKS_DATA[bank_id], lbp.tubes_per_layer(bank_id)
        )
        front_centre = frame.origin + frame.row_step * (frame.rows - 1) / 2
        assert np.dot(frame.depth_axis, front_centre) > 0
        assert np.linalg.norm(front_centre + frame.layer_step) > np.linalg.norm(front_centre)


@pytest.mark.parametrize("pixel_reference", lbp.PIXEL_REFERENCES)
@pytest.mark.parametrize("pixels_per_straw", [1, 3, 256])
def test_pixel_offsets_cover_the_straw_exactly(pixel_reference, pixels_per_straw):
    """No half-pixel shift: the pixel centres are symmetric around the straw middle."""
    bank = lbp.compute_bank_pixels(
        2, REFERENCE_BANKS_DATA[2], pixels_per_straw, pixel_reference=pixel_reference
    )
    along = bank.pixel_centres.reshape(-1, 7, pixels_per_straw, 3) @ bank.frame.along
    start = (bank.frame.origin + lbp.tube_offsets(bank.frame)) @ bank.frame.along
    along = along - start[:, None, None]
    half = bank.pixel_length / 2
    assert np.allclose(along[..., 0], half)
    assert np.allclose(along[..., -1], bank.frame.straw_length - half)
    assert np.allclose(along + along[..., ::-1], bank.frame.straw_length)


@pytest.fixture(scope="module")
def measured_corners():
    """Corners calibrated from the package survey (all 9 banks)."""
    with contextlib.redirect_stdout(io.StringIO()):
        data = survey.get_measured_calibration_data(SAM606.path, verbose=False)
        detectors = calibration.Detectors(data, True, run_checks=False)
    return detectors.formatted_data()


def _handedness(frame):
    return np.linalg.det(np.array([frame.along, frame.row_axis, frame.depth_axis]))


def test_measured_corners_pass_the_pixel_model_checks(measured_corners):
    """The whole survey -> calibration -> pixel chain: the rounded (2 decimals)
    calibrated corners pass the corner-consistency and mirrored-bank checks, and
    give the nominal tube grid and straw lengths."""
    assert sorted(measured_corners) == list(range(9))
    pixels = lbp.compute_instrument_pixels(measured_corners, 2)
    for bank_id, bank in pixels.items():
        frame = bank.frame
        corners = measured_corners[bank_id]
        a, b = np.array(corners["A"]), np.array(corners["B"])
        # rounding to 0.01 mm gives at most ~0.02 mm, well inside CORNER_TOLERANCE
        assert np.abs(a[3] - (a[1] + a[2] - a[0])).max() <= 0.02 + 1e-9
        assert np.abs((b - a) - (b[0] - a[0])).max() <= 0.02 + 1e-9
        assert frame.mirrored == (bank_id in lbp.MIRRORED_BANK_IDS)
        assert frame.straw_length == pytest.approx(c.STRAW_LENGTHS[bank_id], abs=0.01)
        assert np.linalg.norm(frame.row_step) == pytest.approx(c.TUBE_GRID_SIDE, abs=0.01)
        assert np.linalg.norm(frame.layer_step) == pytest.approx(c.TUBE_GRID_BASE, abs=0.01)
        assert _handedness(frame) == pytest.approx(1.0)
        assert bank.pixel_ids[0] == lbp.bank_first_pixel_id(bank_id, 2)


@pytest.mark.parametrize("source", ["reference", "geant4"])
def test_bank_frames_are_right_handed(source):
    """(along, row_axis, depth_axis) is right-handed for every correctly ordered
    bank (the check behind rejecting swapped corners)."""
    for bank_id in c.BANK_IDS:
        corners = (
            REFERENCE_BANKS_DATA[bank_id] if source == "reference" else g4.bank_corners(bank_id)
        )
        frame = lbp.BankFrame.from_corners(corners, lbp.tubes_per_layer(bank_id))
        assert _handedness(frame) == pytest.approx(1.0)


def test_geant4_corners_rounded_like_formatted_data_stay_within_tolerance():
    """Corners rounded to 0.01 mm (formatted_data default) are accepted for every bank
    and move the pixels by at most ~0.02 mm."""
    for bank_id in c.BANK_IDS:
        exact = g4.bank_corners(bank_id)
        rounded = {end: [tuple(round(v, 2) for v in p) for p in exact[end]] for end in "AB"}
        bank = lbp.compute_bank_pixels(bank_id, rounded, 4)
        assert np.abs(bank.positions - g4.bank_pixel_centres(bank_id, 4)).max() < 0.03


def test_invalid_corners_raise():
    with pytest.raises(ValueError, match="4 'A' and 4 'B' points"):
        lbp.BankFrame.from_corners({"A": [(0, 0, 0)] * 3, "B": [(1, 0, 0)] * 4}, 16)
    corners = dict(REFERENCE_BANKS_DATA[1], num_tubes=65)
    with pytest.raises(ValueError, match="num_tubes 65"):
        lbp.compute_bank_pixels(1, corners)


def _moved_corner(bank_id, end, index, shift_mm):
    corners = {
        "A": list(REFERENCE_BANKS_DATA[bank_id]["A"]),
        "B": list(REFERENCE_BANKS_DATA[bank_id]["B"]),
    }
    corners[end][index] = tuple(np.add(corners[end][index], (0.0, 0.0, shift_mm)))
    return corners


@pytest.mark.parametrize("end, index", [("A", 3), ("B", 1), ("B", 2), ("B", 3)])
def test_inconsistent_corners_are_rejected(end, index):
    """Only A[0], A[1], A[2] and B[0] define the model; the others must agree."""
    lbp.compute_bank_pixels(3, _moved_corner(3, end, index, 0.05), 2)  # rounding is fine
    with pytest.raises(ValueError, match="inconsistent corners"):
        lbp.compute_bank_pixels(3, _moved_corner(3, end, index, 1.0), 2)


def test_corners_of_a_mirrored_bank_for_a_normal_bank_are_rejected():
    """Corners in the wrong order give the wrong straw pattern: bank 1 (mirrored)
    corners used for bank 3 (normal) are detected."""
    corners = dict(REFERENCE_BANKS_DATA[1])
    corners.pop("num_tubes", None)
    with pytest.raises(ValueError, match="mirrored"):
        lbp.compute_bank_pixels(3, corners, 2)


@pytest.mark.parametrize("bank_id", [0, 1, 6])
def test_swapped_pixel_ends_are_rejected(bank_id):
    corners = REFERENCE_BANKS_DATA[bank_id]
    with pytest.raises(ValueError, match="left-handed frame"):
        lbp.compute_bank_pixels(bank_id, {"A": corners["B"], "B": corners["A"]}, 2)


@pytest.mark.parametrize("bank_id", [0, 3, 5])
def test_swapped_back_top_and_front_bottom_corners_are_rejected(bank_id):
    corners = REFERENCE_BANKS_DATA[bank_id]
    swapped = {end: [corners[end][i] for i in (0, 2, 1, 3)] for end in "AB"}
    with pytest.raises(ValueError, match="tube pitches"):  # the pitch check catches it first
        lbp.compute_bank_pixels(bank_id, swapped, 2)


@pytest.mark.parametrize("source, target", [(3, 7), (0, 3), (1, 5)])
def test_corners_of_another_bank_with_the_same_mirroring_are_rejected(source, target):
    corners = {end: REFERENCE_BANKS_DATA[source][end] for end in "AB"}
    with pytest.raises(ValueError, match="tube pitches|straw length"):
        lbp.compute_bank_pixels(target, corners, 2)


def _stretched(bank_id, extra_mm):
    """REFERENCE_BANKS_DATA corners of a bank with the straws longer by extra_mm (both ends)."""
    a = np.array(REFERENCE_BANKS_DATA[bank_id]["A"], dtype=float)
    b = np.array(REFERENCE_BANKS_DATA[bank_id]["B"], dtype=float)
    along = (b[0] - a[0]) / np.linalg.norm(b[0] - a[0])
    return {"A": a - 0.5 * extra_mm * along, "B": b + 0.5 * extra_mm * along}


@pytest.mark.parametrize("bank_id", [2, 3, 5])
def test_corners_with_the_wrong_straw_length_are_rejected(bank_id):
    """Untrimmed tube ends (+30 mm), corners at the pixel centres instead of the straw ends
    (shorter by one pixel, here 512 pixels per straw) and 0.2 mm too long straws are
    rejected; 0.05 mm (rounding) is accepted."""
    for extra in (30.0, -c.STRAW_LENGTHS[bank_id] / 512, 0.2, -0.2):
        with pytest.raises(ValueError, match="straw length"):
            lbp.compute_bank_pixels(bank_id, _stretched(bank_id, extra), 2)
    lbp.compute_bank_pixels(bank_id, _stretched(bank_id, 0.05), 2)


def test_nan_corner_is_rejected():
    corners = {end: [list(p) for p in REFERENCE_BANKS_DATA[3][end]] for end in "AB"}
    corners["A"][0][2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        lbp.compute_bank_pixels(3, corners, 2)


def test_negative_bank_id_is_rejected():
    with pytest.raises(ValueError, match="unknown bank id"):
        lbp.compute_bank_pixels(-1, REFERENCE_BANKS_DATA[8], 2)


@pytest.mark.parametrize("pixels_per_straw", [0, -2, 2.5])
def test_compute_bank_pixels_rejects_invalid_pixels_per_straw(pixels_per_straw):
    with pytest.raises(ValueError, match="positive integer"):
        lbp.compute_bank_pixels(3, REFERENCE_BANKS_DATA[3], pixels_per_straw)


# --- bank origin: first pixel end (corner A[0]) or centre of mass -------------------


@pytest.mark.parametrize("pixel_reference", lbp.PIXEL_REFERENCES)
@pytest.mark.parametrize(
    "bank_ids, pixels_per_straw",
    [(ALL_BANKS, 3), ((2,), 256)],  # all banks with few pixels, one small bank in full
    ids=["all-banks-3px", "bank2-256px"],
)
def test_both_bank_origins_give_the_same_absolute_positions(
    bank_ids, pixels_per_straw, pixel_reference
):
    """The bank origin only splits the absolute positions differently into
    translation + offsets: the pixel positions and ids are the same."""
    for bank_id in bank_ids:
        pixels = {
            origin: lbp.compute_bank_pixels(
                bank_id,
                REFERENCE_BANKS_DATA[bank_id],
                pixels_per_straw,
                pixel_reference=pixel_reference,
                bank_origin=origin,
            )
            for origin in lbp.BANK_ORIGINS
        }
        first_end, centre = pixels[lbp.FIRST_PIXEL_END], pixels[lbp.CENTRE_OF_MASS]
        assert np.allclose(first_end.translation, REFERENCE_BANKS_DATA[bank_id]["A"][0])
        assert np.linalg.norm(centre.translation - first_end.translation) > 100.0  # really moved
        assert np.allclose(centre.positions, first_end.positions, rtol=0, atol=1e-9)
        assert np.allclose(centre.pixel_centres, first_end.pixel_centres, rtol=0, atol=1e-9)
        assert np.array_equal(centre.pixel_ids, first_end.pixel_ids)
        assert first_end.bank_origin == lbp.FIRST_PIXEL_END
        assert centre.bank_origin == lbp.CENTRE_OF_MASS


@pytest.mark.parametrize("bank_id", ALL_BANKS)
def test_centre_of_mass_is_the_mean_of_the_pixel_centres(bank_id):
    """The translation is the mean of the pixel centres (the mean offset of the
    pixel centres is zero), whatever the pixel reference and the pixels per straw."""
    translations = []
    for pixels_per_straw in (1, 2, 5):
        for pixel_reference in lbp.PIXEL_REFERENCES:
            bank = lbp.compute_bank_pixels(
                bank_id,
                REFERENCE_BANKS_DATA[bank_id],
                pixels_per_straw,
                pixel_reference=pixel_reference,
                bank_origin=lbp.CENTRE_OF_MASS,
            )
            assert np.allclose(bank.translation, bank.pixel_centres.mean(axis=0), atol=1e-9)
            assert np.allclose((bank.pixel_centres - bank.translation).mean(axis=0), 0, atol=1e-9)
            translations.append(bank.translation)
    assert np.allclose(translations, translations[0], rtol=0, atol=1e-9)


def test_centre_of_mass_is_the_centre_of_the_tube_grid():
    """For a regular bank the centre of mass is the midpoint of the 8 corners."""
    for bank_id in ALL_BANKS:
        corners = REFERENCE_BANKS_DATA[bank_id]
        bank = lbp.compute_bank_pixels(bank_id, corners, 2, bank_origin=lbp.CENTRE_OF_MASS)
        corner_mean = np.mean(np.vstack([corners["A"], corners["B"]]), axis=0)
        assert np.allclose(bank.translation, corner_mean, atol=0.02)  # corner rounding


def test_unknown_bank_origin_raises():
    with pytest.raises(ValueError, match="bank_origin must be one of"):
        lbp.compute_bank_pixels(3, REFERENCE_BANKS_DATA[3], 2, bank_origin="bank_centre")


def _reference_corners(bank_id):
    return {end: np.array(REFERENCE_BANKS_DATA[bank_id][end], dtype=float) for end in "AB"}


@pytest.mark.parametrize("bank_id", [3, 6])
def test_corner_tolerance_at_its_limit(bank_id):
    """A[3] off the model by 0.2 mm is rejected (tolerance 0.1 mm), by 0.05 mm accepted."""
    for shift, accepted in ((0.05, True), (0.2, False)):
        corners = _reference_corners(bank_id)
        corners["A"][3] = corners["A"][3] + shift * np.array([0.0, 0.0, 1.0])
        if accepted:
            lbp.compute_bank_pixels(bank_id, corners, 2)
        else:
            with pytest.raises(ValueError, match="inconsistent corners"):
                lbp.compute_bank_pixels(bank_id, corners, 2)


@pytest.mark.parametrize("bank_id", [3, 6])
def test_pitch_tolerance_at_its_limit(bank_id):
    """The bottom tubes (A[2], A[3], B[2], B[3]) moved so that the tube pitch in a layer is
    0.3 mm (accepted) or 0.7 mm (rejected, tolerance 0.5 mm) larger than 28.40 mm."""
    for extra, accepted in ((0.3, True), (0.7, False)):
        corners = _reference_corners(bank_id)
        row_step = (corners["A"][2] - corners["A"][0]) / (c.TUBES_PER_LAYER[bank_id] - 1)
        shift = (c.TUBES_PER_LAYER[bank_id] - 1) * extra * row_step / np.linalg.norm(row_step)
        for end in "AB":
            corners[end][2] = corners[end][2] + shift
            corners[end][3] = corners[end][3] + shift
        if accepted:
            lbp.compute_bank_pixels(bank_id, corners, 2)
        else:
            with pytest.raises(ValueError, match="tube pitches"):
                lbp.compute_bank_pixels(bank_id, corners, 2)


@pytest.mark.parametrize("pixels_per_straw", [True, False, 0, -1, 2.0, "4"])
def test_compute_bank_pixels_rejects_bad_pixel_numbers(pixels_per_straw):
    with pytest.raises(ValueError, match="positive integer"):
        lbp.compute_bank_pixels(3, REFERENCE_BANKS_DATA[3], pixels_per_straw)
