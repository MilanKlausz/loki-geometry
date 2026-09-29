"""Detailed tests of the pixel geometry (bank_pixels.py), the NeXus-JSON export
(nexus_json.py) and the reference corner data (reference_data.py).

Several expected values are copied from the current LOKI instrument file
loki.json (512 pixels per straw); they are hard-coded below.
"""

import json

import numpy as np
import pytest

from loki_geometry import bank_pixels as lbp
from loki_geometry import constants as c
from loki_geometry import nexus_json as lnj
from loki_geometry import verify_json as vnj
from loki_geometry.reference_data import (
    REFERENCE_BANKS_DATA,
    convert_reference_data_to_calibration_data,
)

ALL_BANKS = range(c.NUMBER_OF_BANKS)
# --- values taken from loki.json (512 pixels per straw) ---------------------------

# detector_number of the first pixel of each bank
LOKI_JSON_FIRST_IDS_512 = (1, 802817, 1032193, 1204225, 1433601, 1605633, 2007041, 2465793, 2752513)
LOKI_JSON_PIXELS_PER_BANK_512 = (
    802816,
    229376,
    172032,
    229376,
    172032,
    401408,
    458752,
    286720,
    458752,
)
# bank translations (trans_<id+1>, detector_carriage_zero for bank 0): magnitude [m], unit vector
LOKI_JSON_TRANSLATIONS = {
    0: (5.0975599309865895, (0.09808614450232256, 0.15321055771262784, 0.9833135986357836)),
    1: (3.0264972377320944, (0.16520748598953786, 0.23467392969841872, 0.9579324784622967)),
    2: (3.3808738879319353, (0.1585211450545504, -0.07394537870589542, 0.9845827174689975)),
    3: (2.997243043948889, (0.1668199717768788, -0.09553112503774741, 0.9813485115724095)),
    4: (3.3699024365996118, (-0.06661617189918438, -0.0741861239912517, 0.9950169368652239)),
    5: (1.6727540228616997, (0.418471568702289, 0.6556074503553417, 0.6285383180255708)),
    6: (2.0188920948381566, (0.5900018148793079, -0.3046225212196401, 0.7477318891186284)),
    7: (1.4237046781197287, (0.351196429768247, -0.25662625516025345, 0.9004465741399993)),
    8: (1.7887985600676228, (-0.18203279411611917, -0.30746894160022586, 0.9339844280380243)),
}
# offsets [mm] of the first pixel of the 7 straws of tube 0 (pixel ids first + s * 512)
LOKI_JSON_TUBE0_STRAW_OFFSETS_MM = {
    0: [[-0.977, -1.14, -7.67], [-0.977, -7.21, -2.85], [-0.977, -6.07, 4.82], [-0.977, 0.0, 0.0],
        [-0.977, 6.07, -4.82], [-0.977, 7.21, 2.85], [-0.977, 1.14, 7.67]],
    1: [[-0.977, 0.38, -7.74], [-0.977, 6.9, -3.54], [-0.977, 6.52, 4.2], [-0.977, 0.0, 0.0],
        [-0.977, -6.52, -4.2], [-0.977, -6.9, 3.54], [-0.977, -0.38, 7.74]],
    2: [[0.54, 0.488, -7.74], [6.97, 0.488, -3.4], [6.43, 0.488, 4.34], [0.0, 0.488, 0.0],
        [-6.43, 0.488, -4.34], [-6.97, 0.488, 3.4], [-0.54, 0.488, 7.74]],
    3: [[-0.977, -0.38, -7.75], [-0.977, -6.9, -3.54], [-0.977, -6.52, 4.2], [-0.977, 0.0, 0.0],
        [-0.977, 6.52, -4.2], [-0.977, 6.9, 3.54], [-0.977, 0.38, 7.75]],
    4: [[-0.54, 0.488, -7.74], [-6.97, 0.488, -3.4], [-6.43, 0.488, 4.34], [0.0, 0.488, 0.0],
        [6.43, 0.488, -4.34], [6.97, 0.488, 3.4], [0.54, 0.488, 7.74]],
    5: [[-1.172, -1.21, -7.66], [-1.172, 6.03, -4.88], [-1.172, 7.24, 2.78], [-1.172, 0.0, 0.0],
        [-1.172, -7.24, -2.78], [-1.172, -6.03, 4.88], [-1.172, 1.21, 7.66]],
    6: [[-0.28, 1.172, -7.75], [6.57, 1.172, -4.12], [6.85, 1.172, 3.63], [0.0, 1.172, 0.0],
        [-6.85, 1.172, -3.63], [-6.57, 1.172, 4.12], [0.28, 1.172, 7.75]],
    7: [[-1.172, 1.21, -7.66], [-1.172, -6.03, -4.88], [-1.172, -7.24, 2.78], [-1.172, 0.0, 0.0],
        [-1.172, 7.24, -2.78], [-1.172, 6.03, 4.88], [-1.172, -1.21, 7.66]],
    8: [[0.28, 1.172, -7.75], [-6.57, 1.172, -4.12], [-6.85, 1.172, 3.63], [0.0, 1.172, 0.0],
        [6.85, 1.172, -3.63], [6.57, 1.172, 4.12], [-0.28, 1.172, 7.75]],
}  # fmt: skip
# pixel offsets [m] of a few pixels per bank (local index in the bank: offset), 5 decimals
# across the straws, and the pixel_shape vertices [m] (a pixel_base shape, see below)
LOKI_JSON_SAMPLE_OFFSETS_M = {
    0: {0: (-0.0009765625, -0.00114, -0.00767), 511: (-0.9990234375, -0.00114, -0.00767),
        1536: (-0.0009765625, 0.0, 0.0), 401664: (-0.5009765625, 0.01142, 0.04485),
        802441: (-0.2685546875, -1.54202, 0.08645), 802815: (-0.9990234375, -1.54202, 0.08645)},
    1: {0: (-0.0009765625, 0.00038, -0.00774), 511: (-0.9990234375, 0.00038, -0.00774),
        1536: (-0.0009765625, 0.0, 0.0), 114944: (-0.5009765625, -0.00695333333333334, 0.04576),
        229001: (-0.2685546875, -0.43528, 0.13023), 229375: (-0.9990234375, -0.43528, 0.13023)},
    2: {0: (0.00054, 0.00048828125, -0.00774), 511: (0.00054, 0.49951171875, -0.00774),
        1536: (0.0, 0.00048828125, 0.0), 86272: (-0.00791333333333342, 0.25048828125, 0.04559333333333338),
        171657: (-0.32467, 0.13427734375, 0.11217), 172031: (-0.32467, 0.49951171875, 0.11217)},
    3: {0: (-0.0009765625, -0.00038, -0.00775), 511: (-0.9990234375, -0.00038, -0.00775),
        1536: (-0.0009765625, 0.0, 0.0), 114944: (-0.5009765625, 0.00694666666666667, 0.04575),
        229001: (-0.2685546875, -0.41253, 0.04577), 229375: (-0.9990234375, -0.41253, 0.04577)},
    4: {0: (-0.00054, 0.00048828125, -0.00774), 511: (-0.00054, 0.49951171875, -0.00774),
        1536: (0.0, 0.00048828125, 0.0), 86272: (0.00790666666666668, 0.25048828125, 0.04559333333333338),
        171657: (-0.29823, 0.13427734375, 0.0633), 172031: (-0.29823, 0.49951171875, 0.0633)},
    5: {0: (-0.001171875, -0.00121, -0.00766), 511: (-1.198828125, -0.00121, -0.00766),
        1536: (-0.001171875, 0.0, 0.0), 200960: (-0.601171875, 0.00255666666666662, 0.04620666666666665),
        401033: (-0.322265625, -0.72447, 0.31897), 401407: (-1.198828125, -0.72447, 0.31897)},
    6: {0: (-0.00028, 0.001171875, -0.00775), 511: (-0.00028, 1.198828125, -0.00775),
        1536: (0.0, 0.001171875, 0.0), 229632: (-0.00301333333333348, 0.601171875, 0.04618333333333331),
        458377: (-0.86921, 0.322265625, 0.25053), 458751: (-0.86921, 1.198828125, 0.25053)},
    7: {0: (-0.001171875, 0.00121, -0.00766), 511: (-1.198828125, 0.00121, -0.00766),
        1536: (-0.001171875, 0.0, 0.0), 143616: (-0.601171875, -0.00255666666666666, 0.04620666666666665),
        286345: (-0.322265625, -0.5215, -0.07373), 286719: (-1.198828125, -0.5215, -0.07373)},
    8: {0: (0.00028, 0.001171875, -0.00775), 511: (0.00028, 1.198828125, -0.00775),
        1536: (0.0, 0.001171875, 0.0), 229632: (0.00301333333333337, 0.601171875, 0.04618333333333317),
        458377: (-0.8617, 0.322265625, -0.07254), 458751: (-0.8617, 1.198828125, -0.07254)},
}  # fmt: skip
LOKI_JSON_PIXEL_SHAPE_VERTICES_M = {
    0: ((0.0, 0.0, 0.0), (0.0, -0.002415912617779831, 0.002868077095067739), (-0.001953125, 0.0, 0.0)),
    1: ((0.0, 0.0, 0.0), (0.0, -0.0027514190021446145, 0.0025479783112572862), (-0.001953125, 0.0, 0.0)),
    2: ((0.0, 0.0, 0.0), (-0.0028043736453638736, 0.0, 0.002489575959312858), (0.0, 0.0009765625, 0.0)),
    3: ((0.0, 0.0, 0.0), (0.0, -0.0026875706994991854, 0.002615236841128057), (-0.001953125, 0.0, 0.0)),
    4: ((0.0, 0.0, 0.0), (-0.0026321597107023453, 0.0, 0.002670998925000784), (0.0, 0.0009765625, 0.0)),
    5: ((0.0, 0.0, 0.0), (0.0, -0.0021720306529693584, 0.0030569237547838023), (-0.00234375, 0.0, 0.0)),
    6: ((0.0, 0.0, 0.0), (-0.002523471559563634, 0.0, 0.0027739126316583007), (0.0, 0.00234375, 0.0)),
    7: ((0.0, 0.0, 0.0), (0.0, -0.0031656205537874595, 0.0020103100530610638), (-0.00234375, 0.0, 0.0)),
    8: ((0.0, 0.0, 0.0), (-0.0029001153354025234, 0.0, 0.002377357996045844), (0.0, 0.00234375, 0.0)),
}  # fmt: skip


def loki_json_structure(bank_id):
    """Structure of a loki_detector_<id> group in loki.json, in the form of _structure()
    (without detector_events)."""
    units = (("units", "string"),)
    datasets = [
        ("detector_number", "dataset", "int32", ()),
        (
            "pixel_shape",
            "group",
            (("NX_class", "NXcylindrical_geometry"),),
            (("cylinders", "dataset", "int32", ()), ("vertices", "dataset", "float", units)),
        ),
        ("x_pixel_offset", "dataset", "float", units),
        ("y_pixel_offset", "dataset", "float", units),
        ("z_pixel_offset", "dataset", "float", units),
    ]
    if bank_id != lnj.CARRIAGE_BANK_ID:
        datasets.append(
            (
                "transformations",
                "group",
                (("NX_class", "NXtransformations"),),
                ((f"trans_{bank_id + 1}", *LOKI_JSON_TRANSLATION_STRUCTURE[1:]),),
            )
        )
    datasets.append(("depends_on", "dataset", "string", ()))
    return (lnj.bank_name(bank_id), "group", (("NX_class", "NXdetector"),), tuple(datasets))


LOKI_JSON_TRANSLATION_STRUCTURE = (
    "detector_carriage_zero",
    "dataset",
    "float",
    (
        ("transformation_type", None),
        ("depends_on", "string"),
        ("vector", "float"),
        ("units", "string"),
    ),
)
LOKI_JSON_DEPENDS_ON = ["/entry/instrument/detector_carriage/value"] + [
    f"/entry/instrument/loki_detector_{b}/transformations/trans_{b + 1}" for b in range(1, 9)
]


def child(node, name):
    return vnj._child(node, name)


def values(node):
    return node["config"]["values"]


def corner_tube_ids(bank_id):
    """Tube ids of the corners A[0..3] / B[0..3]: front top, back top, front bottom, back bottom."""
    rows = lbp.tubes_per_layer(bank_id)
    return (0, 3 * rows, rows - 1, 4 * rows - 1)


# ---------------------------------------------------------------------------
# reference_data.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bank_id", ALL_BANKS)
def test_reference_corners_form_a_regular_bank(bank_id):
    """The model uses only A[0], A[1], A[2], B[0]; the other corners must agree with it."""
    a = np.array(REFERENCE_BANKS_DATA[bank_id]["A"], dtype=float)
    b = np.array(REFERENCE_BANKS_DATA[bank_id]["B"], dtype=float)
    assert np.allclose(a[3], a[1] + a[2] - a[0], atol=0.02)  # parallelogram
    assert np.allclose(b - a, b[0] - a[0], atol=1e-9)  # parallel tubes of equal length
    assert np.linalg.norm(b[0] - a[0]) == pytest.approx(c.STRAW_LENGTHS[bank_id])
    assert REFERENCE_BANKS_DATA[bank_id]["tube_length"] * 1000 == c.STRAW_LENGTHS[bank_id]
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


# ---------------------------------------------------------------------------
# bank_pixels.py: tube / straw / pixel layout and numbering
# ---------------------------------------------------------------------------


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


def test_bank_pixel_id_ranges_match_loki_json():
    for bank_id in ALL_BANKS:
        ids = lbp.bank_pixel_ids(bank_id, 512)
        assert ids[0] == LOKI_JSON_FIRST_IDS_512[bank_id]
        assert len(ids) == LOKI_JSON_PIXELS_PER_BANK_512[bank_id]
    assert lbp.bank_pixel_ids(8, 512)[-1] == 3211264


def test_pixel_ids_fit_the_int32_dtype_of_detector_number():
    assert lbp.bank_pixel_ids(8, 1024)[-1] == 896 * 7 * 1024 < 2**31 - 1


@pytest.mark.parametrize("bank_id", ALL_BANKS)
def test_tube0_straw_positions_match_loki_json(bank_id):
    """Straw order and mirroring in the upside-down banks and the pixel centre
    offset along the straw, for all 7 straws of tube 0."""
    bank = lbp.compute_bank_pixels(bank_id, REFERENCE_BANKS_DATA[bank_id], 512)
    first_pixels = bank.offsets[[straw * 512 for straw in range(7)]]
    # loki.json: 5 decimals in m, straw distance sqrt(1.14^2 + 7.67^2) = 7.754 mm
    assert np.allclose(first_pixels, LOKI_JSON_TUBE0_STRAW_OFFSETS_MM[bank_id], atol=0.015)


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


# ---------------------------------------------------------------------------
# nexus_json.py
# ---------------------------------------------------------------------------


def json_faces_mm(output, bank_id):
    """(first face centres, second face centres) of all pixel cylinders of a bank
    as described by the JSON output [mm]."""
    group = output[lnj.bank_name(bank_id)]
    ids, positions = vnj.bank_pixels_from_group(
        group, output.get(lnj.CARRIAGE_ZERO_NAME), use_pixel_shape=False
    )
    vertices = np.array(values(child(child(group, "pixel_shape"), "vertices"))) * 1000
    return ids, positions + vertices[0], positions + vertices[2]


@pytest.mark.parametrize("pixel_reference", lbp.PIXEL_REFERENCES)
@pytest.mark.parametrize("bank_id", [0, 1, 6, 7])
def test_json_pixel_cylinders_tile_the_straws(pixel_reference, bank_id):
    """The pixel cylinders of the JSON output fill every straw from its first to
    its last pixel end without gaps or overlaps (whatever the pixel reference):
    guards against the half-pixel pixel_shape shift of the current loki.json."""
    n = 3
    pixels = lbp.compute_instrument_pixels(
        {bank_id: REFERENCE_BANKS_DATA[bank_id]}, n, pixel_reference=pixel_reference
    )
    output = json.loads(json.dumps(lnj.build_output(pixels, "test")))
    _, first, second = json_faces_mm(output, bank_id)
    first = first.reshape(-1, 7, n, 3)
    second = second.reshape(-1, 7, n, 3)
    frame = pixels[bank_id].frame
    straw_starts = (
        frame.origin + lbp.tube_offsets(frame)[:, None, :] + lbp.straw_offsets(frame)[None, :, :]
    )
    atol = 1e-4  # 1e-8 m rounding
    assert np.allclose(first[:, :, 0], straw_starts, atol=atol)
    assert np.allclose(second[:, :, :-1], first[:, :, 1:], atol=atol)
    assert np.allclose(second[:, :, -1], straw_starts + frame.along * frame.straw_length, atol=atol)


def test_json_is_in_metres():
    bank_id = 5
    pixels = lbp.compute_instrument_pixels({bank_id: REFERENCE_BANKS_DATA[bank_id]}, 2)
    group = lnj.build_output(pixels, "test")[lnj.bank_name(bank_id)]
    offsets = np.column_stack([values(child(group, f"{a}_pixel_offset")) for a in "xyz"])
    assert np.allclose(offsets, pixels[bank_id].offsets / 1000, atol=1e-8)
    translation = child(child(group, "transformations"), "trans_6")
    vector = np.array(vnj._attribute(translation, "vector"))
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert np.allclose(
        values(translation) * vector, np.array(REFERENCE_BANKS_DATA[5]["A"][0]) / 1000
    )
    vertices = np.array(values(child(child(group, "pixel_shape"), "vertices")))
    assert np.linalg.norm(vertices[1] - vertices[0]) == pytest.approx(c.STRAW_RADIUS / 1000)

    def all_units(node):
        for attr in node.get("attributes", []):
            if attr["name"] == "units":
                yield attr["values"]
        for sub in node.get("children", []):
            yield from all_units(sub)

    assert set(all_units(group)) == {"m"}


def test_translations_match_loki_json():
    pixels = lbp.compute_instrument_pixels(REFERENCE_BANKS_DATA, 1)
    output = lnj.build_output(pixels, "test")
    for bank_id, (magnitude, vector) in LOKI_JSON_TRANSLATIONS.items():
        if bank_id == lnj.CARRIAGE_BANK_ID:
            node = output[lnj.CARRIAGE_ZERO_NAME]
        else:
            group = output[lnj.bank_name(bank_id)]
            node = child(child(group, "transformations"), f"trans_{bank_id + 1}")
        assert values(node) == pytest.approx(magnitude, abs=1e-8)
        assert np.allclose(vnj._attribute(node, "vector"), vector, atol=1e-12)


def test_pixel_shape_is_the_same_cylinder_for_both_references():
    """Same cylinder, only placed differently relative to the offset point."""
    shapes = {
        ref: lnj.pixel_shape_vertices(
            lbp.compute_bank_pixels(8, REFERENCE_BANKS_DATA[8], 4, pixel_reference=ref)
        )
        for ref in lbp.PIXEL_REFERENCES
    }
    centre, base = shapes[lbp.PIXEL_CENTRE], shapes[lbp.PIXEL_BASE]
    assert np.allclose(centre - centre[0], base - base[0])
    assert np.allclose(base[0], 0.0)
    assert np.allclose((centre[0] + centre[2]) / 2, 0.0)


def _structure(node):
    """Names, types, dtypes and attributes (without values) of a NeXus-JSON node tree."""
    if "module" in node:
        config = node.get("config", {})
        return (
            config.get("name"),
            node["module"],
            config.get("dtype"),
            tuple((a["name"], a.get("dtype")) for a in node.get("attributes", [])),
        )
    return (
        node.get("name"),
        node.get("type"),
        tuple((a["name"], a.get("values")) for a in node.get("attributes", [])),
        tuple(
            _structure(sub)
            for sub in node.get("children", [])
            if sub.get("name") != "detector_events"
        ),
    )


def test_json_structure_matches_loki_json():
    """Same groups, datasets, dtypes, attributes and depends_on paths as the instrument file."""
    pixels = lbp.compute_instrument_pixels(REFERENCE_BANKS_DATA, 1)
    output = lnj.build_output(pixels, "test")
    for bank_id in ALL_BANKS:
        ours = output[lnj.bank_name(bank_id)]
        assert _structure(ours) == loki_json_structure(bank_id)
        assert values(child(ours, "depends_on")) == LOKI_JSON_DEPENDS_ON[bank_id]
    assert _structure(output[lnj.CARRIAGE_ZERO_NAME]) == LOKI_JSON_TRANSLATION_STRUCTURE


@pytest.fixture(scope="module")
def pixels_512():
    return lbp.compute_instrument_pixels(REFERENCE_BANKS_DATA, 512)


@pytest.mark.parametrize("bank_id", ALL_BANKS)
def test_pixel_offsets_agree_with_loki_json(pixels_512, bank_id):
    """loki.json was made from the same corners; its straw positions are rounded
    to 5 decimals in m, so the offsets agree within 0.011 mm."""
    offsets = pixels_512[bank_id].offsets
    for index, expected in LOKI_JSON_SAMPLE_OFFSETS_M[bank_id].items():
        assert np.abs(offsets[index] - np.array(expected) * 1000).max() < 0.011


@pytest.mark.parametrize("bank_id", ALL_BANKS)
def test_loki_json_pixel_shape_is_shifted_by_half_a_pixel(pixels_512, bank_id):
    """Documents the known issue of loki.json: its offsets point to the pixel
    centres, but its pixel shape starts at the offset point (a pixel_base shape),
    so its pixel volumes are shifted by half a pixel along the straws. Our
    pixel_centre output centres the shape on the offset point."""
    bank = pixels_512[bank_id]
    first_offset = np.array(LOKI_JSON_SAMPLE_OFFSETS_M[bank_id][0]) * 1000
    along = np.dot(first_offset, bank.frame.along)
    assert along == pytest.approx(bank.pixel_length / 2, abs=1e-6)  # pixel centre offsets
    vertices = np.array(LOKI_JSON_PIXEL_SHAPE_VERTICES_M[bank_id]) * 1000
    shape_centre = (vertices[0] + vertices[2]) / 2
    assert np.linalg.norm(shape_centre) == pytest.approx(bank.pixel_length / 2, abs=1e-6)
    ours = lnj.pixel_shape_vertices(bank) * 1000
    assert np.allclose(ours[0] + ours[2], 0.0, atol=1e-5)


# --- command line ------------------------------------------------------------------


def run_main(tmp_path, *args):
    filename = tmp_path / "out.json"
    lnj.main(["-o", str(filename), *args])
    return json.loads(filename.read_text())


def test_main_bank_subset_keeps_the_global_pixel_ids(tmp_path):
    data = run_main(tmp_path, "--pixels-per-straw", "2", "--banks", "5")
    ids = values(child(data["loki_detector_5"], "detector_number"))
    assert ids[0] == lbp.bank_first_pixel_id(5, 2) == 1 + 2 * 7 * sum(c.TUBES_PER_BANK[:5])
    assert len(ids) == 112 * 7 * 2
    assert lnj.CARRIAGE_ZERO_NAME not in data


def test_main_writes_the_carriage_zero_with_bank_0(tmp_path):
    data = run_main(tmp_path, "--pixels-per-straw", "1", "--banks", "0")
    assert np.allclose(
        vnj.translation_mm(data[lnj.CARRIAGE_ZERO_NAME]), REFERENCE_BANKS_DATA[0]["A"][0]
    )
    assert child(data["loki_detector_0"], "transformations") is None


def test_main_default_is_pixel_centre_and_256_pixels(tmp_path):
    data = run_main(tmp_path, "--banks", "2")
    assert data["info"]["pixels_per_straw"] == 256
    assert data["info"]["pixel_reference"] == lbp.PIXEL_CENTRE
    vertices = np.array(values(child(child(data["loki_detector_2"], "pixel_shape"), "vertices")))
    assert np.allclose(vertices[0] + vertices[2], 0.0)
    assert np.linalg.norm(vertices[2] - vertices[0]) == pytest.approx(0.5 / 256, abs=1e-8)


@pytest.mark.parametrize("pixels_per_straw", ["0", "-2"])
def test_main_rejects_invalid_pixels_per_straw(tmp_path, pixels_per_straw):
    with pytest.raises(SystemExit, match="2"):
        lnj.main(
            [
                "-o",
                str(tmp_path / "out.json"),
                "--pixels-per-straw",
                pixels_per_straw,
                "--banks",
                "2",
            ]
        )


@pytest.mark.parametrize("banks", [["9"], ["-1"], []])
def test_main_rejects_unknown_or_missing_bank_ids(tmp_path, banks):
    with pytest.raises(SystemExit, match="2"):
        lnj.main(["-o", str(tmp_path / "out.json"), "--banks", *banks])
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("pixels_per_straw", [0, -2, 2.5])
def test_compute_bank_pixels_rejects_invalid_pixels_per_straw(pixels_per_straw):
    with pytest.raises(ValueError, match="positive integer"):
        lbp.compute_bank_pixels(3, REFERENCE_BANKS_DATA[3], pixels_per_straw)


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
