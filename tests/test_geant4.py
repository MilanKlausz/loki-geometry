"""Tests of the Python port of the Geant4 pixel positions (geant4.py) and of the
calibration example data generated with it (geant4_example_data.py).

The reference values below are hard-coded outputs of the LOKI Geant4 geometry
package G4GeoLoki (https://github.com/ess-dg/dgcode_projects/tree/main/LOKI/G4GeoLoki):
  - sb_loki_info --nominal_source_sample_distance_meters 0 -p <pixel>  (256 px, rear 5 m)
  - sb_loki_info --nominal_source_sample_distance_meters 0 -s 2 -r 10 -p <pixel>  (2 px, 10 m)
  - G4GeoLoki.LokiAimHelper: AimHelper(rear, n).getPixelCentreCoordinates(pixel)
    (1 and 3 px, rear 5000 / 4099 mm), 6 decimals
sb_loki_info also printed the decoded bank / tube / straw / in-straw ids, which
were identical to the (bank, tube, straw, pixel) tuples used to build the ids.
(The port was also compared once with LokiAimHelper for all pixels with 1, 2,
3, 7, 256 and 512 pixels per straw: largest deviation 1e-12 mm.)
"""

import numpy as np
import pytest

from loki_geometry import constants as c
from loki_geometry import geant4 as g4
from loki_geometry.geant4_example_data import example_calibration_data


def test_port_matches_sb_loki_info_values():
    assert g4.validate() < 1e-6  # the reference values have 6 decimals


# (bank, tube, straw, pixel in straw, Geant4 pixel id, sb_loki_info centre [mm]):
# per bank: first pixel of the bank, last pixel of the bank, a back layer tube
# (straw 0), last front layer tube (straw 6, middle pixel), first tube of the
# second layer (centre straw, last pixel).
SB_LOKI_INFO_256_REAR_5M = [
    (0, 0, 0, 0, 0, (498.046875, 782.138834, 5020.365870)),
    (0, 223, 6, 255, 401407, (-498.046875, -763.298499, 5083.812565)),
    (0, 169, 0, 17, 302865, (431.640625, 772.579168, 5099.144305)),
    (0, 111, 6, 128, 200576, (-1.953125, -775.858722, 5031.293609)),
    (0, 56, 3, 255, 101375, (-498.046875, 787.280112, 5038.959478)),
    (1, 0, 0, 0, 401408, (498.046875, 709.880036, 2907.048408)),
    (1, 63, 6, 255, 516095, (-498.046875, 275.807099, 3014.136609)),
    (1, 49, 0, 17, 489233, (431.640625, 670.697533, 2990.129226)),
    (1, 31, 6, 128, 458624, (-1.953125, 283.089050, 2960.629850)),
    (1, 16, 3, 255, 431103, (-498.046875, 706.610892, 2926.060712)),
    (2, 0, 0, 0, 516096, (535.411818, -249.023438, 3336.602846)),
    (2, 47, 6, 255, 602111, (212.441204, 249.023438, 3425.664070)),
    (2, 37, 0, 17, 582417, (494.497995, -215.820312, 3418.844867)),
    (2, 23, 6, 128, 558976, (220.842118, 0.976562, 3372.321548)),
    (2, 12, 3, 255, 538623, (531.745227, 249.023438, 3355.542517)),
    (3, 0, 0, 0, 602112, (498.046875, -285.986364, 2949.358621)),
    (3, 63, 6, 255, 716799, (-498.046875, -699.700772, 2971.826396)),
    (3, 49, 0, 17, 689937, (431.640625, -303.323016, 3026.798078)),
    (3, 31, 6, 128, 659328, (-1.953125, -706.982722, 2918.319637)),
    (3, 16, 3, 255, 631807, (-498.046875, -282.717219, 2968.370925)),
    (4, 0, 0, 0, 716800, (-223.974843, -249.023438, 3361.113468)),
    (4, 47, 6, 255, 802815, (-523.878179, 249.023438, 3401.153449)),
    (4, 37, 0, 17, 783121, (-239.685924, -215.820312, 3438.899012)),
    (4, 23, 6, 128, 759680, (-532.279093, 0.976562, 3347.810927)),
    (4, 12, 3, 255, 739327, (-220.308252, 249.023438, 3380.053138)),
    (5, 0, 0, 0, 802816, (697.656250, 1097.898686, 1059.258015)),
    (5, 111, 6, 255, 1003519, (-497.656250, 369.871220, 1355.329902)),
    (5, 85, 0, 17, 955153, (617.968750, 1076.533910, 1148.595787)),
    (5, 55, 6, 128, 903040, (97.656250, 366.057363, 1301.464751)),
    (5, 28, 3, 255, 854015, (-497.656250, 1098.586568, 1078.537067)),
    (6, 0, 0, 0, 1003520, (1191.465130, -612.656250, 1517.459360)),
    (6, 127, 6, 255, 1232895, (321.479923, 582.656250, 1744.820854)),
    (6, 97, 0, 17, 1177361, (1159.522335, -532.968750, 1603.583392)),
    (6, 63, 6, 128, 1118080, (324.164878, -12.656250, 1690.887645)),
    (6, 32, 3, 255, 1061887, (1189.831906, 582.656250, 1536.681420)),
    (7, 0, 0, 0, 1232896, (497.656250, -366.468319, 1289.490670)),
    (7, 79, 6, 255, 1376255, (-697.656250, -884.376678, 1192.720500)),
    (7, 61, 0, 17, 1342225, (417.968750, -399.274666, 1361.748352)),
    (7, 39, 6, 128, 1304448, (-102.343750, -880.562821, 1138.855349)),
    (7, 20, 3, 255, 1269759, (-697.656250, -367.156201, 1308.769722)),
    (8, 0, 0, 0, 1376256, (-326.088849, -547.656250, 1679.410129)),
    (8, 127, 6, 255, 1605631, (-1186.856205, 647.656250, 1582.870085)),
    (8, 97, 0, 17, 1550097, (-349.976782, -467.968750, 1755.085724)),
    (8, 63, 6, 128, 1490816, (-1189.541159, 52.343750, 1528.936876)),
    (8, 32, 3, 255, 1434623, (-324.455625, 647.656250, 1698.632189)),
]


SB_LOKI_INFO_2_REAR_10M = [
    (0, 0, 0, 0, 0, (250.000000, 782.138834, 10020.365870)),
    (0, 223, 6, 1, 3135, (-250.000000, -763.298499, 10083.812565)),
    (2, 0, 0, 0, 4032, (535.411818, -125.000000, 3336.602846)),
    (2, 47, 6, 1, 4703, (212.441204, 125.000000, 3425.664070)),
    (5, 0, 0, 0, 6272, (400.000000, 1097.898686, 1059.258015)),
    (5, 111, 6, 1, 7839, (-200.000000, 369.871220, 1355.329902)),
    (8, 0, 0, 0, 10752, (-326.088849, -250.000000, 1679.410129)),
    (8, 127, 6, 1, 12543, (-1186.856205, 350.000000, 1582.870085)),
]


# (pixels per straw, rear distance [mm], bank, tube, straw, pixel in straw,
#  Geant4 pixel id, AimHelper centre [mm]); odd pixel counts incl. 1, and a
#  non-default rear distance: front top tube / 2nd back layer tube / last tube.
G4_AIMHELPER_1PX_AND_3PX = [
    (1, 5000.0, 0, 0, 3, 0, 3, (0.000000, 781.000000, 5012.700000)),
    (1, 5000.0, 0, 169, 5, 0, 1188, (0.000000, 764.232080, 5088.631759)),
    (1, 5000.0, 0, 223, 0, 0, 1561, (0.000000, -761.020832, 5099.144305)),
    (1, 5000.0, 1, 0, 3, 0, 1571, (0.000000, 710.251867, 2899.307333)),
    (1, 5000.0, 1, 49, 5, 0, 1916, (0.000000, 677.959246, 2978.839629)),
    (1, 5000.0, 1, 63, 0, 0, 2009, (0.000000, 275.063438, 3029.618759)),
    (1, 5000.0, 4, 0, 3, 0, 2803, (-224.508709, 0.000000, 3353.381877)),
    (1, 5000.0, 4, 37, 5, 0, 3064, (-247.182477, 0.000000, 3427.763968)),
    (1, 5000.0, 4, 47, 0, 0, 3129, (-522.810447, 0.000000, 3416.616629)),
    (1, 5000.0, 5, 0, 3, 0, 3139, (100.000000, 1096.679639, 1051.604492)),
    (1, 5000.0, 5, 85, 5, 0, 3736, (100.000000, 1081.333486, 1136.059776)),
    (1, 5000.0, 5, 111, 0, 0, 3913, (100.000000, 372.309314, 1370.636949)),
    (1, 5000.0, 7, 0, 3, 0, 4819, (-100.000000, -365.249272, 1281.837146)),
    (1, 5000.0, 7, 61, 5, 0, 5248, (-100.000000, -404.074242, 1349.212341)),
    (1, 5000.0, 7, 79, 0, 0, 5369, (-100.000000, -886.814771, 1208.027547)),
    (1, 5000.0, 8, 0, 3, 0, 5379, (-325.798102, 50.000000, 1671.665585)),
    (1, 5000.0, 8, 97, 5, 0, 6060, (-356.247633, 50.000000, 1743.217113)),
    (1, 5000.0, 8, 127, 0, 0, 6265, (-1187.437699, 50.000000, 1598.359173)),
    (3, 4099.0, 0, 0, 3, 0, 9, (333.333333, 781.000000, 4111.700000)),
    (3, 4099.0, 0, 169, 5, 2, 3566, (-333.333333, 764.232080, 4187.631759)),
    (3, 4099.0, 0, 223, 0, 1, 4684, (0.000000, -761.020832, 4198.144305)),
    (3, 4099.0, 1, 0, 3, 0, 4713, (333.333333, 710.251867, 2899.307333)),
    (3, 4099.0, 1, 49, 5, 2, 5750, (-333.333333, 677.959246, 2978.839629)),
    (3, 4099.0, 1, 63, 0, 1, 6028, (0.000000, 275.063438, 3029.618759)),
    (3, 4099.0, 4, 0, 3, 0, 8409, (-224.508709, -166.666667, 3353.381877)),
    (3, 4099.0, 4, 37, 5, 2, 9194, (-247.182477, 166.666667, 3427.763968)),
    (3, 4099.0, 4, 47, 0, 1, 9388, (-522.810447, 0.000000, 3416.616629)),
    (3, 4099.0, 5, 0, 3, 0, 9417, (500.000000, 1096.679639, 1051.604492)),
    (3, 4099.0, 5, 85, 5, 2, 11210, (-300.000000, 1081.333486, 1136.059776)),
    (3, 4099.0, 5, 111, 0, 1, 11740, (100.000000, 372.309314, 1370.636949)),
    (3, 4099.0, 7, 0, 3, 0, 14457, (300.000000, -365.249272, 1281.837146)),
    (3, 4099.0, 7, 61, 5, 2, 15746, (-500.000000, -404.074242, 1349.212341)),
    (3, 4099.0, 7, 79, 0, 1, 16108, (-100.000000, -886.814771, 1208.027547)),
    (3, 4099.0, 8, 0, 3, 0, 16137, (-325.798102, -350.000000, 1671.665585)),
    (3, 4099.0, 8, 97, 5, 2, 18182, (-356.247633, 450.000000, 1743.217113)),
    (3, 4099.0, 8, 127, 0, 1, 18796, (-1187.437699, 50.000000, 1598.359173)),
]


@pytest.mark.parametrize("bank, tube, straw, pixel, pixel_id, expected", SB_LOKI_INFO_256_REAR_5M)
def test_edge_pixels_match_sb_loki_info(bank, tube, straw, pixel, pixel_id, expected):
    assert g4.pixel_id(bank, tube, straw, pixel) == pixel_id
    assert np.allclose(g4.pixel_centre(pixel_id), expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("bank, tube, straw, pixel, pixel_id, expected", SB_LOKI_INFO_2_REAR_10M)
def test_2_pixels_per_straw_and_rear_10m_match_sb_loki_info(
    bank, tube, straw, pixel, pixel_id, expected
):
    assert g4.pixel_id(bank, tube, straw, pixel, straw_pixel_number=2) == pixel_id
    centre = g4.pixel_centre(pixel_id, straw_pixel_number=2, rear_bank_distance=10000.0)
    assert np.allclose(centre, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    "n, rear, bank, tube, straw, pix, pixel, expected", G4_AIMHELPER_1PX_AND_3PX
)
def test_odd_pixel_numbers_match_aimhelper(n, rear, bank, tube, straw, pix, pixel, expected):
    assert g4.pixel_id(bank, tube, straw, pix, straw_pixel_number=n) == pixel
    centre = g4.pixel_centre(pixel, straw_pixel_number=n, rear_bank_distance=rear)
    assert np.allclose(centre, expected, atol=1e-6, rtol=0)
    # the vectorised helper uses the same values (1 px: the np.zeros(1) branch)
    local = pixel - g4.bank_pixel_offset(bank, n)
    assert np.allclose(g4.bank_pixel_centres(bank, n, rear)[local], expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("n", [1, 2])
def test_pixel_id_is_a_bijection_onto_all_pixels(n):
    ids = [
        g4.pixel_id(b, t, s, p, n)
        for b in c.BANK_IDS
        for t in range(c.TUBES_PER_BANK[b])
        for s in range(c.STRAWS_PER_TUBE)
        for p in range(n)
    ]
    assert ids == list(range(g4.total_pixels(n)))  # in order, no gaps / duplicates


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


@pytest.fixture(scope="module")
def straws():
    """{bank: array (tube, straw, pixel, xyz)} with 2 pixels per straw [mm]."""
    return {
        b: g4.bank_pixel_centres(b, 2).reshape(c.TUBES_PER_BANK[b], 7, 2, 3) for b in c.BANK_IDS
    }


@pytest.mark.parametrize("bank", c.BANK_IDS)
def test_straws_are_along_x_or_y_with_the_documented_first_pixel_end(straws, bank):
    """First pixel at +x (horizontal banks) / -y (vertical banks), pixel pitch
    = straw length / pixels, straws parallel to the x or y axis."""
    along = 1 if bank in c.VERTICAL_BANK_IDS else 0
    step = straws[bank][:, :, 1, :] - straws[bank][:, :, 0, :]
    expected = np.zeros(3)
    expected[along] = (1 if along == 1 else -1) * c.STRAW_LENGTHS[bank] / 2
    assert np.allclose(step, expected, atol=1e-9)


@pytest.mark.parametrize("bank", c.BANK_IDS)
def test_tube_layers_and_straw_hexagon(straws, bank):
    tubes = straws[bank].mean(axis=2)  # straw midpoints, (tube, straw, xyz)
    rows = c.TUBES_PER_LAYER[bank]
    axes = tubes[:, 3, :]  # centre straw = tube axis
    # outer straws are STRAW_STRAW_DISTANCE from the tube axis
    distances = np.linalg.norm(tubes - axes[:, None, :], axis=2)
    assert np.allclose(np.delete(distances, 3, axis=1), c.STRAW_STRAW_DISTANCE)
    # tube pitch within a layer and direction of increasing tube id
    u = axes[rows - 1] - axes[0]
    u /= np.linalg.norm(u)
    assert np.allclose(np.linalg.norm(np.diff(axes[:rows], axis=0), axis=1), c.TUBE_GRID_SIDE)
    # layers: front layer = tubes 0 .. rows-1, then further from the sample
    layer_distance = [
        np.linalg.norm(axes[layer * rows : (layer + 1) * rows], axis=1).mean() for layer in range(4)
    ]
    assert np.all(np.diff(layer_distance) > 20)
    # shift front -> back layer (constants.py): along the tube ids for upside
    # down banks, against them for normally mounted ones
    shift = (axes[3 * rows :] - axes[:rows]).mean(axis=0)
    sign = 1 if bank in c.UPSIDE_DOWN_BANK_IDS else -1
    assert shift @ u == pytest.approx(sign * c.FRONT_TO_BACK_IN_PLANE, abs=1e-6)
    assert np.linalg.norm(shift - (shift @ u) * u) == pytest.approx(
        c.FRONT_TO_BACK_NORMAL, abs=1e-6
    )


def test_upside_down_banks_are_mirror_images(straws):
    """Straw pattern in the tube (component along increasing tube id) is
    mirrored for the upside down banks."""
    patterns = {}
    for bank in c.BANK_IDS:
        tubes = straws[bank].mean(axis=2)
        rows = c.TUBES_PER_LAYER[bank]
        u = tubes[rows - 1, 3] - tubes[0, 3]
        u /= np.linalg.norm(u)
        patterns[bank] = (tubes[0] - tubes[0, 3]) @ u
    for bank in c.BANK_IDS:
        mirrored = bank in c.UPSIDE_DOWN_BANK_IDS
        assert np.allclose(patterns[bank], (-1 if mirrored else 1) * patterns[0], atol=1e-9)


def test_rear_bank_moves_rigidly_with_the_rear_distance():
    for pixel in (0, 12345, g4.bank_pixel_offset(1) - 1):
        near = np.array(g4.pixel_centre(pixel, rear_bank_distance=5000.0))
        far = np.array(g4.pixel_centre(pixel, rear_bank_distance=10000.0))
        assert np.allclose(far - near, (0, 0, 5000.0))
    # the other banks do not depend on it
    pixel = g4.bank_pixel_offset(1)
    assert g4.pixel_centre(pixel, rear_bank_distance=10000.0) == g4.pixel_centre(pixel)


def test_total_pixels():
    assert g4.total_pixels() == 1_605_632
    assert g4.total_pixels(512) == 3_211_264


@pytest.mark.parametrize("bank_id", [0, 3, 6])
def test_bank_corners_do_not_depend_on_the_pixels_per_straw(bank_id):
    reference = g4.bank_corners(bank_id, 2)
    for pixels_per_straw in (1, 3, 256):
        corners = g4.bank_corners(bank_id, pixels_per_straw)
        assert np.allclose(corners["A"], reference["A"], atol=1e-9)
        assert np.allclose(corners["B"], reference["B"], atol=1e-9)
    with pytest.raises(ValueError):
        g4.bank_corners(bank_id, 0)


@pytest.mark.parametrize("pixel", [-1, g4.bank_pixel_offset(9), 10**9])
def test_pixel_id_out_of_range_raises_a_clear_error(pixel):
    with pytest.raises(ValueError, match="out of range"):
        g4.pixel_centre(pixel)


@pytest.mark.parametrize(
    "args", [(9, 0, 0, 0), (1, 64, 0, 0), (1, 0, 7, 0), (1, 0, 0, 256), (1, -1, 0, 0)]
)
def test_pixel_id_arguments_out_of_range_raise(args):
    with pytest.raises(ValueError, match="out of range"):
        g4.pixel_id(*args)


@pytest.mark.parametrize("bank_id", [-1, 9])
def test_bank_pixel_centres_rejects_bad_bank_ids(bank_id):
    """A negative bank id must not index the per-bank tables from the end."""
    with pytest.raises(ValueError, match="out of range"):
        g4.bank_pixel_centres(bank_id, 2)


def test_non_integer_pixel_ids_are_rejected():
    """Ids must be integers (10.5 would be a position between two pixels)."""
    with pytest.raises(ValueError, match="must be an integer"):
        g4.pixel_centre(10.5)
    with pytest.raises(ValueError, match="must be an integer"):
        g4.pixel_id(0, 0.5, 0, 0)
    assert g4.pixel_centre(np.int64(10)) == g4.pixel_centre(10)  # numpy integers are fine


@pytest.mark.parametrize("pixels_per_straw", [2.0, True, 0])
def test_pixels_per_straw_must_be_a_positive_integer(pixels_per_straw):
    with pytest.raises(ValueError, match="positive integer"):
        g4.pixel_centre(0, pixels_per_straw)
    with pytest.raises(ValueError, match="positive integer"):
        g4.bank_corners(1, pixels_per_straw)


def test_validate_does_not_pass_nan(monkeypatch):
    """A NaN position must fail the validation (max(0, nan) is 0 and nan > tolerance
    is False, so it would otherwise pass)."""
    monkeypatch.setattr(g4, "pixel_centre", lambda pixel: (np.nan, 0.0, 0.0))
    assert g4.main([]) == 1


def test_validate_cli_exit_code(capsys):
    assert g4.main([]) == 0
    assert "Largest deviation" in capsys.readouterr().out
    assert g4.main(["--tolerance", "1e-9"]) == 1  # the reference values have 6 decimals


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
