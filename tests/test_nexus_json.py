"""Tests of the NeXus-JSON export (nexus_json.py), including the comparison with
the current LOKI instrument file loki.json (512 pixels per straw), whose
values are hard-coded in loki_json_reference.py."""

import contextlib
import io
import json

import numpy as np
import pytest
from geant4_reference import GEANT4_SAM606_PIXELS
from loki_json_reference import (  # the values of loki.json
    LOKI_JSON_FIRST_IDS_512,
    LOKI_JSON_PIXEL_SHAPE_VERTICES_M,
    LOKI_JSON_PIXELS_PER_BANK_512,
    LOKI_JSON_SAMPLE_OFFSETS_M,
    LOKI_JSON_TRANSLATIONS,
    LOKI_JSON_TUBE0_STRAW_OFFSETS_MM,
)

from loki_geometry import bank_pixels as lbp
from loki_geometry import calibration, survey
from loki_geometry import constants as c
from loki_geometry import nexus_json as lnj
from loki_geometry import verify_json as vnj
from loki_geometry.reference_data import REFERENCE_BANKS_DATA
from loki_geometry.survey_files import SAM606

ALL_BANKS = range(c.NUMBER_OF_BANKS)


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


@pytest.fixture(scope="module")
def reference_pixels():
    return lbp.compute_instrument_pixels(REFERENCE_BANKS_DATA, pixels_per_straw=2)


def test_translation_dataset_points_to_the_position():
    node = lnj.translation_dataset("trans_2", (500.0, 710.24, 2899.18))
    assert np.allclose(vnj.translation_mm(node), (500.0, 710.24, 2899.18))
    assert vnj._attribute(node, "units") == "m"
    assert vnj._attribute(node, "transformation_type") == "translation"


def test_bank_group_structure(reference_pixels):
    node = lnj.bank_group(reference_pixels[3])
    names = [c.get("name") or c["config"]["name"] for c in node["children"]]
    assert names == [
        "detector_number",
        "pixel_shape",
        "x_pixel_offset",
        "y_pixel_offset",
        "z_pixel_offset",
        "transformations",
        "depends_on",
    ]
    n_pixels = 64 * 7 * 2
    for name in ("detector_number", "x_pixel_offset", "y_pixel_offset", "z_pixel_offset"):
        assert len(child(node, name)["config"]["values"]) == n_pixels
    assert (
        child(node, "depends_on")["config"]["values"]
        == "/entry/instrument/loki_detector_3/transformations/trans_4"
    )
    assert vnj._attribute(node, "NX_class") == "NXdetector"


def test_carriage_bank_depends_on_the_carriage(reference_pixels):
    output = lnj.build_output(reference_pixels, "test")
    bank0 = output["loki_detector_0"]
    assert child(bank0, "transformations") is None
    assert (
        child(bank0, "depends_on")["config"]["values"]
        == "/entry/instrument/detector_carriage/value"
    )
    assert np.allclose(
        vnj.translation_mm(output["detector_carriage_zero"]), REFERENCE_BANKS_DATA[0]["A"][0]
    )


def shape_vertices_mm(bank_pixels):
    return (
        np.array(child(lnj.pixel_shape_group(bank_pixels), "vertices")["config"]["values"]) * 1000
    )


@pytest.mark.parametrize("pixel_reference", lbp.PIXEL_REFERENCES)
def test_pixel_shape_is_a_cylinder_of_one_pixel(pixel_reference):
    bank = lbp.compute_bank_pixels(5, REFERENCE_BANKS_DATA[5], 2, pixel_reference=pixel_reference)
    vertices = shape_vertices_mm(bank)
    axis = vertices[2] - vertices[0]
    assert np.linalg.norm(axis) == pytest.approx(1200.0 / 2, abs=1e-4)
    assert np.allclose(
        axis / np.linalg.norm(axis), bank.frame.along, atol=1e-6
    )  # from the first to the last pixel end
    radius = vertices[1] - vertices[0]
    assert np.linalg.norm(radius) == pytest.approx(lbp.STRAW_RADIUS, abs=1e-4)
    assert np.dot(radius, axis) == pytest.approx(0.0, abs=1e-6)


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


def json_faces_mm(output, bank_id):
    """(first face centres, second face centres) of all pixel cylinders of a bank
    as described by the JSON output [mm]."""
    group = output[lnj.bank_name(bank_id)]
    ids, positions = vnj.pixel_positions_from_tree(lnj.instrument_tree(output), False)[bank_id]
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


def test_bank_pixel_id_ranges_match_loki_json():
    for bank_id in ALL_BANKS:
        ids = lbp.bank_pixel_ids(bank_id, 512)
        assert ids[0] == LOKI_JSON_FIRST_IDS_512[bank_id]
        assert len(ids) == LOKI_JSON_PIXELS_PER_BANK_512[bank_id]
    assert lbp.bank_pixel_ids(8, 512)[-1] == 3211264


@pytest.mark.parametrize("bank_id", ALL_BANKS)
def test_tube0_straw_positions_match_loki_json(bank_id):
    """Straw order and mirroring in the upside-down banks and the pixel centre
    offset along the straw, for all 7 straws of tube 0."""
    bank = lbp.compute_bank_pixels(bank_id, REFERENCE_BANKS_DATA[bank_id], 512)
    first_pixels = bank.offsets[[straw * 512 for straw in range(7)]]
    # loki.json: 5 decimals in m, straw distance sqrt(1.14^2 + 7.67^2) = 7.754 mm
    assert np.allclose(first_pixels, LOKI_JSON_TUBE0_STRAW_OFFSETS_MM[bank_id], atol=0.015)


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


@pytest.fixture(scope="module")
def measured_corners():
    """Corners calibrated from the package survey (all 9 banks)."""
    with contextlib.redirect_stdout(io.StringIO()):
        data = survey.get_measured_calibration_data(SAM606.path, verbose=False)
        detectors = calibration.Detectors(data, True, run_checks=False)
    return detectors.formatted_data()


def test_measured_corners_json_without_bank_0(tmp_path, measured_corners):
    """Measured data without bank 0 (e.g. a survey of banks 1-8): no detector_carriage_zero,
    global ids kept, and the JSON reproduces the model positions."""
    corners = {bank_id: v for bank_id, v in measured_corners.items() if bank_id != 0}
    pixels = lbp.compute_instrument_pixels(corners, 2)
    output = lnj.build_output(pixels, "measured")
    assert lnj.CARRIAGE_ZERO_NAME not in output
    assert lnj.bank_name(0) not in output
    filename = tmp_path / "measured.json"
    lnj.write_json(output, filename)
    banks = vnj.read_pixel_positions(filename)
    assert sorted(banks) == list(range(1, 9))
    for bank_id, (ids, positions) in banks.items():
        assert np.array_equal(ids, pixels[bank_id].pixel_ids)
        assert np.abs(positions - pixels[bank_id].pixel_centres).max() < 1e-4


def run_main(tmp_path, *args):
    filename = tmp_path / "out.json"
    lnj.main(["-o", str(filename), *args])
    return lnj.tree_nodes(json.loads(filename.read_text()))


def test_main_writes_selected_banks(tmp_path):
    filename = tmp_path / "out.json"
    lnj.main(
        [
            "-o",
            str(filename),
            "--pixels-per-straw",
            "2",
            "--banks",
            "1",
            "2",
            "--pixel-reference",
            "pixel_base",
        ]
    )
    data = lnj.tree_nodes(json.loads(filename.read_text()))
    assert {key for key in data if key.startswith("loki_detector")} == {
        "loki_detector_1",
        "loki_detector_2",
    }
    assert data["info"]["pixels_per_straw"] == 2
    assert data["info"]["pixel_reference"] == "pixel_base"
    assert vnj.main([str(filename), "--tolerance", "1.0"]) == 0


def test_main_bank_subset_keeps_the_global_pixel_ids(tmp_path):
    data = run_main(tmp_path, "--pixels-per-straw", "2", "--banks", "5")
    ids = values(child(data["loki_detector_5"], "detector_number"))
    assert ids[0] == lbp.bank_first_pixel_id(5, 2) == 1 + 2 * 7 * sum(c.TUBES_PER_BANK[:5])
    assert len(ids) == 112 * 7 * 2
    assert lnj.CARRIAGE_ZERO_NAME not in data


def test_main_writes_the_carriage_zero_with_bank_0(tmp_path, measured_corners):
    """detector_carriage_zero is the bank 0 position (its corner A[0]) of the calibration: the
    CAD geometry, or the survey (bank 0 at the carriage zero position, see carriage_position)."""
    data = run_main(
        tmp_path, "--pixels-per-straw", "1", "--banks", "0", "-c", "nominal-cad-geometry"
    )
    assert np.allclose(
        vnj.translation_mm(data[lnj.CARRIAGE_ZERO_NAME]), REFERENCE_BANKS_DATA[0]["A"][0]
    )
    assert child(data["loki_detector_0"], "transformations") is None
    data = run_main(tmp_path, "--pixels-per-straw", "1", "--banks", "0")
    assert np.allclose(
        vnj.translation_mm(data[lnj.CARRIAGE_ZERO_NAME]), measured_corners[0]["A"][0], atol=0.01
    )


def test_main_default_is_pixel_centre_and_256_pixels(tmp_path):
    data = run_main(tmp_path, "--banks", "2")
    assert data["info"]["pixels_per_straw"] == 256
    assert data["info"]["pixel_reference"] == lbp.PIXEL_CENTRE
    vertices = np.array(values(child(child(data["loki_detector_2"], "pixel_shape"), "vertices")))
    assert np.allclose(vertices[0] + vertices[2], 0.0)
    assert np.linalg.norm(vertices[2] - vertices[0]) == pytest.approx(0.5 / 256, abs=1e-8)


def test_main_decimals_option_rounds_the_metre_values(tmp_path):
    """--decimals is otherwise untested: 4 decimals in m = 0.1 mm grid."""
    filename = tmp_path / "out.json"
    lnj.main(
        [
            "-o",
            str(filename),
            "-c",
            "nominal-cad-geometry",
            "--banks",
            "7",
            "--pixels-per-straw",
            "2",
            "--decimals",
            "4",
        ]
    )
    group = lnj.tree_nodes(json.loads(filename.read_text()))["loki_detector_7"]
    offsets = np.column_stack(
        [vnj._child(group, f"{axis}_pixel_offset")["config"]["values"] for axis in "xyz"]
    )
    assert np.allclose(offsets * 1e4, np.round(offsets * 1e4), atol=1e-6)
    exact = lbp.compute_bank_pixels(7, REFERENCE_BANKS_DATA[7], 2).offsets / 1000
    assert np.abs(offsets - exact).max() <= 0.5e-4 + 1e-12
    translation = vnj._child(vnj._child(group, "transformations"), "trans_8")
    assert translation["config"]["values"] == round(translation["config"]["values"], 4)


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


@pytest.mark.parametrize("decimals", ["3", "-1"])
def test_main_rejects_too_coarse_decimals(tmp_path, decimals):
    with pytest.raises(SystemExit, match="2"):
        lnj.main(["-o", str(tmp_path / "out.json"), "--banks", "2", "--decimals", decimals])
    assert not (tmp_path / "out.json").exists()


def test_main_bank_origin_centre_of_mass(tmp_path):
    """--bank-origin centre_of_mass: the translations (incl. detector_carriage_zero)
    are the centres of mass, the offsets are relative to them, the choice is
    recorded; the default is the first pixel end (corner A[0])."""
    cad = ("-c", "nominal-cad-geometry")
    default = run_main(tmp_path, *cad, "--pixels-per-straw", "2", "--banks", "0", "3")
    assert default["info"]["bank_origin"] == lbp.FIRST_PIXEL_END
    assert np.allclose(
        vnj.translation_mm(default[lnj.CARRIAGE_ZERO_NAME]), REFERENCE_BANKS_DATA[0]["A"][0]
    )
    data = run_main(
        tmp_path,
        *cad,
        "--pixels-per-straw",
        "2",
        "--banks",
        "0",
        "3",
        "--bank-origin",
        "centre_of_mass",
    )
    assert data["info"]["bank_origin"] == lbp.CENTRE_OF_MASS
    assert "centre of mass" in data["info"]["bank_translation"]
    expected = lbp.compute_instrument_pixels(
        {b: REFERENCE_BANKS_DATA[b] for b in (0, 3)}, 2, bank_origin=lbp.CENTRE_OF_MASS
    )
    assert np.allclose(
        vnj.translation_mm(data[lnj.CARRIAGE_ZERO_NAME]), expected[0].translation, atol=1e-5
    )
    trans_4 = child(child(data["loki_detector_3"], "transformations"), "trans_4")
    assert np.allclose(vnj.translation_mm(trans_4), expected[3].translation, atol=1e-5)
    offsets = np.column_stack(
        [values(child(data["loki_detector_3"], f"{axis}_pixel_offset")) for axis in "xyz"]
    )
    assert np.allclose(offsets.mean(axis=0), 0.0, atol=1e-8)  # pixel centre offsets [m]


# --- the main output: the measured geometry --------------------------------------------


@pytest.fixture(scope="module")
def default_json(tmp_path_factory):
    """loki-pixel-json with its defaults: the calibration of the default survey (SAM-606),
    256 pixels per straw."""
    path = tmp_path_factory.mktemp("default") / "pixels.json"
    with contextlib.redirect_stdout(io.StringIO()):
        assert lnj.main(["-o", str(path)]) == 0
    return path


def test_the_default_output_is_the_measured_geometry(default_json):
    """Survey -> calibration -> JSON: the pixel positions of the file (translation + offset +
    pixel shape centre, read back like a NeXus reader) equal the Geant4 pixel centres with the
    same calibration (G4GeoLoki, hard-coded) for banks 1-8, within the rounding of the file (8
    decimals in m). (Bank 0: the file has it at its carriage zero position, Geant4 at the rear
    detector distance; test_geometry_info compares it at the same distance.)"""
    banks = vnj.read_pixel_positions(str(default_json))
    checked = 0
    for pixel_id, expected in GEANT4_SAM606_PIXELS.items():
        bank_id = max(b for b in ALL_BANKS if lbp.bank_first_pixel_id(b) - 1 <= pixel_id)
        if bank_id == 0:
            continue  # the file has it at its carriage zero position, Geant4 at the rear distance
        ids, positions = banks[bank_id]
        index = pixel_id + 1 - ids[0]
        assert ids[index] == pixel_id + 1
        assert np.abs(positions[index] - np.array(expected)).max() < 1e-4, pixel_id
        checked += 1
    assert checked == 24


def test_the_default_output_records_its_calibration(default_json):
    from loki_geometry import __version__

    info = json.loads(default_json.read_text())["info"]
    assert info["calibration"] == "2026-September-SAM-606"
    assert info["written_by"] == f"loki-geometry {__version__}"
    sources = info["bank_corner_sources"]
    for bank_id in range(9):
        assert sources[str(bank_id)] == f"survey {SAM606.file_name} (sha256 {SAM606.checksum})"


def test_the_default_output_is_the_geometry_tool_positions(default_json):
    """Round trip (not a geometry check: both sides use the same corners and pixel code; the
    geometry is checked against Geant4 and loki.json elsewhere): the positions written to the
    file are those of loki-info / LokiGeometry with the default calibration (all banks)."""
    from loki_geometry.geometry_info import LokiGeometry

    geometry = LokiGeometry()
    for bank_id, (_ids, positions) in vnj.read_pixel_positions(str(default_json)).items():
        assert np.abs(positions - geometry.bank_pixel_centres(bank_id)).max() < 1e-4, bank_id


@pytest.mark.parametrize("calibration", ["nominal-cad-geometry", "nominal-geant4-geometry"])
def test_main_with_another_calibration(tmp_path, calibration):
    """Round trip for the other calibrations (the same code on both sides, see above)."""
    from loki_geometry.geometry_info import LokiGeometry

    data = run_main(tmp_path, "-c", calibration, "--pixels-per-straw", "2")
    assert data["info"]["calibration"] == calibration
    geometry = LokiGeometry(calibration, 2)
    banks = vnj.read_pixel_positions(str(tmp_path / "out.json"))
    for bank_id, (_ids, positions) in banks.items():
        assert np.abs(positions - geometry.bank_pixel_centres(bank_id)).max() < 1e-4


def test_main_rejects_an_unknown_calibration(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        lnj.main(["-o", str(tmp_path / "x.json"), "-c", "nominal-geometry"])
    assert error.value.code == 2
    assert "unknown calibration" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()


def test_main_writes_the_instrument_tree(tmp_path):
    """The file is the instrument tree of loki.json (/entry/instrument with the bank groups and,
    with bank 0, the detector carriage), and bank 0 depends on the carriage position, which
    depends on detector_carriage_zero, as in loki.json."""
    with contextlib.redirect_stdout(io.StringIO()):
        lnj.main(
            ["-o", str(tmp_path / "tree.json"), "--pixels-per-straw", "2", "--banks", "0", "5"]
        )
    tree = json.loads((tmp_path / "tree.json").read_text())
    (entry,) = tree["children"]
    assert entry["name"] == "entry" and vnj._attribute(entry, "NX_class") == "NXentry"
    (instrument,) = entry["children"]
    assert (
        instrument["name"] == "instrument"
        and vnj._attribute(instrument, "NX_class") == "NXinstrument"
    )
    assert [g["name"] for g in instrument["children"]] == [
        "loki_detector_0",
        "loki_detector_5",
        "detector_carriage",
    ]
    carriage = instrument["children"][2]
    assert vnj._attribute(carriage, "NX_class") == "NXpositioner"
    value = child(carriage, "value")
    assert vnj._attribute(value, "depends_on") == lnj.CARRIAGE_ZERO_PATH
    assert vnj._attribute(value, "vector") == [0.0, 0.0, 1.0]
    assert values(child(value, "value")) == 0.0
    assert values(child(instrument["children"][0], "depends_on")) == lnj.CARRIAGE_VALUE_PATH
    assert values(child(carriage, "depends_on")) == lnj.CARRIAGE_ZERO_PATH
    nodes = lnj.tree_nodes(tree)
    assert set(nodes) == {"info", "loki_detector_0", "loki_detector_5", lnj.CARRIAGE_ZERO_NAME}
    # the positions read from the tree (as from loki.json) are those of the nodes
    banks = vnj.read_pixel_positions(tmp_path / "tree.json")
    assert sorted(banks) == [0, 5]


def _positions_from_local_transformations(tree):
    """{bank_id: pixel positions [mm]} as a viewer that follows only the translations inside each
    group (as jsondisplay; a depends_on elsewhere counts as no translation)."""
    positions = {}
    for node in tree["children"][0]["children"][0]["children"]:
        if not node["name"].startswith("loki_detector_"):
            continue
        datasets = {
            c["config"]["name"]: c["config"]["values"]
            for c in node["children"]
            if c.get("module") == "dataset"
        }
        local = {}
        for child in node["children"]:
            if child.get("name") == "transformations":
                for t in child["children"]:
                    attributes = {a["name"]: a["values"] for a in t["attributes"]}
                    local[t["config"]["name"]] = (t["config"]["values"], attributes)
        translation, name = np.zeros(3), datasets["depends_on"].split("/")[-1]
        while name in local:
            value, attributes = local[name]
            assert attributes["transformation_type"] == "translation"
            translation += value * np.array(attributes["vector"])
            name = attributes["depends_on"].split("/")[-1]
        offsets = np.column_stack([datasets[f"{axis}_pixel_offset"] for axis in "xyz"])
        positions[int(node["name"].split("_")[-1])] = (offsets + translation) * 1000.0
    return positions


def test_visualize_writes_4_pixels_per_straw_and_bank_0_without_the_carriage(tmp_path):
    """--visualize: 4 pixels per straw (for geometry viewers); banks 1-8 as in the normal output,
    bank 0 placed by its own detector_carriage_zero (no detector carriage), so that a viewer that
    follows only the transformations inside a group places every bank right."""
    with contextlib.redirect_stdout(io.StringIO()):
        lnj.main(["-o", str(tmp_path / "view.json"), "--visualize", "--banks", "0", "1", "5"])
        lnj.main(
            [
                "-o",
                str(tmp_path / "plain.json"),
                "--pixels-per-straw",
                "4",
                "--banks",
                "0",
                "1",
                "5",
            ]
        )
    view = json.loads((tmp_path / "view.json").read_text())
    plain = json.loads((tmp_path / "plain.json").read_text())
    assert view["info"]["pixels_per_straw"] == 4
    view_nodes = {n["name"]: n for n in view["children"][0]["children"][0]["children"]}
    plain_nodes = {n["name"]: n for n in plain["children"][0]["children"][0]["children"]}
    assert set(view_nodes) == {"loki_detector_0", "loki_detector_1", "loki_detector_5"}
    assert "detector_carriage" in plain_nodes
    for name in ("loki_detector_1", "loki_detector_5"):
        assert view_nodes[name] == plain_nodes[name]
    depends_on = [
        c
        for c in view_nodes["loki_detector_0"]["children"]
        if c.get("config", {}).get("name") == "depends_on"
    ]
    assert [c["config"]["values"] for c in depends_on] == [
        f"{lnj.INSTRUMENT_PATH}/loki_detector_0/transformations/{lnj.CARRIAGE_ZERO_NAME}"
    ]
    # the same positions (loki-geometry reader), and also for a viewer without links between groups
    view_banks = vnj.read_pixel_positions(tmp_path / "view.json")
    plain_banks = vnj.read_pixel_positions(tmp_path / "plain.json")
    local = _positions_from_local_transformations(view)
    for bank_id in (0, 1, 5):
        np.testing.assert_array_equal(view_banks[bank_id][0], plain_banks[bank_id][0])
        np.testing.assert_allclose(view_banks[bank_id][1], plain_banks[bank_id][1], atol=1e-9)
        np.testing.assert_allclose(local[bank_id], plain_banks[bank_id][1], atol=1e-6)
    # the normal output is not usable that way for bank 0 (the reason for the change)
    assert (
        np.abs(_positions_from_local_transformations(plain)[0] - plain_banks[0][1]).max() > 1000.0
    )


def test_main_reports_errors_cleanly(tmp_path, capsys):
    """An output in a directory that does not exist: an error message (exit code 2), not a
    traceback."""
    with pytest.raises(SystemExit) as error:
        lnj.main(["-o", str(tmp_path / "no_dir" / "x.json"), "--banks", "3"])
    assert error.value.code == 2
    assert "cannot write" in capsys.readouterr().err


def test_info_records_the_rounding_and_the_carriage_only_with_bank_0(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        lnj.main(
            [
                "-o",
                str(tmp_path / "a.json"),
                "--pixels-per-straw",
                "2",
                "--banks",
                "3",
                "--decimals",
                "6",
            ]
        )
        lnj.main(["-o", str(tmp_path / "b.json"), "--pixels-per-straw", "2", "--banks", "0"])
    a = json.loads((tmp_path / "a.json").read_text())["info"]
    b = json.loads((tmp_path / "b.json").read_text())["info"]
    assert a["decimals"] == 6 and b["decimals"] == lnj.DEFAULT_DECIMALS
    assert "carriage_bank" not in a and "carriage_bank" in b


def test_a_calibrated_rear_bank_through_to_the_carriage_zero(tmp_path):
    """Bank 0 from tube data (calibration.py, here the Geant4 rear bank moved by (3, -2, 7) mm),
    through the corners and the JSON file: bank 0 is placed by the carriage (value 0 +
    detector_carriage_zero = its translation), and the positions read back from the file are
    those of the moved Geant4 bank."""
    import copy

    from loki_geometry import geant4 as g4
    from loki_geometry.calibration import Detectors
    from loki_geometry.geant4_example_data import example_calibration_data

    shift = np.array([3.0, -2.0, 7.0])
    data = copy.deepcopy([bank for bank in example_calibration_data if bank["bank_id"] == 0])
    for tube in data[0]["positions"]:
        for end in ("first_pixel_end", "last_pixel_end"):
            tube[end] = list(np.asarray(tube[end], dtype=float) + shift)
    with contextlib.redirect_stdout(io.StringIO()):
        detectors = Detectors(data, True, intended=None, run_checks=False)
    corners = detectors.formatted_data(as_straw_ends=True, decimals=None)
    pixels = lbp.compute_instrument_pixels(corners, 2)
    filename = tmp_path / "bank0.json"
    lnj.write_json(lnj.instrument_tree(lnj.build_output(pixels, "test")), filename)
    tree = json.loads(filename.read_text())
    carriage_zero = lnj.tree_nodes(tree)[lnj.CARRIAGE_ZERO_NAME]
    assert np.allclose(vnj.translation_mm(carriage_zero), pixels[0].translation, atol=1e-5)
    ids, positions = vnj.read_pixel_positions(filename)[0]
    expected = (
        np.array([g4.pixel_centre(int(i) - 1, 2) for i in ids]) + shift
    )  # the moved Geant4 bank
    assert np.abs(positions - expected).max() < 1e-4
