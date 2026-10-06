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
    """Corners calibrated from the package survey (banks 1-8; bank 0 was not measured)."""
    with contextlib.redirect_stdout(io.StringIO()):
        data = survey.get_measured_calibration_data(SAM606.path, verbose=False)
        detectors = calibration.Detectors(data, True, run_checks=False)
    return detectors.formatted_data()


def test_measured_corners_json_without_bank_0(tmp_path, measured_corners):
    """Measured data lacks bank 0: no detector_carriage_zero, global ids kept,
    and the JSON reproduces the model positions."""
    pixels = lbp.compute_instrument_pixels(measured_corners, 2)
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
    return json.loads(filename.read_text())


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
    data = json.loads(filename.read_text())
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
    group = json.loads(filename.read_text())["loki_detector_7"]
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
    same calibration (G4GeoLoki, hard-coded) for the surveyed banks 1-8, within the rounding of
    the file (8 decimals in m)."""
    banks = vnj.read_pixel_positions(str(default_json))
    checked = 0
    for pixel_id, expected in GEANT4_SAM606_PIXELS.items():
        bank_id = max(b for b in ALL_BANKS if lbp.bank_first_pixel_id(b) - 1 <= pixel_id)
        if bank_id == 0:
            continue  # not surveyed: the CAD geometry here, Geant4 puts it at the rear distance
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
    assert sources["0"].startswith("REFERENCE_BANKS_DATA")  # bank 0 was not surveyed
    for bank_id in range(1, 9):
        assert sources[str(bank_id)] == f"survey {SAM606.file_name} (sha256 {SAM606.checksum})"


def test_the_default_output_is_the_geometry_tool_positions(default_json):
    """The same positions as loki-info / LokiGeometry with the default calibration (all banks)."""
    from loki_geometry.geometry_info import LokiGeometry

    geometry = LokiGeometry()
    for bank_id, (_ids, positions) in vnj.read_pixel_positions(str(default_json)).items():
        assert np.abs(positions - geometry.bank_pixel_centres(bank_id)).max() < 1e-4, bank_id


@pytest.mark.parametrize("calibration", ["nominal-cad-geometry", "nominal-geant4-geometry"])
def test_main_with_another_calibration(tmp_path, calibration):
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


def test_visualize_writes_a_nexus_tree(tmp_path):
    """--visualize: the bank groups inside /entry/instrument (for geometry viewers), with 4
    pixels per straw; the same groups as without the option."""
    with contextlib.redirect_stdout(io.StringIO()):
        lnj.main(["-o", str(tmp_path / "view.json"), "--visualize", "--banks", "1", "5"])
        lnj.main(
            ["-o", str(tmp_path / "plain.json"), "--pixels-per-straw", "4", "--banks", "1", "5"]
        )
    view = json.loads((tmp_path / "view.json").read_text())
    plain = json.loads((tmp_path / "plain.json").read_text())
    (entry,) = view["children"]
    assert entry["name"] == "entry" and vnj._attribute(entry, "NX_class") == "NXentry"
    (instrument,) = entry["children"]
    assert (
        instrument["name"] == "instrument"
        and vnj._attribute(instrument, "NX_class") == "NXinstrument"
    )
    assert [g["name"] for g in instrument["children"]] == ["loki_detector_1", "loki_detector_5"]
    assert instrument["children"] == [plain["loki_detector_1"], plain["loki_detector_5"]]
    assert view["info"]["pixels_per_straw"] == 4
