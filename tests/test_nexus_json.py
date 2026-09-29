"""Tests of the NeXus-JSON export and of its verification against Geant4."""

import json

import numpy as np
import pytest

from loki_geometry import bank_pixels as lbp
from loki_geometry import geant4 as g4
from loki_geometry import nexus_json as lnj
from loki_geometry import verify_json as vnj
from loki_geometry.reference_data import REFERENCE_BANKS_DATA


def child(node, name):
    return vnj._child(node, name)


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


def test_pixel_shape_is_centred_on_pixel_centres():
    bank = lbp.compute_bank_pixels(5, REFERENCE_BANKS_DATA[5], 2, pixel_reference=lbp.PIXEL_CENTRE)
    vertices = shape_vertices_mm(bank)
    assert np.allclose(vertices[0] + vertices[2], 0.0, atol=1e-5)


def test_pixel_shape_starts_at_pixel_bases():
    bank = lbp.compute_bank_pixels(5, REFERENCE_BANKS_DATA[5], 2, pixel_reference=lbp.PIXEL_BASE)
    vertices = shape_vertices_mm(bank)
    assert np.allclose(vertices[0], 0.0, atol=1e-5)


def test_round_trip_through_a_file(tmp_path, reference_pixels):
    filename = tmp_path / "pixels.json"
    lnj.write_json(lnj.build_output(reference_pixels, "test"), filename)
    banks = vnj.read_pixel_positions(filename)
    assert sorted(banks) == list(range(9))
    for bank_id, (ids, positions) in banks.items():
        assert np.array_equal(ids, reference_pixels[bank_id].pixel_ids)
        assert (
            np.abs(positions - reference_pixels[bank_id].positions).max() < 1e-4
        )  # 1e-8 m rounding


@pytest.mark.parametrize("pixel_reference", lbp.PIXEL_REFERENCES)
def test_verification_of_geant4_corners_passes(tmp_path, pixel_reference):
    """Both pixel reference conventions describe the same pixel centres."""
    corners = {bank_id: g4.bank_corners(bank_id, 2) for bank_id in (0, 2, 7)}
    filename = tmp_path / "pixels.json"
    pixels = lbp.compute_instrument_pixels(corners, 2, pixel_reference=pixel_reference)
    lnj.write_json(lnj.build_output(pixels, "geant4"), filename)
    results = vnj.compare_with_geant4(vnj.read_pixel_positions(filename), 2)
    assert max(deviation for deviation, _ in results.values()) < 1e-4


def test_verification_detects_an_inconsistent_pixel_shape(tmp_path):
    """Pixel centre offsets with a pixel_base shape (as in the current loki.json)."""
    corners = {6: g4.bank_corners(6, 2)}
    bank_centre = lbp.compute_instrument_pixels(corners, 2, pixel_reference=lbp.PIXEL_CENTRE)
    bank_base = lbp.compute_instrument_pixels(corners, 2, pixel_reference=lbp.PIXEL_BASE)
    output = lnj.build_output(bank_centre, "geant4")
    shape = child(output["loki_detector_6"], "pixel_shape")
    shape["children"] = child(lnj.bank_group(bank_base[6]), "pixel_shape")["children"]
    filename = tmp_path / "pixels.json"
    lnj.write_json(output, filename)
    deviation, _ = vnj.compare_with_geant4(vnj.read_pixel_positions(filename), 2)[6]
    assert deviation == pytest.approx(bank_centre[6].pixel_length / 2, abs=1e-4)
    ignoring_shape, _ = vnj.compare_with_geant4(
        vnj.read_pixel_positions(filename, use_pixel_shape=False), 2
    )[6]
    assert ignoring_shape < 1e-4


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
