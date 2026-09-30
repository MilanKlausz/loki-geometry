"""Tests of the verification of NeXus-JSON pixel positions against Geant4 (verify_json.py)."""

import json

import numpy as np
import pytest

from loki_geometry import bank_pixels as lbp
from loki_geometry import geant4 as g4
from loki_geometry import nexus_json as lnj
from loki_geometry import verify_json as vnj
from loki_geometry.reference_data import REFERENCE_BANKS_DATA


@pytest.fixture(scope="module")
def reference_pixels():
    return lbp.compute_instrument_pixels(REFERENCE_BANKS_DATA, pixels_per_straw=2)


def child(node, name):
    return vnj._child(node, name)


def _own_output(banks, pixels_per_straw=2):
    from loki_geometry import bank_pixels as lbp
    from loki_geometry import nexus_json as lnj

    corners = {b: g4.bank_corners(b, pixels_per_straw) for b in banks}
    pixels = lbp.compute_instrument_pixels(corners, pixels_per_straw)
    return lnj.build_output(pixels, "test")


def _as_instrument_file(output):
    """The same groups arranged like a full instrument file (loki.json):
    /entry/instrument/{detector_carriage/transformations/..., loki_detector_N}."""
    banks = [node for key, node in output.items() if key.startswith("loki_detector")]
    instrument = banks
    if vnj.CARRIAGE_ZERO_NAME in output:
        transformations = {
            "name": "transformations",
            "type": "group",
            "children": [output[vnj.CARRIAGE_ZERO_NAME]],
        }
        carriage = {"name": "detector_carriage", "type": "group", "children": [transformations]}
        instrument = [carriage] + banks
    instrument_group = {"name": "instrument", "type": "group", "children": instrument}
    return {"children": [{"name": "entry", "type": "group", "children": [instrument_group]}]}


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


@pytest.mark.parametrize("layout", ["own", "instrument"])
def test_read_pixel_positions_layouts(tmp_path, layout):
    output = _own_output((0, 6))
    data = output if layout == "own" else _as_instrument_file(output)
    filename = tmp_path / "pixels.json"
    filename.write_text(json.dumps(data))
    banks = vnj.read_pixel_positions(filename)
    assert sorted(banks) == [0, 6]
    results = vnj.compare_with_geant4(banks, 2)
    assert max(deviation for deviation, _ in results.values()) < 1e-4


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


def test_verification_fails_for_a_shifted_bank_and_units_are_mm(tmp_path):
    output = _own_output((3,))
    group = output["loki_detector_3"]
    offsets = vnj._config(vnj._child(group, "y_pixel_offset"))
    offsets["values"] = [v + 0.002 for v in offsets["values"]]  # +2 mm
    filename = tmp_path / "pixels.json"
    filename.write_text(json.dumps(output))
    deviation, _ = vnj.compare_with_geant4(vnj.read_pixel_positions(filename), 2)[3]
    assert deviation == pytest.approx(2.0, abs=1e-3)
    assert vnj.main([str(filename), "--tolerance", "1.0"]) == 1
    assert vnj.main([str(filename), "--tolerance", "2.1"]) == 0


def test_pixels_per_straw_is_deduced_per_file(tmp_path, capsys):
    """Deduction from the first bank (here a vertical, 500 mm bank with 3 px)."""
    assert vnj.main([_write(tmp_path, _output((4, 8), n=3))]) == 0
    assert "3 pixels per straw" in capsys.readouterr().out


def test_verify_json_wrong_pixels_per_straw_is_an_error(tmp_path, capsys):
    filename = str(tmp_path / "p.json")
    lnj.main(["-o", filename, "--pixels-per-straw", "2", "--banks", "3"])
    with pytest.raises(SystemExit):
        vnj.main([filename, "--pixels-per-straw", "4"])
    assert "pixel ids do not match" in capsys.readouterr().err


def test_bank_0_without_carriage_is_an_error(tmp_path):
    output = _own_output((0,))
    del output[vnj.CARRIAGE_ZERO_NAME]
    filename = tmp_path / "pixels.json"
    filename.write_text(json.dumps(output))
    with pytest.raises(ValueError):
        vnj.read_pixel_positions(filename)


@pytest.mark.parametrize("extra_args", [[], ["--pixels-per-straw", "2"]])
def test_file_without_banks_does_not_pass_verification(tmp_path, capsys, extra_args):
    filename = tmp_path / "empty.json"
    filename.write_text(json.dumps({"info": {}}))
    with pytest.raises(SystemExit) as error:
        vnj.main([str(filename)] + extra_args)
    assert error.value.code == 2
    assert "no LOKI detector groups" in capsys.readouterr().err
    with pytest.raises(ValueError, match="no LOKI detector groups"):
        vnj.read_pixel_positions(filename)


def _output(banks, n=2):
    corners = {b: g4.bank_corners(b, n) for b in banks}
    return lnj.build_output(lbp.compute_instrument_pixels(corners, n), "test")


def _write(tmp_path, data):
    filename = tmp_path / "pixels.json"
    filename.write_text(json.dumps(data))
    return str(filename)


def _without(output, bank, names):
    group = output[f"loki_detector_{bank}"]
    group["children"] = [
        ch
        for ch in group["children"]
        if (ch.get("name") or vnj._config(ch).get("name")) not in names
    ]
    return output


@pytest.mark.parametrize(
    "output",
    [
        # event-only bank groups, as in loki-dynamic.json (no pixel datasets)
        pytest.param(
            lambda: _without(
                _output((3,)),
                3,
                {
                    "detector_number",
                    "x_pixel_offset",
                    "y_pixel_offset",
                    "z_pixel_offset",
                    "pixel_shape",
                },
            ),
            id="no-pixel-datasets",
        ),
        pytest.param(lambda: _without(_output((3,)), 3, {"pixel_shape"}), id="no-pixel-shape"),
        pytest.param(
            lambda: _without(_output((3,)), 3, {"transformations"}), id="no-transformations"
        ),
        pytest.param(lambda: _without(_output((3,)), 3, {"depends_on"}), id="no-depends_on"),
    ],
)
def test_malformed_bank_groups_give_a_clean_error(tmp_path, capsys, output):
    """A bank group without a part of its geometry (e.g. an event-only group) is a
    usage error (exit code 2) naming the group, not a traceback."""
    filename = _write(tmp_path, output())
    with pytest.raises(SystemExit) as error:
        vnj.main([filename])
    assert error.value.code == 2
    assert "loki_detector_3" in capsys.readouterr().err


def test_carriage_group_without_transformations_gives_a_clean_error(tmp_path):
    """An instrument file whose detector_carriage has no transformations group:
    bank 0 depends on the carriage, so its translation is missing (ValueError)."""
    output = _output((0,))
    output.pop(vnj.CARRIAGE_ZERO_NAME)
    carriage = {"name": "detector_carriage", "type": "group", "children": []}
    instrument = {
        "name": "instrument",
        "type": "group",
        "children": [carriage, output["loki_detector_0"]],
    }
    data = {"children": [{"name": "entry", "type": "group", "children": [instrument]}]}
    with pytest.raises(ValueError, match="detector_carriage_zero"):
        vnj.read_pixel_positions(_write(tmp_path, data))


def test_verify_json_cli_reports_a_missing_file_cleanly(tmp_path):
    with pytest.raises(SystemExit):
        vnj.main([str(tmp_path / "missing.json")])


@pytest.mark.parametrize("pixel_reference", lbp.PIXEL_REFERENCES)
def test_both_bank_origins_give_the_same_positions_in_the_json_file(tmp_path, pixel_reference):
    """Written to a file and read back, both bank origins give the same pixel
    positions, which agree with Geant4."""
    corners = {bank_id: g4.bank_corners(bank_id, 2) for bank_id in (0, 2, 5, 8)}
    positions = {}
    for origin in lbp.BANK_ORIGINS:
        pixels = lbp.compute_instrument_pixels(
            corners, 2, pixel_reference=pixel_reference, bank_origin=origin
        )
        filename = tmp_path / f"{origin}.json"
        lnj.write_json(lnj.build_output(pixels, "geant4"), filename)
        positions[origin] = vnj.read_pixel_positions(filename)
        results = vnj.compare_with_geant4(positions[origin], 2)
        assert max(deviation for deviation, _ in results.values()) < 1e-4
    for bank_id in corners:
        ids_a, pos_a = positions[lbp.FIRST_PIXEL_END][bank_id]
        ids_b, pos_b = positions[lbp.CENTRE_OF_MASS][bank_id]
        assert np.array_equal(ids_a, ids_b)
        assert np.allclose(pos_a, pos_b, rtol=0, atol=2e-5)  # 1e-8 m rounding of each part
