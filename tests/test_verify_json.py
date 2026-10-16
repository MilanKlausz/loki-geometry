"""Tests of the verification of NeXus-JSON pixel positions against Geant4 (verify_json.py)."""

import contextlib
import io
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
    /entry/instrument/{detector_carriage (with value and transformations), loki_detector_N},
    the carriage first and without the info of the output."""
    tree = lnj.instrument_tree({"info": {}, **output})
    del tree["info"]
    instrument = tree["children"][0]["children"][0]
    instrument["children"] = instrument["children"][-1:] + instrument["children"][:-1]
    return tree


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
    del output[lnj.CARRIAGE_ZERO_NAME]
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
    output.pop(lnj.CARRIAGE_ZERO_NAME)
    carriage = {"name": "detector_carriage", "type": "group", "children": []}
    instrument = {
        "name": "instrument",
        "type": "group",
        "children": [carriage, output["loki_detector_0"]],
    }
    data = {"children": [{"name": "entry", "type": "group", "children": [instrument]}]}
    with pytest.raises(ValueError, match="detector_carriage/value not found"):
        vnj.read_pixel_positions(_write(tmp_path, data))


def test_verify_json_cli_reports_a_missing_file_cleanly(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        vnj.main([str(tmp_path / "missing.json")])
    assert error.value.code == 2
    assert "cannot read" in capsys.readouterr().err


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


# --- the reference: the calibration of the file --------------------------------------


@pytest.fixture(scope="module")
def measured_json(tmp_path_factory):
    """A loki-pixel-json output of the measured geometry (the default calibration)."""
    path = tmp_path_factory.mktemp("measured") / "measured.json"
    with contextlib.redirect_stdout(io.StringIO()):
        assert lnj.main(["-o", str(path), "--pixels-per-straw", "2"]) == 0
    return path


def test_a_file_is_compared_with_its_recorded_calibration(measured_json, capsys):
    """The calibration recorded in the file is used (a round trip: the same corners and pixel code
    on both sides; the agreement with Geant4 is tested in test_geometry_info / test_nexus_json)."""
    assert vnj.main([str(measured_json), "--tolerance", "0.001"]) == 0
    out = capsys.readouterr().out
    assert "deviation from the 2026-September-SAM-606 pixel centres" in out
    assert "> tolerance" not in out


def test_a_file_without_calibration_is_compared_with_the_nominal_geant4_geometry(
    measured_json, tmp_path, capsys
):
    """E.g. loki.json: the nominal Geant4 geometry; the measured banks deviate from it."""
    data = json.loads(measured_json.read_text())
    del data["info"]
    path = tmp_path / "no_info.json"
    path.write_text(json.dumps(data))
    assert vnj.main([str(path)]) == 1
    out = capsys.readouterr().out
    assert "deviation from the nominal-geant4-geometry pixel centres" in out
    assert "bank 2: largest 3" in out  # the middle left bank is ~32 mm farther than nominal


def test_the_calibration_option_overrides_the_file(measured_json, capsys):
    assert vnj.main([str(measured_json), "-c", "nominal-cad-geometry"]) == 1
    assert "deviation from the nominal-cad-geometry pixel centres" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        vnj.main([str(measured_json), "-c", "unknown"])
    assert error.value.code == 2
    assert "unknown calibration" in capsys.readouterr().err


# ---- the depends_on chain (as in the instrument file, the file to verify after the update) ----


def _tree(banks, n=2):
    return lnj.instrument_tree(_output(banks, n))


def _node(tree, path):
    return vnj.node_index(tree)[path]


def _set_attribute(node, name, value):
    for attribute in node.setdefault("attributes", []):
        if attribute["name"] == name:
            attribute["values"] = value
            return
    node["attributes"].append({"name": name, "values": value})


TRANS_2 = "/entry/instrument/loki_detector_1/transformations/trans_2"
CARRIAGE_VALUE = "/entry/instrument/detector_carriage/value"


def _depends_on_dataset(tree, bank):
    return vnj._child(_node(tree, f"/entry/instrument/loki_detector_{bank}"), "depends_on")


@pytest.mark.parametrize(
    "change, message",
    [
        (
            lambda t: _set_attribute(_node(t, TRANS_2), "transformation_type", "rotation"),
            "only translations",
        ),
        (
            lambda t: _set_attribute(_node(t, TRANS_2), "vector", [0.0, 0.0, 2.0]),
            "not a unit vector",
        ),
        (lambda t: _set_attribute(_node(t, TRANS_2), "units", "s"), "not a length unit"),
        (lambda t: _set_attribute(_node(t, TRANS_2), "depends_on", TRANS_2), "loops"),
        (
            lambda t: _node(t, TRANS_2)["attributes"].remove(
                next(a for a in _node(t, TRANS_2)["attributes"] if a["name"] == "depends_on")
            ),
            "has no depends_on",
        ),
        (
            lambda t: _depends_on_dataset(t, 1)["config"].update(values=TRANS_2 + "x"),
            "not found",
        ),
        (
            lambda t: t["children"][0]["children"][0]["children"].append(
                json.loads(json.dumps(_node(t, "/entry/instrument/loki_detector_1")))
            ),
            "two nodes",
        ),
    ],
)
def test_a_broken_transformation_chain_is_an_error(tmp_path, change, message):
    tree = _tree((0, 1, 2))
    change(tree)
    with pytest.raises(ValueError, match=message):
        vnj.read_pixel_positions(_write(tmp_path, tree))


@pytest.mark.parametrize(
    "change, banks_off",
    [
        # the translation in mm instead of m: 3 mm from the sample instead of 3 m
        (lambda t: _set_attribute(_node(t, TRANS_2), "units", "mm"), {1}),
        # bank 2 placed by the translation of bank 1
        (lambda t: _depends_on_dataset(t, 2)["config"].update(values=TRANS_2), {2}),
        # the carriage position does not depend on detector_carriage_zero
        (lambda t: _set_attribute(_node(t, CARRIAGE_VALUE), "depends_on", "."), {0}),
        # the carriage moved by 100 mm
        (
            lambda t: _node(t, CARRIAGE_VALUE + "/value")["config"].update(values=100.0),
            {0},
        ),
    ],
)
def test_a_wrong_transformation_chain_shows_as_a_deviation(tmp_path, capsys, change, banks_off):
    tree = _tree((0, 1, 2))
    change(tree)
    assert vnj.main([_write(tmp_path, tree), "-c", "nominal-geant4-geometry"]) == 1
    flagged = {
        int(line.split()[1].rstrip(":"))
        for line in capsys.readouterr().out.splitlines()
        if "> tolerance" in line
    }
    assert flagged == banks_off


def test_swapped_pixel_ids_in_a_bank_are_an_error(tmp_path):
    output = _output((3,))
    ids = vnj._child(output["loki_detector_3"], "detector_number")["config"]["values"]
    ids[5], ids[6] = ids[6], ids[5]
    with pytest.raises(SystemExit):
        vnj.main([_write(tmp_path, output), "-c", "nominal-geant4-geometry"])


def test_the_deviation_is_the_3d_distance(tmp_path, capsys):
    """Offsets shifted by (1, 1, 1) mm: the reported deviation is sqrt(3) mm."""
    output = _output((3,))
    for axis in "xyz":
        offsets = vnj._child(output["loki_detector_3"], f"{axis}_pixel_offset")["config"]
        offsets["values"] = [v + 0.001 for v in offsets["values"]]
    assert vnj.main([_write(tmp_path, output), "-c", "nominal-geant4-geometry"]) == 1
    assert "largest 1.7321" in capsys.readouterr().out


def test_ignore_pixel_shape_on_the_command_line(tmp_path, capsys):
    """A pixel_base file: with the pixel shape the positions agree; without it
    (--ignore-pixel-shape) they are half a pixel off (bank 3: 1000 mm / 4 pixels -> 125 mm)."""
    filename = str(tmp_path / "base.json")
    with contextlib.redirect_stdout(io.StringIO()):
        lnj.main(
            [
                "-o",
                filename,
                "--pixels-per-straw",
                "4",
                "--banks",
                "3",
                "--pixel-reference",
                "pixel_base",
            ]
        )
    assert vnj.main([filename]) == 0
    assert "largest 0.0000" in capsys.readouterr().out
    assert vnj.main([filename, "--ignore-pixel-shape", "--tolerance", "200"]) == 0
    assert "largest 125.0000" in capsys.readouterr().out


def test_a_file_with_the_nodes_at_the_top_level_can_still_be_read(tmp_path, reference_pixels):
    """The layout of older loki-pixel-json versions (the nodes at the top level)."""
    filename = tmp_path / "old.json"
    filename.write_text(json.dumps(lnj.build_output(reference_pixels, "test")))
    banks = vnj.read_pixel_positions(filename)
    assert sorted(banks) == list(range(9))
    for bank_id, (_, positions) in banks.items():
        assert np.abs(positions - reference_pixels[bank_id].positions).max() < 1e-4


def test_a_relative_depends_on_of_a_bank_group_is_resolved_in_the_group(tmp_path):
    """NeXus allows a depends_on relative to its group: the same positions as the absolute one."""
    tree = _tree([1])
    absolute = vnj.pixel_positions_from_tree(tree)[1][1]
    _depends_on_dataset(tree, 1)["config"]["values"] = "transformations/trans_2"
    assert np.array_equal(vnj.pixel_positions_from_tree(tree)[1][1], absolute)


def test_bank_0_with_the_carriage_at_its_survey_position_is_where_it_was_measured(measured_json):
    """Survey -> JSON -> detector carriage value: with the carriage position of the survey, the
    rear bank of the file is where it was measured. Tube 0 (centre straw) is compared with the
    measured tube C3P1 (trimmed to the straw length) within the residuals of the bank model
    (< 1 mm: the measured tubes are 1.4 mm farther apart than 55 tube pitches)."""
    from loki_geometry import survey
    from loki_geometry import survey_files as sf

    tree = json.loads(measured_json.read_text())
    at_zero = vnj.pixel_positions_from_tree(tree)[0][1]
    vnj.node_index(tree)[f"{CARRIAGE_VALUE}/value"]["config"]["values"] = (
        sf.SAM606.carriage_position
    )
    moved = vnj.pixel_positions_from_tree(tree)[0][1]
    assert np.allclose(moved - at_zero, [0.0, 0.0, sf.SAM606.carriage_position], atol=1e-6)
    raw = survey.read_measured_points(sf.SAM606.path)
    first, last = (sf.tank_window_frame(raw[label]) for label in ("C3P1I", "C3P1II"))
    first, last, _, _ = survey.find_effective_tube_ends(first, last, 1000.0)
    n = 2
    expected = first + (last - first) * ((np.arange(n) + 0.5) / n)[:, None]
    tube_0_centre_straw = moved[3 * n : 4 * n]
    assert np.linalg.norm(tube_0_centre_straw - expected, axis=1).max() < 1.0
