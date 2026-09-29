"""Tests of the command-line entry points, the README claims and the packaging metadata.

Fast smoke tests of every console script (small options only), the README
Python snippet and example commands, the numbers stated in the README, the
package exports and the package data.
"""

import importlib
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

import loki_geometry
from loki_geometry import calibration, survey
from loki_geometry import constants as c
from loki_geometry import geant4 as g4
from loki_geometry import nexus_json as lnj
from loki_geometry import nominal_corners as nc
from loki_geometry import verify_json as vnj
from loki_geometry.geant4_example_data import example_calibration_data
from loki_geometry.reference_data import REFERENCE_BANKS_DATA

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(REPO_DIR, "README.md")
PYPROJECT = os.path.join(REPO_DIR, "pyproject.toml")
SRC_DIR = os.path.join(REPO_DIR, "src")


def read(path):
    with open(path, encoding="utf-8") as file:
        return file.read()


def console_scripts():
    """{script name: "module:function"} from [project.scripts] (no tomllib on 3.9/3.10)."""
    text = read(PYPROJECT)
    section = re.search(r"^\[project\.scripts\]\n(.*?)(?=^\[)", text, re.S | re.M).group(1)
    return dict(re.findall(r'^([\w-]+)\s*=\s*"([\w.:]+)"', section, re.M))


def readme_code_blocks(language=""):
    return re.findall(r"^```" + language + r"\n(.*?)^```", read(README), re.S | re.M)


def readme_commands():
    """README example command lines that start with a console script name."""
    scripts = console_scripts()
    lines = []
    for block in readme_code_blocks():
        for line in block.splitlines():
            line = line.split("#", 1)[0].strip()
            if line.split(" ", 1)[0] in scripts:
                lines.append(line)
    return lines


def run_entry_point(script, args, monkeypatch, capsys):
    """Call the console script function like the installed wrapper does
    (sys.exit(func()) with sys.argv set); returns (exit code, stdout)."""
    module_name, function_name = console_scripts()[script].split(":")
    function = getattr(importlib.import_module(module_name), function_name)
    monkeypatch.setattr(sys, "argv", [script, *args])
    try:
        code = function()
    except SystemExit as exit:
        code = exit.code
    return code or 0, capsys.readouterr().out


def run_module(module, *args, cwd=None):
    env = dict(os.environ, PYTHONPATH=SRC_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run(
        [sys.executable, "-m", f"loki_geometry.{module}", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=60,
    )


# --- packaging -----------------------------------------------------------------


def test_console_scripts_point_to_callables():
    scripts = console_scripts()
    assert set(scripts) == {
        "loki-calibrate",
        "loki-survey",
        "loki-pixel-json",
        "loki-verify-json",
        "loki-geant4-validate",
        "loki-nominal-corners",
    }
    for target in scripts.values():
        module_name, function_name = target.split(":")
        assert callable(getattr(importlib.import_module(module_name), function_name))


def test_package_data_is_present_and_complete():
    """The survey file (package data) has the 32 points of README: 2 tubes x 2 ends, banks 1-8."""
    assert os.path.dirname(survey.DEFAULT_SURVEY_FILE).startswith(
        os.path.dirname(loki_geometry.__file__)
    )
    points = survey.read_measured_points()
    assert len(points) == 32
    assert re.search(r'loki_geometry\s*=\s*\["data/\*\.txt"\]', read(PYPROJECT))


def test_init_exports_what_the_readme_uses():
    for name in (
        "REFERENCE_BANKS_DATA",
        "compute_instrument_pixels",
        "compute_bank_pixels",
        "BankPixels",
        "BankFrame",
        "PIXEL_CENTRE",
        "PIXEL_BASE",
        "constants",
    ):
        assert hasattr(loki_geometry, name), name
    assert re.fullmatch(r"\d+\.\d+\.\d+", loki_geometry.__version__)


# --- README consistency ------------------------------------------------------------


def test_readme_module_table_lists_every_module_and_script():
    text = read(README)
    modules = sorted(
        name[:-3]
        for name in os.listdir(os.path.join(SRC_DIR, "loki_geometry"))
        if name.endswith(".py") and name != "__init__.py"
    )
    for module in modules:
        assert f"| `{module}`" in text, module
    for script, target in console_scripts().items():
        module = target.split(":")[0].rsplit(".", 1)[1]
        assert f"| `{module}` (`{script}`)" in text, script


def test_readme_numbers_match_the_constants():
    text = " ".join(read(README).split())  # README lines are wrapped
    assert sum(c.TUBES_PER_BANK) * c.STRAWS_PER_TUBE * c.DEFAULT_PIXELS_PER_STRAW == 1_605_632
    assert f"{c.TUBE_GRID_SIDE:.2f}" in text  # 28.40
    assert f"{c.FRONT_TO_BACK_NORMAL:.2f}" in text  # 78.78
    assert f"{c.FRONT_TO_BACK_IN_PLANE:.2f}" in text  # 18.84
    assert f"({c.TUBE_OUTER_RADIUS} mm)" in text
    assert f"({c.STRAW_RADIUS} mm)" in text
    assert f"{c.STRAW_STRAW_DISTANCE} mm to the side" in text
    offsets = [next(v for v in c.BANK_OFFSETS[b] if v) for b in (5, 6, 7, 8)]
    assert offsets == [100.0, -15.0, -100.0, 50.0]  # "+100 / −15 / −100 / +50 mm for banks 5–8"
    assert c.VERTICAL_BANK_IDS == (2, 4, 6, 8)
    assert c.UPSIDE_DOWN_BANK_IDS == (1, 2, 5, 6)
    assert survey.SAMPLE_TO_TANK_WINDOW_DISTANCE == 100.0


def test_readme_half_pixel_shifts_cover_all_banks():
    text = read(README)
    for half_pixel in sorted({length / 512 / 2 for length in c.STRAW_LENGTHS}):
        assert f"{half_pixel:.2f}" in text, f"{half_pixel:.2f} mm missing"


def test_readme_python_snippet_runs(capsys):
    # the usage example (the other Python block, the survey entry example, is a template)
    (snippet,) = [block for block in readme_code_blocks("python") if "Detectors(" in block]
    namespace = {}
    exec(compile(snippet, "README.md", "exec"), namespace)
    assert namespace["x"] == pytest.approx(498.046875)  # pixel 425472 of _SB_LOKI_INFO_REFERENCE
    assert sorted(namespace["corners"]) == list(range(1, 9))
    assert sorted(namespace["pixels"]) == list(range(9))
    assert "Corner positions compared to intended" in capsys.readouterr().out


def test_readme_command_flags_exist(monkeypatch, capsys):
    """Every option used in the README examples is accepted by that command."""
    help_texts = {}
    for line in readme_commands():
        script, *args = line.split()
        options = [a for a in args if a.startswith("-")]
        if not options:
            continue
        if script not in help_texts:
            code, help_texts[script] = run_entry_point(script, ["--help"], monkeypatch, capsys)
            assert code == 0
        for option in options:
            assert option in help_texts[script], f"{script} {option}"


# --- command line smoke tests --------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ["--data", "reference", "--verbose"],
        ["--data", "example", "--verbose"],
        ["--data", "measured", "--verbose", "--compare-to-intended", "--print"],
        ["--sample-to-window-distance", "119", "--verbose"],
    ],
)
def test_loki_calibrate_readme_examples(args, monkeypatch, capsys):
    code, out = run_entry_point("loki-calibrate", args, monkeypatch, capsys)
    assert code == 0
    assert "=== Quality report ===" in out
    if "--print" in args:
        assert "'A': [" in out and "Corner positions compared to intended" in out
    if "119" in args:
        assert "sample to tank window distance 119.0 mm" in out


def test_loki_calibrate_reference_self_test_reports_zero_differences(monkeypatch, capsys):
    """The quality report compares the reference data with itself: differences 0.00
    (the comparison with REFERENCE_BANKS_DATA itself is a separate test)."""
    code, out = run_entry_point(
        "loki-calibrate", ["--data", "reference", "--verbose"], monkeypatch, capsys
    )
    summary = out.split("Summary (measured - intended")[1]
    rows = [line for line in summary.splitlines() if re.match(r"\s+\d \|", line)]
    assert len(rows) == 9
    for row in rows:
        assert "+0.000" in row and "+0.00 |" in row


def test_loki_survey_and_geant4_validate(monkeypatch, capsys):
    code, out = run_entry_point("loki-survey", [], monkeypatch, capsys)
    assert code == 0 and "Calibration data:" in out and "WARNING" not in out
    code, out = run_entry_point("loki-geant4-validate", [], monkeypatch, capsys)
    assert code == 0
    deviation = float(re.search(r"values: (\S+) mm", out).group(1))
    assert deviation < 1e-6  # README: "matches sb_loki_info to 1e-6 mm"


def test_loki_nominal_corners(monkeypatch, capsys):
    code, out = run_entry_point("loki-nominal-corners", [], monkeypatch, capsys)
    assert code == 0
    assert out.count("Arrays match within tolerance") == 2 * c.NUMBER_OF_BANKS
    code, out = run_entry_point("loki-nominal-corners", ["3"], monkeypatch, capsys)
    assert out.count("Arrays match within tolerance") == 2


def test_loki_nominal_corners_help(monkeypatch, capsys):
    code, out = run_entry_point("loki-nominal-corners", ["--help"], monkeypatch, capsys)
    assert code == 0 and "usage" in out


def test_loki_geant4_validate_fails_on_deviation(monkeypatch, capsys):
    monkeypatch.setattr(g4, "_SB_LOKI_INFO_REFERENCE", {425472: (0.0, 0.0, 0.0)})
    code, _ = run_entry_point("loki-geant4-validate", [], monkeypatch, capsys)
    assert code != 0


def test_loki_pixel_json_and_verify_json_readme_examples(tmp_path, monkeypatch, capsys):
    """README examples with small options (2 pixels per straw), including the failure exit code."""
    monkeypatch.chdir(tmp_path)
    code, out = run_entry_point(
        "loki-pixel-json",
        ["-o", "test.json", "--pixels-per-straw", "2", "--banks", "1", "2"],
        monkeypatch,
        capsys,
    )
    assert code == 0 and "2 banks, 1568 pixels" in out
    code, out = run_entry_point(
        "loki-pixel-json",
        ["-o", "base.json", "--pixels-per-straw", "2", "--pixel-reference", "pixel_base"],
        monkeypatch,
        capsys,
    )
    assert code == 0 and "9 banks, 12544 pixels" in out
    assert (
        json.loads((tmp_path / "base.json").read_text())["info"]["pixel_reference"] == "pixel_base"
    )
    # (--ignore-pixel-shape is only meaningful for pixel centre offsets)
    for args in (["test.json"], ["base.json"], ["test.json", "--ignore-pixel-shape"]):
        code, out = run_entry_point("loki-verify-json", args, monkeypatch, capsys)
        assert code == 0, out
        assert "> tolerance" not in out
    # REFERENCE_BANKS_DATA deviates up to ~1 mm from Geant4 (bank 8), so a tight tolerance fails
    code, out = run_entry_point(
        "loki-verify-json", ["base.json", "--tolerance", "0.5"], monkeypatch, capsys
    )
    assert code == 1 and "bank 8" in out and "> tolerance!" in out


def test_verify_json_wrong_pixels_per_straw_is_an_error(tmp_path, capsys):
    filename = str(tmp_path / "p.json")
    lnj.main(["-o", filename, "--pixels-per-straw", "2", "--banks", "3"])
    with pytest.raises(SystemExit):
        vnj.main([filename, "--pixels-per-straw", "4"])
    assert "pixel ids do not match" in capsys.readouterr().err


def test_verify_json_requires_the_carriage_translation_for_bank_0(tmp_path):
    output = lnj.build_output(
        lnj.compute_instrument_pixels({0: REFERENCE_BANKS_DATA[0]}, 1), "test"
    )
    del output[lnj.CARRIAGE_ZERO_NAME]
    filename = tmp_path / "p.json"
    lnj.write_json(output, filename)
    with pytest.raises(ValueError, match="detector_carriage_zero"):
        vnj.read_pixel_positions(filename)


def test_verify_json_reads_a_full_instrument_file_structure(tmp_path):
    """README: 'works on the full instrument file too' (/entry/instrument/<groups>,
    bank 0 translation inside detector_carriage/transformations), without loki.json."""
    corners = {bank_id: g4.bank_corners(bank_id, 2) for bank_id in (0, 4)}
    pixels = lnj.compute_instrument_pixels(corners, 2)
    output = lnj.build_output(pixels, "test")
    carriage = lnj.group(
        "detector_carriage",
        [lnj.group("transformations", [output.pop(lnj.CARRIAGE_ZERO_NAME)], "NXtransformations")],
        "NXpositioner",
    )
    instrument = lnj.group(
        "instrument",
        [carriage, output["loki_detector_0"], output["loki_detector_4"]],
        "NXinstrument",
    )
    full = {"children": [lnj.group("entry", [instrument], "NXentry")]}
    filename = tmp_path / "full.json"
    filename.write_text(json.dumps(full))
    results = vnj.compare_with_geant4(vnj.read_pixel_positions(filename), 2)
    assert sorted(results) == [0, 4]
    assert max(deviation for deviation, _ in results.values()) < 1e-4


def test_python_m_module_invocation():
    """README: the tools run as `python3 -m loki_geometry.<module>` with PYTHONPATH=src."""
    result = run_module("nominal_corners", "3")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Arrays match within tolerance") == 2
    result = run_module("nexus_json", "--help")
    assert result.returncode == 0 and "--pixel-reference" in result.stdout


# --- other public functions without coverage in the original tests -------------------


def test_adjust_calibration_data_to_effective_tube_lengths(capsys):
    adjusted = survey.adjust_calibration_data_to_effective_tube_lengths(example_calibration_data)
    assert "stretched" in capsys.readouterr().out
    for original, bank in zip(example_calibration_data, adjusted):
        for before, after in zip(original["positions"], bank["positions"]):
            assert after["distance"] == pytest.approx(c.STRAW_LENGTHS[bank["bank_id"]])
            assert np.allclose(
                np.add(after["first_pixel_end"], after["last_pixel_end"]),
                np.add(before["first_pixel_end"], before["last_pixel_end"]),
            )
            # the input is not modified: still the pixel centres (half a pixel short)
            assert before["distance"] == pytest.approx(
                c.STRAW_LENGTHS[bank["bank_id"]] * (1 - 1 / c.DEFAULT_PIXELS_PER_STRAW), abs=1e-5
            )


def test_calibration_checks_pass_for_the_reference_data(capsys):
    calibration.Detectors(
        calibration.convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA), True
    )
    out = capsys.readouterr().out
    assert "All bank base vectors are orthogonal." in out
    assert "All bank tilt angles agree" in out
    assert "All bank sample-bank distances agree" in out
    assert "Warning" not in out


def test_nominal_corners_print_formatted_data_prints_the_corners(capsys):
    assert nc.Detectors().print_formatted_data() is None
    assert "'A': [" in capsys.readouterr().out
