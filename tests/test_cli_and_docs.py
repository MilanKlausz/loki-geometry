"""Tests of the command-line entry points, the README claims and the packaging.

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

import pytest

import loki_geometry
from loki_geometry import constants as c
from loki_geometry import geant4 as g4
from loki_geometry import survey, survey_files
from loki_geometry.survey_files import SAM606

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


def test_console_scripts_point_to_callables():
    scripts = console_scripts()
    assert set(scripts) == {
        "loki-calibrate",
        "loki-survey",
        "loki-pixel-json",
        "loki-verify-json",
        "loki-geant4-validate",
        "loki-nominal-corners",
        "loki-info",
    }
    for target in scripts.values():
        module_name, function_name = target.split(":")
        assert callable(getattr(importlib.import_module(module_name), function_name))


def test_package_data_is_present_and_complete():
    """Every registered survey file is installed with the package (package data), and
    the SAM-606 file has the 32 points of the README: 2 tubes x 2 ends, banks 1-8."""
    package_dir = os.path.dirname(loki_geometry.__file__)
    for entry in survey_files.SURVEY_FILES.values():
        assert os.path.isfile(os.path.join(package_dir, "data", entry.file_name))
        assert entry.file_name.endswith(".txt")  # the package data pattern
    points = survey.read_measured_points(SAM606.path)
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


def test_readme_module_table_lists_every_module_and_script():
    text = read(README)
    modules = sorted(
        name[:-3]
        for name in os.listdir(os.path.dirname(loki_geometry.__file__))
        if name.endswith(".py") and name != "__init__.py"
    )
    for module in modules:
        assert f"| `{module}`" in text, module
    for script, target in console_scripts().items():
        module = target.split(":")[0].rsplit(".", 1)[1]
        assert f"| `{module}` (`{script}`)" in text, script


def test_readme_numbers_match_the_constants():
    text = " ".join(read(README).split())  # README lines are wrapped
    total = sum(c.TUBES_PER_BANK) * c.STRAWS_PER_TUBE * c.DEFAULT_PIXELS_PER_STRAW
    assert f"{total:,}" in text  # 1,605,632
    assert f"{c.TUBE_GRID_SIDE:.2f}" in text  # 28.40
    assert f"{c.FRONT_TO_BACK_NORMAL:.2f}" in text  # 78.78
    assert f"{c.FRONT_TO_BACK_IN_PLANE:.2f}" in text  # 18.84
    assert f"({c.TUBE_OUTER_RADIUS} mm)" in text
    assert f"({c.STRAW_RADIUS} mm)" in text
    assert f"{c.STRAW_STRAW_DISTANCE} mm to the side" in text
    offsets = [next(v for v in c.BANK_OFFSETS[b] if v) for b in (5, 6, 7, 8)]
    assert " / ".join(f"{v:+.0f}".replace("-", "−") for v in offsets) + " mm for banks 5–8" in text
    assert "Vertical tubes: banks " + ", ".join(map(str, c.VERTICAL_BANK_IDS)) in text
    assert "Packs mounted upside down: " + ", ".join(map(str, c.UPSIDE_DOWN_BANK_IDS)) in text


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
            # the option as a whole word of the help text (not e.g. "-o" inside "--output")
            pattern = r"(?<![\w-])" + re.escape(option) + r"(?![\w-])"
            assert re.search(pattern, help_texts[script]), f"{script} {option}"


@pytest.mark.parametrize(
    "args",
    [
        ["--data", "example", "--verbose"],
        ["--data", "measured", "--verbose", "--compare-to-intended", "--print"],
    ],
)
def test_loki_calibrate_readme_examples(args, monkeypatch, capsys):
    code, out = run_entry_point("loki-calibrate", args, monkeypatch, capsys)
    assert code == 0
    assert "=== Quality report ===" in out
    if "--print" in args:
        assert "'A': [" in out and "Corner positions compared to intended" in out


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


def test_python_m_module_invocation():
    """README: the tools run as `python3 -m loki_geometry.<module>` with PYTHONPATH=src."""
    result = run_module("nominal_corners", "3")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Arrays match within tolerance") == 2
    result = run_module("nexus_json", "--help")
    assert result.returncode == 0 and "--pixel-reference" in result.stdout


def _readme_text():
    with open(README, encoding="utf-8") as file:
        return " ".join(file.read().split())


def test_readme_self_test_threshold_is_the_cli_threshold():
    text = _readme_text()
    # calibration.main: tolerance = 0.1 for the --data reference self test (and the
    # actual largest difference, see test_calibration.py)
    assert "0.066 mm, the rounding of the CAD values; exit code 1 above 0.1 mm" in text
