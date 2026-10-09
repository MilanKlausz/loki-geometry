"""Tests of the bank calibration file for the Geant4 geometry (geant4_calibration.py)
and of the loki-calibrate --geant4-calibration-file / --geant4-calibration-name options."""

import contextlib
import io
import tempfile

import numpy as np
import pytest

from loki_geometry import calibration
from loki_geometry import constants as c
from loki_geometry import geant4 as g4
from loki_geometry import geant4_calibration as gc
from loki_geometry.calibration import Detectors
from loki_geometry.geant4_example_data import example_calibration_data
from loki_geometry.reference_data import (
    REFERENCE_BANKS_DATA,
    convert_reference_data_to_calibration_data,
)
from loki_geometry.survey import get_measured_calibration_data
from loki_geometry.survey_files import SAM606

SAM606_NAME = "2026-September-SAM-606"


def _quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def _detectors(data):
    return _quiet(Detectors, data, True, intended=None, run_checks=False)


@pytest.fixture(scope="module")
def measured():
    return _detectors(_quiet(get_measured_calibration_data, SAM606.path, verbose=False))


@pytest.fixture(scope="module")
def example():
    return _detectors(example_calibration_data)


def _calibrate(args, capsys):
    code = calibration.main(args)
    return code, capsys.readouterr()


# --- names -------------------------------------------------------------------


@pytest.mark.parametrize("name", [SAM606_NAME, "a", "test_1.2"])
def test_valid_names(name):
    gc.check_name(name)
    assert gc.file_name_for(name) == f"bank_calibration_{name}.txt"


@pytest.mark.parametrize(
    "name", ["", "-a", "a b", "a/b", "a\tb", "ü", None, gc.NOMINAL_NAME, "abc\n", "abc\r", ".a"]
)
def test_invalid_names_are_rejected(name):
    with pytest.raises(ValueError):
        gc.check_name(name)


def test_the_calibration_names_of_the_survey_files():
    """Unique, valid, and not one of the two nominal names of the geometry tool."""
    from loki_geometry import geometry_info as gi
    from loki_geometry.survey_files import SURVEY_FILES

    names = [entry.calibration_name for entry in SURVEY_FILES.values() if entry.calibration_name]
    assert len(names) == len(set(names))
    for name in names:
        gc.check_name(name)
        assert name not in (gi.NOMINAL_CAD, gi.NOMINAL_GEANT4)


def test_the_sam606_entry_has_its_calibration_name():
    assert SAM606.calibration_name == SAM606_NAME
    gc.check_name(SAM606.calibration_name)


# --- writing and reading -------------------------------------------------------


def test_round_trip_is_exact(measured, tmp_path):
    """Full precision: the values read back are exactly the values of calibration.Bank."""
    path = gc.write_calibration_file(measured, tmp_path / "x.txt", "test", ["a comment", ""])
    name, banks = gc.read_calibration_file(path)
    assert name == "test"
    assert sorted(banks) == list(range(1, 9))
    for bank_id, values in banks.items():
        bank = measured.banks[bank_id]
        assert np.array_equal(values["F"], bank.front_face_centre)
        assert np.array_equal(values["u"], bank.vector_along_tubes)
        assert np.array_equal(values["w"], bank.vector_perpendicular_tubes)
        assert np.array_equal(values["n"], bank.vector_perpendicular_layers)
    text = path.read_text()
    assert "# a comment\n#\n" in text
    assert "loki_bank_calibration 1\nname test\n" in text


def test_a_directory_gets_the_file_of_the_name(measured, tmp_path):
    path = gc.write_calibration_file(measured, tmp_path, SAM606_NAME)
    assert path == tmp_path / f"bank_calibration_{SAM606_NAME}.txt"
    assert path.is_file()


def test_the_lines_of_the_file(measured):
    """One line per bank: 'bank <id>' and 12 numbers; comments before the data."""
    lines = gc.format_calibration(measured, "test").splitlines()
    data = [line for line in lines if not line.startswith("#")]
    assert data[:2] == ["loki_bank_calibration 1", "name test"]
    assert [line.split()[:2] for line in data[2:]] == [["bank", str(b)] for b in range(1, 9)]
    assert all(len(line.split()) == 14 for line in data[2:])
    assert "#   bank_calibration=test" in lines


def test_all_9_banks_are_written():
    reference = _detectors(convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA))
    _, banks = _parse(gc.format_calibration(reference, "reference"))
    assert sorted(banks) == list(range(9))
    assert np.array_equal(banks[0]["F"], reference.banks[0].front_face_centre)


def test_missing_banks_come_from_the_fallback_with_a_comment(measured):
    """SAM-606 has no bank 0: without a fallback it is not written (Geant4 keeps its nominal
    placement), with one it is written from the fallback, marked with a comment."""
    reference = _detectors(convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA))
    assert 0 not in measured.banks
    _, banks = _parse(gc.format_calibration(measured, "x"))
    assert sorted(banks) == list(range(1, 9))
    text = gc.format_calibration(measured, "x", fallback=reference, fallback_label="the reference")
    _, banks = _parse(text)
    assert sorted(banks) == list(range(9))
    assert "# Bank 0 is not in the calibration input: written from the reference." in text
    assert text.count("is not in the calibration input") == 1
    assert np.array_equal(banks[0]["F"], reference.banks[0].front_face_centre)
    assert np.array_equal(banks[0]["n"], reference.banks[0].vector_perpendicular_layers)
    for bank_id in range(1, 9):  # the measured banks are not replaced
        assert np.array_equal(banks[bank_id]["F"], measured.banks[bank_id].front_face_centre)


def test_no_bank_to_write_is_an_error():
    class NoBanks:
        banks = {}

    with pytest.raises(ValueError, match="no bank to write"):
        gc.format_calibration(NoBanks(), "x")


def test_sam606_front_face_centres_are_pinned(measured):
    """Hard-coded values (mm, NeXus frame, D = 83.11 mm) of two banks: the file content
    does not change unnoticed."""
    _, banks = _parse(gc.format_calibration(measured, SAM606_NAME))
    assert np.allclose(banks[2]["F"], [379.49805, -1.17917, 3360.79387], atol=1e-4)
    assert np.allclose(banks[7]["F"], [-91.96661, -622.78678, 1171.49536], atol=1e-4)
    assert np.allclose(banks[7]["u"], [0.99999569, 0.00194914, -0.00219337], atol=1e-7)


def _parse(text):
    """Read the text of a calibration file with read_calibration_file."""
    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/bank_calibration_x.txt"
        with open(path, "w") as file:
            file.write(text)
        return gc.read_calibration_file(path)


def _geant4_nominal_bank(bank_id):
    """F, u, w, n of the nominal Geant4 bank, from the pixel centres of the Python port
    (centre straw of the front layer tubes 0 and N-1)."""
    n = 2  # 2 pixels per straw: the mean of the two pixel centres is the straw midpoint
    tubes_per_layer = c.TUBES_PER_LAYER[bank_id]

    def centre(tube, pixel):
        return np.array(g4.pixel_centre(g4.pixel_id(bank_id, tube, c.CENTRE_STRAW, pixel, n), n))

    u = centre(0, 0) - centre(0, 1)
    u /= np.linalg.norm(u)
    first = (centre(0, 0) + centre(0, 1)) / 2
    last = (centre(tubes_per_layer - 1, 0) + centre(tubes_per_layer - 1, 1)) / 2
    w = first - last
    w -= np.dot(w, u) * u
    w /= np.linalg.norm(w)
    normal = np.cross(u, w)
    return (first + last) / 2 - normal * c.TUBE_OUTER_RADIUS, u, w, normal


@pytest.mark.parametrize("bank_id", range(9))
def test_the_geant4_example_data_give_the_nominal_geant4_banks(example, bank_id):
    """The file written from the Geant4 example data describes the nominal Geant4 bank
    (computed independently from the Python port of the Geant4 pixel positions), within the
    rounding of the example data (1e-6 mm)."""
    _, banks = _parse(gc.format_calibration(example, "geant4-example"))
    F, u, w, n = _geant4_nominal_bank(bank_id)
    assert np.allclose(banks[bank_id]["F"], F, rtol=0, atol=1e-5)
    for key, vector in zip("uwn", (u, w, n)):
        assert np.allclose(banks[bank_id][key], vector, rtol=0, atol=1e-7)


# --- reading errors ------------------------------------------------------------

GOOD_BANK = "bank 3 0 -500 2900 1 0 0 0 1 0 0 0 1"


@pytest.mark.parametrize(
    "lines,message",
    [
        (["name x", GOOD_BANK], "first line"),
        (["loki_bank_calibration 2", "name x"], "first line"),
        (["loki_bank_calibration 1", GOOD_BANK], "'name' line must come before"),
        (["loki_bank_calibration 1", "name x", "name y"], "repeated 'name'"),
        (["loki_bank_calibration 1", "name x", GOOD_BANK, GOOD_BANK], "repeated bank id"),
        (["loki_bank_calibration 1", "name x", "bank 9 0 0 5000 1 0 0 0 1 0 0 0 1"], "bank id 9"),
        (["loki_bank_calibration 1", "name x", "bank -1 0 0 5000 1 0 0 0 1 0 0 0 1"], "bank id -1"),
        (["loki_bank_calibration 1", "name x", "bank 3 0 -500 2900 1 0 0 0 1 0 0 0"], "12 numbers"),
        (["loki_bank_calibration 1", "name x", "bank 3 0 -500 nan 1 0 0 0 1 0 0 0 1"], "finite"),
        (["loki_bank_calibration 1", "name x", "bank 3 0 -500 2900 1 0 0 0 1 0.1 0 0 1"], "unit"),
        (["loki_bank_calibration 1", "name x", "bank 3 0 -500 2900 1 0 0 0 1 0 0 0 -1"], "u x w"),
        (["loki_bank_calibration 1", "name x", "bank 3 0 -500 -2900 1 0 0 0 1 0 0 0 1"], "away"),
        (["loki_bank_calibration 1", "name x", "banks 3"], "unknown line"),
        (["loki_bank_calibration 1"], "incomplete"),
        (["loki_bank_calibration 1", "name nominal-geant4-geometry"], "reserved"),
        (["loki_bank_calibration 1", "name a/b"], "invalid calibration name"),
        (
            ["loki_bank_calibration 1", "name x", "bank x 0 -500 2900 1 0 0 0 1 0 0 0 1"],
            "f.txt:5: invalid bank id",
        ),
        (
            ["loki_bank_calibration 1", "name x", "bank 3 0 -500 2900 1 0 0 0 1 0 0 a 1"],
            "f.txt:5: .* must be numbers",
        ),
    ],
)
def test_invalid_files_are_rejected(lines, message, tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("# comment\n\n" + "\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=message):
        gc.read_calibration_file(path)


# --- command line --------------------------------------------------------------


def test_cli_writes_the_sam606_file_with_its_default_name(tmp_path, capsys, measured):
    code, out = _calibrate(["--geant4-calibration-file", str(tmp_path)], capsys)
    assert code == 0
    path = tmp_path / f"bank_calibration_{SAM606_NAME}.txt"
    assert f"Geant4 bank calibration '{SAM606_NAME}' written to {path}" in out.out
    reference = _detectors(convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA))
    assert path.read_text() == gc.format_calibration(
        measured,
        SAM606_NAME,
        _cli_comments(path.read_text()),
        fallback=reference,
        fallback_label="REFERENCE_BANKS_DATA (the intended geometry)",
    )
    text = path.read_text()
    assert sorted(gc.read_calibration_file(path)[1]) == list(range(9))
    assert "# Bank 0 is not in the calibration input: written from REFERENCE_BANKS_DATA" in text
    assert SAM606.file_name in text and SAM606.checksum in text
    assert "loki-calibrate --data measured" in text
    assert "Bank 5: tube UP9 constructed parallel to UP3 (bank 5 fix)." in text  # the survey note


def _cli_comments(text):
    """The provenance comments of a file written by the command line."""
    lines = text.splitlines()
    start = lines.index(next(line for line in lines if line.startswith("# Written by")))
    end = lines.index(
        next(line for line in lines if line.startswith(("# Frame:", "# Bank 0 is not")))
    )
    return [line[2:] if line != "#" else "" for line in lines[start:end]]


def test_cli_with_another_file_name_tells_how_to_use_it(tmp_path, capsys):
    code, out = _calibrate(
        [
            "--data",
            "example",
            "--geant4-calibration-file",
            str(tmp_path / "g4.txt"),
            "--geant4-calibration-name",
            "geant4-example",
        ],
        capsys,
    )
    assert code == 0
    assert "bank_calibration=<path of the file>" in out.out
    name, banks = gc.read_calibration_file(tmp_path / "g4.txt")
    assert name == "geant4-example" and sorted(banks) == list(range(9))


@pytest.mark.parametrize(
    "args,message",
    [
        (
            ["--data", "reference", "--geant4-calibration-file", "x.txt"],
            "needs --geant4-calibration-name",
        ),
        (["--geant4-calibration-name", "a"], "needs --geant4-calibration-file"),
        (
            ["--geant4-calibration-file", "x.txt", "--geant4-calibration-name", "a b"],
            "invalid calibration name",
        ),
        (
            ["--geant4-calibration-file", "x.txt", "--geant4-calibration-name", gc.NOMINAL_NAME],
            "reserved",
        ),
        (["--geant4-calibration-file", "/nonexistent/dir/x.txt"], "cannot write"),
    ],
)
def test_cli_errors(args, message, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        calibration.main(args)
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "x.txt").exists()


def test_cli_needs_a_name_for_a_survey_without_calibration_name(tmp_path, capsys, monkeypatch):
    import shutil

    from loki_geometry import survey_files as sf

    shutil.copy(sf.SAM606.path, tmp_path / "noname.txt")
    entry = sf.SurveyFile(
        "noname.txt", sf.SAM606.label_map, frame=sf.SAM606.frame, fixes=sf.SAM606.fixes
    )
    monkeypatch.setitem(sf.SURVEY_FILES, entry.file_name, entry)
    with pytest.raises(SystemExit) as error:
        _quiet(
            calibration.main,
            [
                "--survey-file",
                str(tmp_path / "noname.txt"),
                "--geant4-calibration-file",
                str(tmp_path),
            ],
        )
    assert error.value.code == 2
    assert "needs --geant4-calibration-name" in capsys.readouterr().err
    assert not list(tmp_path.glob("bank_calibration_*"))


def test_cli_with_an_explicit_name_for_the_survey(tmp_path, capsys):
    code, out = _calibrate(
        ["--geant4-calibration-file", str(tmp_path), "--geant4-calibration-name", "my-x"], capsys
    )
    assert code == 0
    path = tmp_path / "bank_calibration_my-x.txt"
    assert gc.read_calibration_file(path)[0] == "my-x"
    assert "bank_calibration=<path of the file>" not in out.out  # the file name fits the name


def test_cli_reference_data_writes_all_banks_from_the_reference(tmp_path, capsys):
    """--data reference: all 9 banks from the input itself (no fallback), e.g. to give Geant4
    the intended (CAD) geometry."""
    code, _ = _calibrate(
        [
            "--data",
            "reference",
            "--geant4-calibration-file",
            str(tmp_path),
            "--geant4-calibration-name",
            "nominal-cad-geometry",
        ],
        capsys,
    )
    assert code == 0
    path = tmp_path / "bank_calibration_nominal-cad-geometry.txt"
    name, banks = gc.read_calibration_file(path)
    assert name == "nominal-cad-geometry" and sorted(banks) == list(range(9))
    assert "is not in the calibration input" not in path.read_text()
    reference = _detectors(convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA))
    for bank_id in range(9):
        assert np.array_equal(banks[bank_id]["F"], reference.banks[bank_id].front_face_centre)


def test_a_missing_directory_is_an_error_not_a_file(tmp_path, example, capsys):
    """A path ending with '/' is a directory: if it does not exist (e.g. a typo in
    G4GeoLoki/data/), nothing is written (API: ValueError; CLI: an error message)."""
    with pytest.raises(ValueError, match="does not exist"):
        gc.write_calibration_file(example, str(tmp_path / "typo") + "/", "my-test")
    with pytest.raises(SystemExit):
        calibration.main(["--geant4-calibration-file", str(tmp_path / "typo2") + "/"])
    assert "does not exist" in capsys.readouterr().err
    assert not (tmp_path / "typo").exists() and not (tmp_path / "typo2").exists()
