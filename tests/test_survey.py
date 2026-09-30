"""Tests of the generic survey chain (survey.py): reading, grouping into tubes,
checks, trimming of the end electronics, command line."""

import contextlib
import copy
import dataclasses
import io

import numpy as np
import pytest

from loki_geometry import constants as c
from loki_geometry import survey
from loki_geometry import survey_files as sf
from loki_geometry.geant4_example_data import example_calibration_data
from loki_geometry.survey_files import SAM606


def _quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


@pytest.fixture
def register(monkeypatch):
    def add(entry):
        monkeypatch.setitem(sf.SURVEY_FILES, entry.file_name, entry)
        return entry

    return add


def _fixed_sam606_points():
    return _quiet(sf.sam606_fix_bank5_mislabelled_tube, survey.read_measured_points(sf.SAM606.path))


def _write(path, points):
    path.write_text(
        "".join(
            f"{k}, {float(x)!r}, {float(y)!r}, {float(z)!r}\n" for k, (x, y, z) in points.items()
        )
    )
    return path


def _nexus_points():
    """SAM-606 (bank 5 fixed) in the NeXus frame, full precision."""
    return {k: sf.tank_window_frame(p) for k, p in _fixed_sam606_points().items()}


def test_survey_file_covers_banks_1_to_8():
    data = survey.get_measured_calibration_data(SAM606.path, verbose=False)
    assert [bank["bank_id"] for bank in data] == list(range(1, 9))
    for bank in data:
        for tube in bank["positions"]:
            assert tube["distance"] == pytest.approx(c.STRAW_LENGTHS[bank["bank_id"]])


def test_measured_first_pixel_ends_are_at_plus_x_or_minus_y():
    data = _quiet(survey.get_measured_calibration_data, SAM606.path, verbose=False)
    for bank in data:
        for tube in bank["positions"]:
            first, last = np.array(tube["first_pixel_end"]), np.array(tube["last_pixel_end"])
            if bank["bank_id"] in c.VERTICAL_BANK_IDS:
                assert first[1] < last[1]
            else:
                assert first[0] > last[0]


def test_find_effective_tube_ends_shortens_and_stretches_symmetrically():
    first, last = np.array([10.0, 0, 0]), np.array([-10.0, 0, 0])
    for length in (5.0, 30.0):
        a, b, input_length, excess = survey.find_effective_tube_ends(first, last, length)
        assert np.linalg.norm(b - a) == pytest.approx(length)
        assert np.allclose((a + b) / 2, 0.0)
        assert input_length == pytest.approx(20.0)
        assert excess == pytest.approx(20.0 - length)


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


def test_read_measured_points_skips_blank_lines_and_rejects_bad_lines(tmp_path):
    good = tmp_path / "a.txt"
    good.write_text("\nT1I, 1, 2, 3\n\n  T1II ,4,5,6  \n")
    points = survey.read_measured_points(good)
    assert sorted(points) == ["T1I", "T1II"]
    assert np.array_equal(points["T1II"], [4.0, 5.0, 6.0])

    bad = tmp_path / "b.txt"
    bad.write_text("T1I, 1, 2\n")  # 3 fields
    with pytest.raises(ValueError, match="Unexpected line format"):
        survey.read_measured_points(bad)

    duplicate = tmp_path / "c.txt"
    duplicate.write_text("T1I, 1, 2, 3\nT1I, 1, 2, 3\n")
    with pytest.raises(ValueError, match="Duplicate label"):
        survey.read_measured_points(duplicate)

    not_a_number = tmp_path / "d.txt"
    not_a_number.write_text("T1I, 1, 2, x\n")
    with pytest.raises(ValueError):
        survey.read_measured_points(not_a_number)


def test_a_utf8_bom_is_ignored(register, tmp_path):
    path = tmp_path / "bom.txt"
    _write(path, _nexus_points())
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    register(sf.SurveyFile("bom.txt", sf.SAM606.label_map))
    assert len(_quiet(survey.get_measured_calibration_data, str(path), verbose=False)) == 8


def test_a_bank_needs_exactly_two_tubes():
    """README 'A new survey': every bank needs exactly 2 front layer tubes."""
    points = {
        f"T{n}{end}": np.array([x, 100.0 * n, 3000.0])
        for n in range(3)
        for end, x in (("I", 500.0), ("II", -500.0))
    }
    three = {"T0": (1, 0), "T1": (1, 5), "T2": (1, 9)}
    with pytest.raises(ValueError, match="expected 2 measured tubes, got 3"):
        survey.group_points_into_tubes(points, three, c.VERTICAL_BANK_IDS, verbose=False)
    one = {"T0": (1, 0)}
    only_t0 = {k: v for k, v in points.items() if k.startswith("T0")}
    with pytest.raises(ValueError, match="got 1"):
        survey.group_points_into_tubes(only_t0, one, c.VERTICAL_BANK_IDS, verbose=False)


def test_note_when_the_II_point_is_the_first_pixel_end(capsys):
    points = {"T0I": np.array([-500.0, 0, 3000]), "T0II": np.array([500.0, 0, 3000])}
    points.update({"T1I": np.array([500.0, 50, 3000]), "T1II": np.array([-500.0, 50, 3000])})
    banks = survey.group_points_into_tubes(
        points, {"T0": (1, 3), "T1": (1, 1)}, c.VERTICAL_BANK_IDS, verbose=True
    )
    assert "for T0 the II point is the A (low pixel id) end" in capsys.readouterr().out
    assert [t["tube_id"] for t in banks[1]] == [1, 3]  # sorted by tube id
    assert banks[1][1]["end_A"][0] == 500.0  # +x end chosen from the coordinates


@pytest.mark.parametrize(
    "change",
    [
        {"C2UP3": (1, 0)},  # two labels for the same tube
        {"C2UP3": (1, 16)},  # tube id outside bank 1's front layer (0..15)
        {"C2UP3": (9, 3), "C2UP16": (9, 0)},  # no bank 9
    ],
    ids=["duplicate-tube", "tube-id-range", "bank-id"],
)
def test_invalid_label_maps_are_rejected_with_a_value_error(register, tmp_path, change):
    path = _write(tmp_path / "badmap.txt", _nexus_points())
    register(sf.SurveyFile("badmap.txt", {**sf.SAM606.label_map, **change}))
    with pytest.raises(ValueError):
        _quiet(survey.get_measured_calibration_data, str(path), verbose=True)


def test_a_tube_with_identical_ends_is_rejected(register, tmp_path):
    points = _nexus_points()
    points["C2UP3II"] = points["C2UP3I"].copy()  # e.g. a duplicated line in the file
    path = _write(tmp_path / "zero.txt", points)
    register(sf.SurveyFile("zero.txt", sf.SAM606.label_map))
    with pytest.raises(ValueError):
        _quiet(survey.get_measured_calibration_data, str(path), verbose=False)


def test_a_frame_may_return_a_list(register, tmp_path):
    path = _write(tmp_path / "listframe.txt", _nexus_points())
    register(sf.SurveyFile("listframe.txt", sf.SAM606.label_map, frame=lambda p: list(p)))
    data = _quiet(survey.get_measured_calibration_data, str(path), verbose=True)
    assert len(data) == 8


def test_repeated_calls_are_identical_and_leave_the_entry_untouched():
    """The fixes mutate the points dict they get; that must be a fresh dict per
    call (no state shared between calls, verbose or not)."""
    label_map = copy.deepcopy(sf.SAM606.label_map)
    runs = [
        _quiet(survey.get_measured_calibration_data, sf.SAM606.path, verbose=v)
        for v in (False, True, False)
    ]
    assert runs[0] == runs[1] == runs[2]
    assert sf.SAM606.label_map == label_map
    assert sf.content_checksum(sf.SAM606.path) == sf.SAM606.checksum


def test_fixes_run_in_order_in_the_file_frame_before_the_frame_conversion(register, tmp_path):
    points = _fixed_sam606_points()
    path = _write(tmp_path / "order.txt", points)
    seen = []

    def first(p, verbose):
        seen.append(("first", p["C2UP3I"].copy()))
        p["C2UP3I"] = p["C2UP3I"] + np.array([0.0, 0.0, 1.0])  # 1 mm along the survey Z
        return p

    def second(p, verbose):
        seen.append(("second", p["C2UP3I"].copy()))
        return p

    register(
        dataclasses.replace(
            sf.SAM606, file_name="order.txt", checksum=None, fixes=(first, second), bank_notes={}
        )
    )
    shifted = _quiet(survey.get_measured_calibration_data, str(path), verbose=False)
    plain = _quiet(survey.get_measured_calibration_data, sf.SAM606.path, verbose=False)
    assert [name for name, _ in seen] == ["first", "second"]
    assert np.allclose(seen[0][1], points["C2UP3I"])  # file frame, not NeXus
    assert np.allclose(seen[1][1], points["C2UP3I"] + [0, 0, 1])  # sees the first fix
    # survey Z -> NeXus y: tube C2UP3 (bank 1, tube 13) midpoint moved by 0.5 mm in y
    mid = lambda t: (np.array(t["first_pixel_end"]) + np.array(t["last_pixel_end"])) / 2  # noqa: E731
    tube = lambda d: next(t for t in d[0]["positions"] if t["tube_id"] == 13)  # noqa: E731
    assert np.allclose(mid(tube(shifted)) - mid(tube(plain)), [0, 0.5, 0], atol=1e-9)


def test_without_frame_the_tube_midpoints_are_the_file_midpoints_exactly(register, tmp_path):
    """frame=None: the values are used as given (only the symmetric trimming,
    which keeps the midpoints)."""
    points = _nexus_points()
    path = _write(tmp_path / "asis.txt", points)
    register(sf.SurveyFile("asis.txt", sf.SAM606.label_map))
    data = _quiet(survey.get_measured_calibration_data, str(path), verbose=False)
    assert [b["bank_id"] for b in data] == list(range(1, 9))
    by_id = {ids: label for label, ids in sf.SAM606.label_map.items()}
    for bank in data:
        for tube in bank["positions"]:
            label = by_id[(bank["bank_id"], tube["tube_id"])]
            expected = (points[label + "I"] + points[label + "II"]) / 2
            got = (np.array(tube["first_pixel_end"]) + np.array(tube["last_pixel_end"])) / 2
            assert np.allclose(got, expected, atol=1e-9)


def test_survey_is_quiet_without_verbose(capsys):
    survey.get_measured_calibration_data(SAM606.path, verbose=False)
    assert capsys.readouterr().out == ""


def test_survey_cli_reports_a_missing_file_cleanly(tmp_path):
    missing = tmp_path / sf.SAM606_FILE_NAME  # known name, file does not exist
    with pytest.raises(SystemExit):
        survey.main(["--survey-file", str(missing)])


def test_without_verbose_only_the_failing_checks_are_printed(register, tmp_path, capsys):
    """The checks always run: a wrongly identified tube is reported (WARNING) even
    without verbose, while correct data prints nothing."""
    survey.get_measured_calibration_data(SAM606.path, verbose=False)
    assert capsys.readouterr().out == ""
    wrong = dict(SAM606.label_map, C2DP16=(3, 14))  # really tube 15
    path = _write(tmp_path / "wrong_ids.txt", _nexus_points())
    register(sf.SurveyFile("wrong_ids.txt", wrong))
    survey.get_measured_calibration_data(str(path), verbose=False)
    out = capsys.readouterr().out
    assert "bank 3 C2DP16 : estimated  15.02, assigned 14  WARNING: mismatch!" in out
    assert "bank 1" not in out  # only the failing lines
    assert "Perpendicular distance of the two tubes" in out  # with the title of the check


def test_a_packaged_survey_file_can_be_given_by_name(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)  # no such file in the current directory
    by_name = survey.get_measured_calibration_data(SAM606.file_name, verbose=False)
    assert by_name == survey.get_measured_calibration_data(SAM606.path, verbose=False)
    with pytest.raises(OSError):  # a path is not looked up in the package data
        survey.get_measured_calibration_data(str(tmp_path / SAM606.file_name), verbose=False)
    assert survey.main(["--survey-file", SAM606.file_name]) == 0


def test_a_fix_needing_a_missing_point_gives_a_clear_error(register, tmp_path):
    points = _fixed_sam606_points()
    del points["C1UP3II"]  # the SAM-606 bank 5 fix needs it
    path = _write(tmp_path / "incomplete.txt", points)
    register(dataclasses.replace(SAM606, file_name="incomplete.txt", checksum=None))
    with pytest.raises(ValueError, match="sam606_fix_bank5_mislabelled_tube .*'C1UP3II'"):
        _quiet(survey.get_measured_calibration_data, str(path), verbose=False)


@pytest.mark.parametrize("result", [(1.0, 2.0), (1.0, 2.0, float("nan"))])
def test_a_frame_must_return_three_finite_numbers(register, tmp_path, result):
    path = _write(tmp_path / "frame.txt", _nexus_points())
    register(sf.SurveyFile("frame.txt", SAM606.label_map, frame=lambda point: result))
    with pytest.raises(ValueError, match="expected 3 finite numbers"):
        survey.get_measured_calibration_data(str(path), verbose=False)
