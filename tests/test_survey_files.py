"""Tests of the survey file registry (survey_files.py) and of the entry of the
SAM-606 survey: the file lookup and checksum, the coordinate frame, the tube
label map, the bank 5 fix, and the use of an entry by survey.py (including
complete entries for new, synthetic survey files)."""

import contextlib
import copy
import dataclasses
import io
import pathlib
import random
import re
import shutil

import numpy as np
import pytest

from loki_geometry import calibration, survey
from loki_geometry import constants as c
from loki_geometry import survey_files as sf
from loki_geometry.reference_data import REFERENCE_BANKS_DATA
from loki_geometry.survey_files import SAM606


def _quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def _sam606_banks():
    """The SAM-606 points after the fixes, in the NeXus frame, grouped into tubes."""
    raw = survey.read_measured_points(sf.SAM606.path)
    for fix in sf.SAM606.fixes:
        raw = _quiet(fix, raw)
    points = {label: sf.SAM606.frame(p) for label, p in raw.items()}
    return _quiet(survey.group_points_into_tubes, points, sf.SAM606.label_map, c.VERTICAL_BANK_IDS)


@pytest.fixture
def register(monkeypatch):
    """Add survey file entries for the duration of a test."""

    def add(entry):
        monkeypatch.setitem(sf.SURVEY_FILES, entry.file_name, entry)
        return entry

    return add


def test_every_entry_has_its_packaged_file_with_the_right_content():
    for name, entry in sf.SURVEY_FILES.items():
        assert entry.file_name == name
        assert sf.survey_file_info(entry.path) is entry  # also checks the checksum
        tubes = list(entry.label_map.values())
        assert len(set(tubes)) == len(tubes)  # no two labels for the same tube


def test_default_survey_file_is_a_registered_packaged_file():
    """The default can be switched to a new survey (README); it must be a packaged
    file with an entry."""
    entry = sf.survey_file_info(sf.DEFAULT_SURVEY_FILE)
    assert sf.DEFAULT_SURVEY_FILE == entry.path
    assert survey.DEFAULT_SURVEY_FILE == sf.DEFAULT_SURVEY_FILE


def test_unknown_survey_file_is_rejected(tmp_path):
    path = tmp_path / "new_survey.txt"
    path.write_text("C1UP3I, 1, 2, 3\n")
    with pytest.raises(ValueError, match="unknown survey file 'new_survey.txt'"):
        sf.survey_file_info(path)
    with pytest.raises(ValueError, match="unknown survey file"):
        survey.get_measured_calibration_data(str(path), verbose=False)


def test_changed_content_under_a_known_name_is_rejected(tmp_path):
    lines = open(sf.SAM606.path).read().splitlines()
    lines[0] = lines[0].replace("3276.7", "3276.8")
    path = tmp_path / sf.SAM606_FILE_NAME
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="does not match"):
        sf.survey_file_info(path)


def test_checksum_ignores_line_endings_blank_lines_and_spaces(tmp_path):
    lines = [line.strip() for line in open(sf.SAM606.path) if line.strip()]
    path = tmp_path / sf.SAM606_FILE_NAME
    path.write_bytes(("\r\n\r\n".join(f"  {line}  " for line in lines) + "\r\n").encode())
    assert sf.content_checksum(path) == sf.SAM606.checksum
    assert sf.survey_file_info(path) is sf.SAM606


def test_lookup_is_by_basename_and_accepts_pathlib(tmp_path):
    """A copy of a known file in any directory, given as str or Path, finds the entry."""
    copy_path = tmp_path / "elsewhere" / sf.SAM606_FILE_NAME
    copy_path.parent.mkdir()
    shutil.copy(sf.SAM606.path, copy_path)
    assert sf.survey_file_info(copy_path) is sf.SAM606
    assert sf.survey_file_info(str(copy_path)) is sf.SAM606
    assert sf.survey_file_info(pathlib.Path(sf.SAM606.path)) is sf.SAM606


def test_survey_cli_rejects_an_unknown_file(tmp_path, capsys):
    path = tmp_path / "new_survey.txt"
    path.write_text("C1UP3I, 1, 2, 3\n")
    with pytest.raises(SystemExit) as error:
        survey.main(["--survey-file", str(path)])
    assert error.value.code == 2
    assert "unknown survey file" in capsys.readouterr().err


def test_calibrate_cli_takes_the_survey_file(tmp_path, capsys):

    assert calibration.main(["--survey-file", sf.SAM606.path]) == 0
    with pytest.raises(SystemExit):
        calibration.main(["--data", "example", "--survey-file", sf.SAM606.path])
    assert "only applies to --data measured" in capsys.readouterr().err


def test_tank_window_frame_is_a_proper_rotation_plus_shift():
    origin = sf.tank_window_frame(np.zeros(3))
    assert np.allclose(origin, (0, 0, sf.SAMPLE_TO_TANK_WINDOW_DISTANCE))
    assert sf.SAMPLE_TO_TANK_WINDOW_DISTANCE == 100.0  # provisional
    matrix = np.array(
        [sf.tank_window_frame(e) - origin for e in np.eye(3)]
    ).T  # columns: images of the survey X, Y, Z axes
    assert np.allclose(matrix, [[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    assert np.linalg.det(matrix) == pytest.approx(1.0)  # handedness preserved


def test_sam606_label_map_covers_two_front_tubes_of_banks_1_to_8():
    by_bank = {}
    for bank_id, tube_id in sf.SAM606.label_map.values():
        assert 0 <= tube_id < c.TUBES_PER_LAYER[bank_id]  # front layer
        by_bank.setdefault(bank_id, []).append(tube_id)
    assert sorted(by_bank) == list(range(1, 9))
    assert all(len(set(tubes)) == 2 for tubes in by_bank.values())


def test_sam606_tube_ids_agree_with_their_spacing():
    for tube_a, tube_b in _sam606_banks().values():
        mid_a = (tube_a["end_A"] + tube_a["end_B"]) / 2
        mid_b = (tube_b["end_A"] + tube_b["end_B"]) / 2
        along = (tube_a["end_B"] - tube_a["end_A"]) / np.linalg.norm(
            tube_a["end_B"] - tube_a["end_A"]
        )
        across = (mid_b - mid_a) - np.dot(mid_b - mid_a, along) * along
        steps = np.linalg.norm(across) / c.TUBE_GRID_SIDE
        assert steps == pytest.approx(
            tube_b["tube_id"] - tube_a["tube_id"], abs=survey.TUBE_ID_TOLERANCE
        )


def test_sam606_label_map_matches_the_nominal_tube_positions():
    """Absolute tube identification: the transverse position of each measured
    tube, in pitches from tube 0 of REFERENCE_BANKS_DATA, is its assigned id."""
    banks = _sam606_banks()
    assert sorted(banks) == list(range(1, 9))
    for bank_id, tubes in banks.items():
        ref = REFERENCE_BANKS_DATA[bank_id]
        top, bottom = np.array(ref["A"][0], float), np.array(ref["A"][2], float)
        axis = 0 if bank_id in c.VERTICAL_BANK_IDS else 1
        step = (bottom - top)[axis] / (c.TUBES_PER_LAYER[bank_id] - 1)
        for tube in tubes:
            mid = (tube["end_A"] + tube["end_B"]) / 2
            estimated = (mid[axis] - top[axis]) / step
            assert estimated == pytest.approx(tube["tube_id"], abs=survey.TUBE_ID_TOLERANCE)


def test_sam606_bank5_fix_rebuilds_a_parallel_tube_one_pitch_from_the_dropped_point():
    raw = survey.read_measured_points(sf.SAM606.path)
    dropped = raw["C1UP8II"].copy()
    fixed = _quiet(sf.sam606_fix_bank5_mislabelled_tube, copy.deepcopy(raw))
    assert np.allclose(fixed["C1UP8I"], raw["C1UP8I"])
    assert np.allclose(fixed["C1UP8II"] - fixed["C1UP8I"], raw["C1UP3II"] - raw["C1UP3I"])
    assert np.linalg.norm(dropped - fixed["C1UP8II"]) == pytest.approx(c.TUBE_GRID_SIDE, abs=1.0)
    # the dropped point (tube UP8) is lower (higher tube id) than the rebuilt UP9 end
    assert dropped[2] < fixed["C1UP8II"][2]
    assert sf.sam606_fix_bank5_mislabelled_tube in sf.SAM606.fixes


def test_sam606_bank5_fix_warns_if_the_assumption_fails(capsys):
    raw = survey.read_measured_points(sf.SAM606.path)
    raw["C1UP8II"] = raw["C1UP8II"] + np.array([0.0, 0.0, 20.0])
    sf.sam606_fix_bank5_mislabelled_tube(raw, verbose=False)
    assert "WARNING: bank 5 fix" in capsys.readouterr().out


def test_sam606_measured_excess_lengths_are_plausible():
    for tubes in _sam606_banks().values():
        for tube in tubes:
            bank_id = sf.SAM606.label_map[tube["label"]][0]
            excess = np.linalg.norm(tube["end_B"] - tube["end_A"]) - c.STRAW_LENGTHS[bank_id]
            assert survey.EXCESS_LENGTH_RANGE[0] <= excess <= survey.EXCESS_LENGTH_RANGE[1]


def test_sam606_bank_note_is_printed_with_the_checks(capsys):
    survey.get_measured_calibration_data(sf.SAM606.path, verbose=True)
    out = capsys.readouterr().out
    assert "bank 5: 0.000 deg (tube UP9 constructed parallel to UP3 (bank 5 fix))" in out
    assert sf.SAM606.description in out


def test_points_must_match_the_label_map():
    raw = _quiet(sf.sam606_fix_bank5_mislabelled_tube, survey.read_measured_points(SAM606.path))
    missing = dict(raw)
    del missing["C2UP3II"]
    with pytest.raises(ValueError, match=r"missing \['C2UP3II'\]"):
        survey.group_points_into_tubes(missing, sf.SAM606.label_map, c.VERTICAL_BANK_IDS)
    extra = dict(raw, C9XP1I=np.zeros(3))
    with pytest.raises(ValueError, match=r"not in the map \['C9XP1I'\]"):
        survey.group_points_into_tubes(extra, sf.SAM606.label_map, c.VERTICAL_BANK_IDS)


def test_without_the_bank5_fix_bank5_tubes_are_not_parallel(register, tmp_path):
    """What the docstring of test_without_the_bank5_fix_the_sam606_file_is_inconsistent
    claims but does not assert: without the fix the parallelism check warns for bank 5
    (1.3 deg), and only for bank 5."""
    path = tmp_path / "unfixed.txt"
    path.write_text(open(sf.SAM606.path).read())
    register(
        dataclasses.replace(
            sf.SAM606, file_name="unfixed.txt", checksum=sf.content_checksum(path), fixes=()
        )
    )
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        survey.get_measured_calibration_data(str(path), verbose=True)
    parallel = out.getvalue().split("Parallelism")[1].split("Perpendicular")[0]
    flagged = re.findall(r"bank (\d): [\d.]+ deg.*WARNING: not parallel", parallel)
    assert flagged == ["5"]


def test_I_II_labels_and_line_order_do_not_matter(register, tmp_path):
    """The A/B ends are chosen from the coordinates. Swap I and II of every tube
    except the bank 5 pair (whose I/II labels carry meaning for the fix) and
    shuffle the lines: the calibration data must not change."""
    lines = [line.strip() for line in open(sf.SAM606.path) if line.strip()]
    swapped = []
    for line in lines:
        label, rest = line.split(",", 1)
        if not label.startswith(("C1UP8", "C1UP3")):
            label = label[:-2] + "I" if label.endswith("II") else label[:-1] + "II"
        swapped.append(f"{label},{rest}")
    random.Random(0).shuffle(swapped)
    path = tmp_path / "swapped.txt"
    path.write_text("\n".join(swapped) + "\n")
    register(
        dataclasses.replace(sf.SAM606, file_name="swapped.txt", checksum=sf.content_checksum(path))
    )
    original = _quiet(survey.get_measured_calibration_data, SAM606.path, verbose=False)
    result = _quiet(survey.get_measured_calibration_data, str(path), verbose=False)
    assert result == original


def test_a_new_survey_file_with_its_own_entry(register, tmp_path):
    """The steps of the README section "A new survey", with a synthetic file: the
    SAM-606 points (bank 5 already fixed) in another frame (the NeXus axes in
    metres, origin 1100 mm downstream of the sample) and with other labels. With
    its own label map and frame it gives exactly the SAM-606 calibration data."""
    origin_from_sample = np.array([0.0, 0.0, 1100.0])  # [mm] origin of the new frame
    raw = _quiet(sf.sam606_fix_bank5_mislabelled_tube, survey.read_measured_points(SAM606.path))
    new_labels = {old: f"T{n}" for n, old in enumerate(sorted(sf.SAM606.label_map))}
    lines = []
    for old, new in new_labels.items():
        for end in ("I", "II"):
            x, y, z = (sf.tank_window_frame(raw[old + end]) - origin_from_sample) / 1000.0  # [m]
            lines.append(f"{new}{end}, {x:.7f}, {y:.7f}, {z:.7f}")
    path = tmp_path / "LOKI_TUBES_2030.txt"
    path.write_text("\n".join(lines) + "\n")

    def metre_frame(point):
        # everything the conversion needs is in the function: units and origin
        return np.asarray(point) * 1000.0 + origin_from_sample

    register(
        sf.SurveyFile(
            file_name="LOKI_TUBES_2030.txt",
            checksum=sf.content_checksum(path),
            description="synthetic test survey",
            frame=metre_frame,
            label_map={new_labels[old]: ids for old, ids in sf.SAM606.label_map.items()},
        )
    )
    new = _quiet(survey.get_measured_calibration_data, str(path), verbose=False)
    expected = _quiet(survey.get_measured_calibration_data, SAM606.path, verbose=False)
    assert [bank["bank_id"] for bank in new] == [bank["bank_id"] for bank in expected]
    for bank_new, bank_expected in zip(new, expected):
        assert bank_new["bank_id"] == bank_expected["bank_id"]
        for tube_new, tube_expected in zip(bank_new["positions"], bank_expected["positions"]):
            assert tube_new["tube_id"] == tube_expected["tube_id"]
            for end in ("first_pixel_end", "last_pixel_end"):
                assert np.allclose(tube_new[end], tube_expected[end], atol=1e-3)


def _write_sam606_in_nexus_frame(path):
    """The SAM-606 points (bank 5 already fixed), converted to NeXus coordinates
    relative to the sample, with the SAM-606 labels."""
    raw = _quiet(sf.sam606_fix_bank5_mislabelled_tube, survey.read_measured_points(SAM606.path))
    lines = []
    for label, point in raw.items():
        x, y, z = sf.tank_window_frame(point)
        lines.append(f"{label}, {x:.3f}, {y:.3f}, {z:.3f}")
    path.write_text("\n".join(lines) + "\n")


def test_a_minimal_entry_uses_the_coordinates_as_given(register, tmp_path):
    """Only file_name and label_map: no frame conversion (the values are NeXus
    coordinates relative to the sample), no checksum, no fixes, no notes."""
    path = tmp_path / "minimal.txt"
    _write_sam606_in_nexus_frame(path)
    entry = register(sf.SurveyFile("minimal.txt", sf.SAM606.label_map))
    assert (entry.frame, entry.checksum, entry.description, entry.fixes, entry.bank_notes) == (
        None,
        None,
        "",
        (),
        {},
    )
    result = _quiet(survey.get_measured_calibration_data, str(path), verbose=False)
    expected = _quiet(survey.get_measured_calibration_data, SAM606.path, verbose=False)
    assert [bank["bank_id"] for bank in result] == [bank["bank_id"] for bank in expected]
    for bank_result, bank_expected in zip(result, expected):
        for tube_result, tube_expected in zip(bank_result["positions"], bank_expected["positions"]):
            for end in ("first_pixel_end", "last_pixel_end"):
                assert np.allclose(tube_result[end], tube_expected[end], atol=1e-3)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        survey.get_measured_calibration_data(str(path), verbose=True)
    assert "no frame conversion for this file" in out.getvalue()
    assert "has no checksum: content not checked" in out.getvalue()


def test_without_a_checksum_a_changed_file_is_used(register, tmp_path):
    path = tmp_path / "minimal.txt"
    _write_sam606_in_nexus_frame(path)
    register(sf.SurveyFile("minimal.txt", sf.SAM606.label_map))
    before = sf.survey_file_info(path)
    checksum_before = sf.content_checksum(path)
    path.write_text(path.read_text() + "C9XP1I, 0, 0, 0\n")  # changed content
    assert sf.content_checksum(path) != checksum_before
    assert sf.survey_file_info(path) is before
