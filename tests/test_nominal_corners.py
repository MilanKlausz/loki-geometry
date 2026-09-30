"""Tests of the reconstruction of the intended corners from the nominal bank
placement (nominal_corners.py)."""

import contextlib
import copy
import io

import numpy as np
import pytest

from loki_geometry import constants as c
from loki_geometry import geant4 as g4
from loki_geometry import nominal_corners
from loki_geometry.reference_data import REFERENCE_BANKS_DATA


def _quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


@pytest.mark.parametrize("bank_id", c.BANK_IDS)
def test_nominal_corners_match_geant4(bank_id):
    """Much tighter than the 1 mm vs REFERENCE_BANKS_DATA: the same rounded
    Geant4 table values give the same corners (up to the 2-decimal rounding)."""
    bank = nominal_corners.Bank(bank_id)
    expected = g4.bank_corners(bank_id)
    assert np.allclose(bank.positionA, expected["A"], atol=0.006)
    assert np.allclose(bank.positionB, expected["B"], atol=0.006)


def test_nominal_corners_print_formatted_data_prints_plain_floats(capsys):
    nominal_corners.Detectors().print_formatted_data()
    assert "np.float64" not in capsys.readouterr().out


def test_nominal_corners_cli_honours_several_bank_ids(capsys):
    assert nominal_corners.cli(["1", "2"]) == 0
    assert capsys.readouterr().out.count("Comparing") == 4


def test_nominal_corners_cli_rejects_unknown_bank_ids(capsys):
    with pytest.raises(SystemExit) as error:
        nominal_corners.cli(["9"])
    assert error.value.code == 2


def test_nominal_corners_cli_fails_with_exit_code_1(monkeypatch):
    changed = copy.deepcopy(REFERENCE_BANKS_DATA)
    changed[4]["A"][0] = tuple(v + 2.0 for v in changed[4]["A"][0])
    monkeypatch.setattr(nominal_corners.Detectors, "_REFERENCE_BANKS_DATA", changed)
    assert _quiet(nominal_corners.cli, ["4"]) == 1
    assert _quiet(nominal_corners.cli, ["3"]) == 0
