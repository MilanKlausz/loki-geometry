"""Tests of the NeXus HDF5 tools (nexus_hdf5.py): the geometry replacement and the simulation files, with a
small synthetic template in the layout of the measured LOKI files (transformation chains through the window guard
and the detector carriage, NXlogs with control-system fields, experiment-specific groups)."""

import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from loki_geometry import bank_pixels as lbp  # noqa: E402
from loki_geometry import constants as c  # noqa: E402
from loki_geometry import nexus_hdf5 as nh  # noqa: E402
from loki_geometry.geometry_info import LokiGeometry, calibration_corners  # noqa: E402
from loki_geometry.nexus_json import pixel_shape_vertices  # noqa: E402

N = 4  # pixels per straw of the synthetic template
INST = "/entry/instrument"


def _translation(group, name, value, vector, depends_on, dtype=np.float32):
    d = group.create_dataset(name, data=dtype(value))
    d.attrs["transformation_type"] = "translation"
    d.attrs["units"] = "m"
    d.attrs["vector"] = np.asarray(vector, dtype=dtype)
    d.attrs["depends_on"] = depends_on
    return d


def _log(group, name, values, units, **attrs):
    g = group.create_group(name)
    g.attrs["NX_class"] = "NXlog"
    for k, v in attrs.items():
        g.attrs[k] = v
    g.create_dataset("value", data=np.asarray(values, dtype=float), maxshape=(None,)).attrs[
        "units"
    ] = units
    t = g.create_dataset("time", data=np.arange(len(values), dtype=np.uint64), maxshape=(None,))
    t.attrs["units"] = "ns"
    for name_ in ("alarm_message", "connection_status"):
        g.create_dataset(name_, data=np.zeros(1, dtype=np.int16))
    return g


def _events(group, name, ids, offsets, pulses, n_pulses):
    nh.write_event_data(
        group.create_group(name),
        np.asarray(ids),
        np.asarray(offsets),
        np.asarray(pulses),
        np.arange(n_pulses, dtype=np.uint64) * np.uint64(nh.PULSE_PERIOD_NS),
    )


def make_template(path, calibration="nominal-cad-geometry"):
    """A small file in the layout of the measured files, with the geometry of a calibration (N pixels per straw)."""
    corners = calibration_corners(calibration)
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        for name, value in (
            ("title", "measured run"),
            ("start_time", "2026-10-19T07:32:19.222Z"),
            ("experiment_identifier", "336349"),
        ):
            entry.create_dataset(name, data=value, dtype=h5py.string_dtype())
        entry.create_dataset("scripts", data=["start_filewriting()"], dtype=h5py.string_dtype())
        entry.create_group("user_SomeOne").attrs["NX_class"] = "NXuser"
        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = "NXsample"
        _translation(sample.create_group("transformations"), "trans_11", 0.0, [0, 0, 1], ".")
        sample.create_dataset(
            "depends_on", data="/entry/sample/transformations/trans_11", dtype=h5py.string_dtype()
        )
        inst = entry.create_group("instrument")
        inst.attrs["NX_class"] = "NXinstrument"
        inst.create_dataset("name", data="LOKI", dtype=h5py.string_dtype())
        wg = inst.create_group("window_guard").create_group("transformations")
        _translation(wg, "distance", 0.0, [0, 0, 1], f"{INST}/window_guard/transformations/height")
        _translation(
            wg, "height", 0.0, [0, 1, 0], f"{INST}/window_guard/transformations/x_translation"
        )
        _translation(wg, "x_translation", 0.0, [1, 0, 0], ".")
        carriage = inst.create_group("detector_carriage")
        carriage.attrs["NX_class"] = "NXpositioner"
        ct = carriage.create_group("transformations")
        zero = calibration_corners(calibration)[0]["A"][0]
        _translation(
            ct,
            nh.CARRIAGE_ZERO_NAME,
            np.linalg.norm(zero) / 1000,
            np.asarray(zero) / np.linalg.norm(zero),
            f"{INST}/window_guard/transformations/distance",
        )
        _log(
            ct,
            "motion",
            [],
            "mm",
            transformation_type="translation",
            vector=np.array([0.0, 0.0, 1.0]),
            depends_on=f"{INST}/detector_carriage/transformations/{nh.CARRIAGE_ZERO_NAME}",
        )
        _log(carriage, "value", [4999.96], "mm")
        _log(carriage, "idle_flag", [1], "")
        source = inst.create_group("source")
        source.attrs["NX_class"] = "NXsource"
        source.create_dataset("name", data="ESS", dtype=h5py.string_dtype())
        _log(source, "beam_current", [2.0], "mA")
        julabo = inst.create_group("julabo_1")
        julabo.attrs["NX_class"] = "NXsensor"
        for m in ("beam_monitor_m1", "beam_monitor_m3"):
            mon = inst.create_group(m)
            mon.attrs["NX_class"] = "NXmonitor"
            _translation(mon.create_group("transformations"), "trans", 1.0, [0, 0, -1], ".")
            mon.create_dataset(
                "depends_on", data=f"{INST}/{m}/transformations/trans", dtype=h5py.string_dtype()
            )
            _events(mon, "monitor_events", [], [], [], 0)
        for b in range(c.NUMBER_OF_BANKS):
            p = lbp.compute_bank_pixels(b, corners[b], N)
            g = inst.create_group(f"loki_detector_{b}")
            g.attrs["NX_class"] = "NXdetector"
            g.create_dataset("detector_number", data=p.pixel_ids.astype(np.int32))
            for i, axis in enumerate("xyz"):
                g.create_dataset(
                    f"{axis}_pixel_offset", data=(p.offsets[:, i] / 1000).astype(np.float32)
                ).attrs["units"] = "m"
            shape = g.create_group("pixel_shape")
            shape.attrs["NX_class"] = "NXcylindrical_geometry"
            shape.create_dataset(
                "vertices", data=(pixel_shape_vertices(p) / 1000).astype(np.float32)
            ).attrs["units"] = "m"
            shape.create_dataset("cylinders", data=np.array([[0, 1, 2]], dtype=np.int32))
            t = g.create_group("transformations")
            if (
                b == 0
            ):  # on the detector carriage (as in the measured files: a zero own translation)
                _translation(
                    t,
                    "translation",
                    0.0,
                    [0.1, 0.2, 0.3],
                    f"{INST}/detector_carriage/transformations/motion",
                )
            else:
                tr = p.translation
                _translation(
                    t,
                    "translation",
                    np.linalg.norm(tr) / 1000,
                    tr / np.linalg.norm(tr),
                    f"{INST}/window_guard/transformations/distance",
                )
            g.create_dataset(
                "depends_on",
                data=f"{INST}/loki_detector_{b}/transformations/translation",
                dtype=h5py.string_dtype(),
            )
            ids = p.pixel_ids[:3]
            _events(g, "detector_events", ids, [10, 20, 30], [0, 0, 1], 2)
    return path


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    return str(make_template(tmp_path_factory.mktemp("nexus") / "template.hdf"))


def test_the_template_positions_are_those_of_its_calibration(template):
    cad = LokiGeometry("nominal-cad-geometry", N)
    with h5py.File(template) as f:
        for b in range(c.NUMBER_OF_BANKS):
            assert np.abs(nh.pixel_positions_mm(f, b) - cad.bank_pixel_centres(b)).max() < 2e-3


def test_replace_geometry_gives_the_pixel_centres_of_the_calibration(template, tmp_path):
    path = tmp_path / "geo.hdf"
    path.write_bytes(open(template, "rb").read())
    geant4 = LokiGeometry("nominal-geant4-geometry", N)
    with h5py.File(path, "r+") as f:
        deviations = nh.replace_geometry(f, "nominal-geant4-geometry", verbose=False)
        assert max(deviations.values()) < 2e-3
        for b in range(c.NUMBER_OF_BANKS):
            assert np.abs(nh.pixel_positions_mm(f, b) - geant4.bank_pixel_centres(b)).max() < 2e-3
        # bank 0: through the detector carriage zero, its own (zero) translation is unchanged
        assert f[f"{INST}/loki_detector_0/transformations/translation"][()] == 0
        zero = nh.transformation_mm(
            f, f"{INST}/detector_carriage/transformations/{nh.CARRIAGE_ZERO_NAME}"
        )
        assert np.allclose(zero, geant4.corners[0]["A"][0], atol=2e-3)


def test_the_carriage_motion_moves_only_bank_0(template, tmp_path):
    path = tmp_path / "carriage.hdf"
    path.write_bytes(open(template, "rb").read())
    with h5py.File(path, "r+") as f:
        before = {b: nh.pixel_positions_mm(f, b, carriage_zero=False) for b in (0, 1)}
        nh._set_log(f[f"{INST}/detector_carriage/transformations/motion"], 5000.0, "mm", 0)
        assert np.allclose(
            nh.pixel_positions_mm(f, 0, carriage_zero=False) - before[0], [0, 0, 5000], atol=1e-3
        )
        assert np.allclose(nh.pixel_positions_mm(f, 1, carriage_zero=False), before[1])


def test_main_geometry_keeps_the_events(template, tmp_path, capsys):
    out = tmp_path / "out.hdf"
    assert nh.main_geometry([template, str(out), "-c", "nominal-geant4-geometry"]) == 0
    assert "largest bank position deviation" in capsys.readouterr().out
    with h5py.File(template) as a, h5py.File(out) as b:
        for k in range(c.NUMBER_OF_BANKS):
            ev = f"{INST}/loki_detector_{k}/detector_events/event_id"
            assert np.array_equal(a[ev][()], b[ev][()])
        assert (
            json.loads(nh._text(b["/entry/geometry_replacement/description"][()]))["calibration"]
            == "nominal-geant4-geometry"
        )
    with pytest.raises(SystemExit):
        nh.main_geometry([template, template])


def _sim_events(tmp_path, n=4):
    """Events in every bank (0-based pixel ids for N pixels per straw), times of flight up to 100 ms."""
    rng = np.random.default_rng(3)
    ids = np.concatenate(
        [
            lbp.bank_first_pixel_id(b, n, 0) + rng.integers(0, lbp.pixels_in_bank(b, n), 50)
            for b in range(c.NUMBER_OF_BANKS)
        ]
    )
    path = tmp_path / "events.npz"
    np.savez(path, pixel_id=ids, time_ms=rng.uniform(0, 100, len(ids)))
    return str(path), ids


def test_simulation_file(template, tmp_path):
    events, ids = _sim_events(tmp_path)
    out = str(tmp_path / "sim.hdf")
    result = nh.write_simulation_file(
        template,
        out,
        [events],
        "nominal-geant4-geometry",
        duration_s=10.0,
        carriage_mm=5000.0,
        title="test",
        verbose=False,
    )
    assert result["events"] == len(ids) and result["pulses"] == 140
    with h5py.File(out) as f:
        entry = f["/entry"]
        # experiment-specific parts are not written, the rest of the structure is
        assert "scripts" not in entry and not any(k.startswith("user_") for k in entry)
        assert "julabo_1" not in f[INST] and "window_guard" in f[INST]
        assert "beam_current" not in f[f"{INST}/source"] and "name" in f[f"{INST}/source"]
        assert "idle_flag" not in f[f"{INST}/detector_carriage"]
        assert "alarm_message" not in f[f"{INST}/detector_carriage/value"]
        assert (
            nh._text(entry["title"][()]) == "test"
            and nh._text(entry["experiment_identifier"][()]) == "simulation"
        )
        # the detector carriage at 5 m: bank 0 moved back by 5 m (geometry of the calibration at carriage zero)
        assert f[f"{INST}/detector_carriage/transformations/motion/value"][()].tolist() == [5000.0]
        geant4 = LokiGeometry("nominal-geant4-geometry", N)
        assert (
            np.abs(
                nh.pixel_positions_mm(f, 0, carriage_zero=False)
                - geant4.bank_pixel_centres(0)
                - [0, 0, 5000]
            ).max()
            < 2e-3
        )
        # all events, in their banks, distributed over the pulses, time offsets within the pulse period
        total = 0
        for b in range(c.NUMBER_OF_BANKS):
            ev = f[f"{INST}/loki_detector_{b}/detector_events"]
            numbers = f[f"{INST}/loki_detector_{b}/detector_number"][()]
            assert set(ev["event_id"][()]) <= set(numbers)
            index, n_ev = ev["event_index"][()], len(ev["event_id"])
            assert (
                len(index) == len(ev["event_time_zero"]) == 140
                and index[0] == 0
                and np.all(np.diff(index) >= 0)
            )
            assert index[-1] <= n_ev and np.all(ev["event_time_offset"][()] < nh.PULSE_PERIOD_NS)
            assert np.all(np.diff(ev["event_time_zero"][()].astype(np.int64)) == nh.PULSE_PERIOD_NS)
            total += n_ev
        assert total == len(ids)
        for m in ("beam_monitor_m1", "beam_monitor_m3"):
            assert len(f[f"{INST}/{m}/monitor_events/event_index"]) == 0
        info = json.loads(nh._text(entry["simulation/description"][()]))
        assert info["calibration"] == "nominal-geant4-geometry" and info["events"] == len(ids)


def test_simulation_events_must_fit_the_pixels_of_the_template(template, tmp_path):
    events, _ = _sim_events(
        tmp_path, n=8
    )  # another number of pixels per straw than the template (4)
    with pytest.raises(ValueError, match="outside the banks"):
        nh.write_simulation_file(
            template,
            str(tmp_path / "bad.hdf"),
            [events],
            "nominal-geant4-geometry",
            1.0,
            verbose=False,
        )


def test_no_wrap_rejects_long_times_of_flight(template, tmp_path):
    events, _ = _sim_events(tmp_path)
    with pytest.raises(ValueError, match="exceed the pulse period"):
        nh.write_simulation_file(
            template,
            str(tmp_path / "w.hdf"),
            [events],
            "nominal-geant4-geometry",
            1.0,
            wrap=False,
            verbose=False,
        )


def test_main_sim(template, tmp_path, capsys):
    events, ids = _sim_events(tmp_path)
    out = tmp_path / "cli.hdf"
    assert (
        nh.main_sim(
            [events, "-t", template, "-o", str(out), "-d", "2", "-c", "nominal-geant4-geometry"]
        )
        == 0
    )
    assert f"{len(ids)} events in 28 pulses" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        nh.main_sim([events, "-t", template, "-o", str(out), "-d", "2", "-c", "unknown"])


def test_uniform_time_offsets_for_sources_without_pulses(template, tmp_path):
    """--uniform-time-offset: event_time_offset flat over the pulse period (as in the measured source tests)."""
    rng = np.random.default_rng(5)
    ids = lbp.bank_first_pixel_id(8, N, 0) + rng.integers(0, lbp.pixels_in_bank(8, N), 20000)
    events = tmp_path / "flat.npz"
    np.savez(events, pixel_id=ids, time_ms=np.full(len(ids), 1.5))  # all at 1.5 ms
    out = str(tmp_path / "flat.hdf")
    nh.write_simulation_file(
        template,
        out,
        [str(events)],
        "nominal-geant4-geometry",
        5.0,
        uniform_time_offset=True,
        verbose=False,
    )
    with h5py.File(out) as f:
        eto = f[f"{INST}/loki_detector_8/detector_events/event_time_offset"][()]
    counts = np.histogram(eto, bins=10, range=(0, nh.PULSE_PERIOD_NS))[0]
    assert eto.max() < nh.PULSE_PERIOD_NS and counts.min() > 0.85 * counts.mean()
