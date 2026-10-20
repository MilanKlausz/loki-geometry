"""
NeXus HDF5 files of LOKI (the format of the files written by the ESS file-writer):

1. replace the detector geometry of a file (measured or simulated) by that of a bank calibration
   (loki-nexus-geometry): the pixel offsets, the pixel shape and the bank positions, following the
   transformation chain of each bank in the file; everything else (events, logs) is kept;

2. write simulated detection events in the format of the measured files (loki-nexus-sim), so that the
   data reduction of the measured data (ess.loki, scipp) can be used: the structure is taken from a
   measured file (the template), without the experiment-specific parts (users, environment sensors,
   accelerator logs, ...), with the geometry of the calibration used in the simulation, and the simulated
   events (detector number, time of flight) distributed over the 14 Hz pulses of a chosen measuring time.

Geometry: the calibrations of geometry_info.py (the same as loki-pixel-json and loki-info). The number of
pixels per straw is that of the file (len(detector_number) / (tubes * 7)). The pixel offsets point to the
pixel centres, and the pixel shape (a cylinder of the straw radius) is centred on them (as in
nexus_json.py). The bank position (corner A[0]) is set through the transformation chain of the bank in the
file: the first transformation named detector_carriage_zero in the chain (bank 0, on the detector
carriage), otherwise the own transformation of the bank, is changed so that the sum of the chain (with the
detector carriage motion at zero) is the bank position of the calibration. Only translations are supported.

Simulated events: MCPL files of detection events (the pixel id, 0-based, in the 'ekin' field and the time of
flight [ms] in the 'time' field; older files: the pixel id in the user flags), or NumPy .npz files with the
arrays pixel_id (0-based) and time_ms. Several files are merged.

Requires h5py (and the mcpl Python package for MCPL input): pip install "loki-geometry[nexus]".
"""

import argparse
import datetime
import json
import os
import re
import shutil
import sys

import numpy as np

from . import __version__
from . import bank_pixels as lbp
from . import constants as c

PULSE_PERIOD_NS = 71428571  # the 14 Hz ESS pulses (as in the measured files)
INSTRUMENT = "/entry/instrument"
CARRIAGE_ZERO_NAME = "detector_carriage_zero"
CARRIAGE = f"{INSTRUMENT}/detector_carriage"
# children of /entry that are not written to simulation files (experiment and facility specific)
DROP_ENTRY = (
    r"^user_",
    r"^scripts$",
    r"^\.ESS_monitor$",
    r"^daq_publication$",
    r"^template_information$",
)
# groups of /entry/instrument that are not written to simulation files (sensors, environment, shutters,
# publications of the data acquisition)
DROP_INSTRUMENT = (
    r"^julabo_",
    r"^thermometer_",
    r"^vacuum$",
    r"^laser_mirror$",
    r"^heavy_shutter$",
    r"^experiment_shutter$",
    r"^efu_publication$",
)
# children of NXlog / NXevent_data / NXpositioner groups that are not written (state of the control system)
DROP_LOG_CHILDREN = (
    "alarm_message",
    "alarm_severity",
    "alarm_time",
    "connection_status",
    "connection_status_time",
    "cue_index",
    "cue_timestamp_zero",
    "idle_flag",
    "target_value",
)


def _h5py():
    try:
        import h5py
    except ImportError as error:  # pragma: no cover
        raise SystemExit('h5py is needed: pip install "loki-geometry[nexus]"') from error
    return h5py


def _text(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return _text(value.reshape(-1)[0]) if value.size else ""
    return str(value)


def _nx_class(node):
    return _text(node.attrs.get("NX_class", ""))


# ---------------------------------------------------------------------------
# Transformation chains
# ---------------------------------------------------------------------------


def _resolve(path, depends_on):
    if depends_on in (".", "") or depends_on.startswith("/"):
        return depends_on
    return path.rsplit("/", 1)[0] + "/" + depends_on


def transformation_chain(f, path):
    """The paths of the transformation chain that starts at path (a transformation), up to '.'."""
    chain, seen = [], set()
    while path not in (".", ""):
        if path in seen:
            raise ValueError(f"the depends_on chain loops at {path}")
        seen.add(path)
        if path not in f:
            raise ValueError(f"depends_on {path} is not in the file")
        chain.append(path)
        path = _resolve(path, _text(f[path].attrs.get("depends_on", ".")))
    return chain


def _unit_factor(units):
    factors = {"m": 1e3, "cm": 10.0, "mm": 1.0, "um": 1e-3}
    if units not in factors:
        raise ValueError(f"unsupported length unit {units!r}")
    return factors[units]


def transformation_mm(f, path, carriage_zero=True):
    """The translation [mm] of a transformation of the chain: a dataset (its value), or an NXlog (its first
    value; 0 if empty, and 0 for the detector carriage motion if carriage_zero)."""
    node = f[path]
    kind = _text(node.attrs.get("transformation_type", ""))
    if kind != "translation":
        raise ValueError(f"{path}: transformation_type {kind!r}, only translations are supported")
    vector = np.asarray(node.attrs.get("vector"), dtype=float).reshape(-1)
    norm = np.linalg.norm(vector)
    if vector.shape != (3,) or norm == 0:
        if node.attrs.get("vector") is None or norm == 0:
            return np.zeros(3)  # (a zero vector: no translation)
        raise ValueError(f"{path}: invalid vector {vector}")
    if hasattr(node, "keys"):  # an NXlog group
        if carriage_zero and path.startswith(CARRIAGE + "/"):
            return np.zeros(3)
        values = node["value"][()] if "value" in node else np.zeros(0)
        value = float(np.reshape(values, -1)[0]) if np.size(values) else 0.0
        units = _text(node["value"].attrs.get("units", "m")) if "value" in node else "m"
    else:
        value = float(np.reshape(node[()], -1)[0])
        units = _text(node.attrs.get("units", "m"))
    return vector / norm * value * _unit_factor(units)


def chain_position_mm(f, path, carriage_zero=True):
    """The position [mm] given by the transformation chain starting at path."""
    return sum(
        (transformation_mm(f, p, carriage_zero) for p in transformation_chain(f, path)), np.zeros(3)
    )


def bank_position_mm(f, bank_id, carriage_zero=True):
    """The position [mm] of a bank (its depends_on chain)."""
    group = f[f"{INSTRUMENT}/loki_detector_{bank_id}"]
    return chain_position_mm(f, _text(group["depends_on"][()]), carriage_zero)


# ---------------------------------------------------------------------------
# Geometry replacement
# ---------------------------------------------------------------------------


def pixels_per_straw_of(group, bank_id):
    n = group["detector_number"].shape[0]
    per_tube = c.TUBES_PER_BANK[bank_id] * c.STRAWS_PER_TUBE
    if n % per_tube:
        raise ValueError(f"bank {bank_id}: {n} pixels is not a multiple of {per_tube} straws")
    return n // per_tube


def _set_translation(f, path, translation_mm):
    """Set a translation dataset to translation_mm (its vector and value, in its units)."""
    node = f[path]
    if hasattr(node, "keys"):
        raise ValueError(f"{path}: an NXlog, can not be set")
    units = _text(node.attrs.get("units", "m"))
    length = float(np.linalg.norm(translation_mm))
    vector = (
        np.asarray(translation_mm, dtype=float) / length
        if length > 0
        else np.array([0.0, 0.0, 1.0])
    )
    old_vector = node.attrs.get("vector")
    node.attrs["vector"] = vector.astype(old_vector.dtype if old_vector is not None else float)
    node[()] = length / _unit_factor(units)


def replace_bank_geometry(f, bank_id, corners):
    """Replace the geometry of loki_detector_<bank_id> by that of its corners (REFERENCE_BANKS_DATA format);
    returns the largest deviation [mm] of the resulting bank position (chain) from the calibration."""
    from .nexus_json import pixel_shape_vertices

    group = f[f"{INSTRUMENT}/loki_detector_{bank_id}"]
    n = pixels_per_straw_of(group, bank_id)
    pixels = lbp.compute_bank_pixels(
        bank_id,
        corners,
        n,
        first_pixel_id=lbp.FIRST_PIXEL_ID,
        pixel_reference=lbp.PIXEL_CENTRE,
        bank_origin=lbp.FIRST_PIXEL_END,
    )
    group["detector_number"][...] = pixels.pixel_ids.astype(group["detector_number"].dtype)
    for i, axis in enumerate("xyz"):
        dataset = group[f"{axis}_pixel_offset"]
        dataset[...] = (
            pixels.offsets[:, i] / _unit_factor(_text(dataset.attrs.get("units", "m")))
        ).astype(dataset.dtype)
    shape = group["pixel_shape"]
    vertices = shape["vertices"]
    vertices[...] = (
        pixel_shape_vertices(pixels) / _unit_factor(_text(vertices.attrs.get("units", "m")))
    ).astype(vertices.dtype)
    shape["cylinders"][...] = np.array([[0, 1, 2]], dtype=shape["cylinders"].dtype)
    # the bank position: the carriage zero (bank 0) or the own translation of the bank
    chain = transformation_chain(f, _text(group["depends_on"][()]))
    adjustable = next((p for p in chain if p.rsplit("/", 1)[-1] == CARRIAGE_ZERO_NAME), chain[0])
    others = sum((transformation_mm(f, p) for p in chain if p != adjustable), np.zeros(3))
    _set_translation(f, adjustable, pixels.translation - others)
    return float(np.abs(bank_position_mm(f, bank_id) - pixels.translation).max())


def replace_geometry(f, calibration, banks=None, verbose=True):
    """Replace the geometry of the banks (default: all in the file) by that of a calibration (name, see
    geometry_info.available_calibrations). Returns {bank_id: largest position deviation [mm]}."""
    from .geometry_info import calibration_corners

    corners = calibration_corners(calibration)
    result = {}
    for bank_id in range(c.NUMBER_OF_BANKS) if banks is None else banks:
        if f"{INSTRUMENT}/loki_detector_{bank_id}" not in f:
            continue
        result[bank_id] = replace_bank_geometry(f, bank_id, corners[bank_id])
        if verbose:
            print(
                f"  bank {bank_id}: geometry of {calibration} "
                f"({pixels_per_straw_of(f[f'{INSTRUMENT}/loki_detector_{bank_id}'], bank_id)} pixels per straw)"
            )
    return result


def pixel_positions_mm(f, bank_id, carriage_zero=True):
    """The pixel centre positions [mm] of a bank in a file: bank position + offsets + the pixel shape centre."""
    group = f[f"{INSTRUMENT}/loki_detector_{bank_id}"]
    offsets = np.column_stack(
        [
            group[f"{a}_pixel_offset"][()]
            * _unit_factor(_text(group[f"{a}_pixel_offset"].attrs.get("units", "m")))
            for a in "xyz"
        ]
    )
    vertices = group["pixel_shape/vertices"]
    v = vertices[()] * _unit_factor(_text(vertices.attrs.get("units", "m")))
    return bank_position_mm(f, bank_id, carriage_zero) + offsets + (v[0] + v[2]) / 2


# ---------------------------------------------------------------------------
# Simulated events
# ---------------------------------------------------------------------------


def read_events(filenames):
    """(pixel ids (0-based), times of flight [ms], {metadata}) of detection event files (MCPL or .npz)."""
    ids, times, meta = [], [], {}
    for name in filenames:
        if name.endswith(".npz"):
            data = np.load(name)
            ids.append(np.asarray(data["pixel_id"], dtype=np.int64))
            times.append(np.asarray(data["time_ms"], dtype=float))
            continue
        try:
            import mcpl
        except ImportError as error:  # pragma: no cover
            raise SystemExit(
                "MCPL input needs the mcpl Python package: pip install mcpl"
            ) from error
        with mcpl.MCPLFile(name) as mf:
            for key in mf.blobs:
                try:
                    meta.setdefault(key, mf.blobs[key].decode())
                except UnicodeDecodeError:
                    pass
            old = mf.opt_userflags  # (older detection files: the pixel id in the user flags)
            for block in mf.particle_blocks:
                ids.append((block.userflags if old else block.ekin).astype(np.int64))
                times.append(np.asarray(block.time, dtype=float))
    if not ids:
        return np.zeros(0, np.int64), np.zeros(0), meta
    return np.concatenate(ids), np.concatenate(times), meta


def pulse_structure(n_events, duration_s, start_ns, rng):
    """Random pulse of each event: (pulse index of each event, event_time_zero of the pulses [ns])."""
    n_pulses = max(1, int(round(duration_s * 1e9 / PULSE_PERIOD_NS)))
    return rng.integers(0, n_pulses, n_events), start_ns + np.arange(
        n_pulses, dtype=np.uint64
    ) * np.uint64(PULSE_PERIOD_NS)


def write_event_data(group, detector_numbers, offsets_ns, pulses, time_zero_ns):
    """An NXevent_data group (as written by the file-writer: event_id, event_time_offset, event_time_zero,
    event_index) of events given with their pulse index."""
    pulses = np.asarray(pulses, dtype=np.int64)
    order = np.lexsort((np.asarray(offsets_ns), pulses))
    counts = np.bincount(pulses, minlength=len(time_zero_ns))
    index = (np.cumsum(counts) - counts).astype(np.int64)  # the first event of each pulse
    group.attrs["NX_class"] = "NXevent_data"
    group.create_dataset("event_id", data=np.asarray(detector_numbers)[order].astype(np.int32))
    eto = group.create_dataset(
        "event_time_offset", data=np.asarray(offsets_ns)[order].astype(np.int32)
    )
    eto.attrs["units"] = "ns"
    etz = group.create_dataset("event_time_zero", data=np.asarray(time_zero_ns, dtype=np.uint64))
    etz.attrs["units"] = "ns"
    etz.attrs["start"] = "1970-01-01T00:00:00Z"
    group.create_dataset("event_index", data=index)


# ---------------------------------------------------------------------------
# Copying the template
# ---------------------------------------------------------------------------


def _copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def copy_filtered(src_group, dst_group, path="/entry", drop_events=True):
    """Copy a group of the template without the experiment-specific parts (see the DROP lists) and without
    the event data (written separately)."""
    _copy_attrs(src_group, dst_group)
    for name, node in src_group.items():
        child = f"{path}/{name}"
        if path == "/entry" and any(re.search(p, name) for p in DROP_ENTRY):
            continue
        if path == INSTRUMENT and any(re.search(p, name) for p in DROP_INSTRUMENT):
            continue
        if name in DROP_LOG_CHILDREN:
            continue
        if path == f"{INSTRUMENT}/source" and hasattr(node, "keys") and _nx_class(node) == "NXlog":
            continue  # the accelerator logs
        if hasattr(node, "keys"):
            if drop_events and _nx_class(node) == "NXevent_data":
                continue
            copy_filtered(node, dst_group.create_group(name), child, drop_events)
        else:
            src_group.copy(node, dst_group, name=name)


def _set_text(group, name, value):
    if name in group:
        del group[name]
    group.create_dataset(name, data=value, dtype=_h5py().string_dtype())


def _set_log(group, value, units, time_ns):
    """Set an NXlog (a single value at time_ns) and its average / minimum / maximum."""
    for name in ("value", "time", "average_value", "minimum_value", "maximum_value"):
        if name in group:
            del group[name]
    v = group.create_dataset("value", data=np.array([value], dtype=float), maxshape=(None,))
    v.attrs["units"] = units
    t = group.create_dataset("time", data=np.array([time_ns], dtype=np.uint64), maxshape=(None,))
    t.attrs["units"] = "ns"
    t.attrs["start"] = "1970-01-01T00:00:00Z"
    for name in ("average_value", "minimum_value", "maximum_value"):
        group.create_dataset(name, data=float(value)).attrs["units"] = units


def _iso(ns):
    return (
        datetime.datetime.fromtimestamp(ns / 1e9, tz=datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )[:-3]
        + "Z"
    )


def write_simulation_file(
    template,
    output,
    event_files,
    calibration,
    duration_s,
    carriage_mm=0.0,
    start_time=None,
    title=None,
    seed=1,
    wrap=True,
    uniform_time_offset=False,
    verbose=True,
):
    """Write a NeXus file of simulated detection events with the structure of the template (see the module
    docstring). carriage_mm: the detector carriage position (from its zero position) of the simulation.
    uniform_time_offset: event_time_offset uniform in the pulse period instead of the simulated time of flight
    (sources without a pulse structure, e.g. the AmBe source tests: flat in time, as the measured data)."""
    h5py = _h5py()
    ids, tof_ms, meta = read_events(event_files)
    start = datetime.datetime.fromisoformat(
        (start_time or "2026-01-01T00:00:00").replace("Z", "+00:00")
    )
    if start.tzinfo is None:
        start = start.replace(tzinfo=datetime.timezone.utc)
    start_ns = int(start.timestamp() * 1e9)
    rng = np.random.default_rng(seed)
    with h5py.File(template, "r") as src, h5py.File(output, "w") as dst:
        _copy_attrs(src, dst)
        copy_filtered(src["/entry"], dst.create_group("entry"))
        entry = dst["/entry"]
        if verbose:
            print(
                f"Template {template}: structure copied (without the experiment-specific parts and events)"
            )
        deviations = replace_geometry(dst, calibration, verbose=verbose)
        pulses, time_zero = pulse_structure(len(ids), duration_s, start_ns, rng)
        end_ns = int(time_zero[-1]) + PULSE_PERIOD_NS
        offsets = np.rint(tof_ms * 1e6).astype(np.int64)
        if uniform_time_offset:
            offsets = rng.integers(0, PULSE_PERIOD_NS, len(ids))
        elif wrap:
            offsets %= PULSE_PERIOD_NS
        elif len(offsets) and offsets.max() >= PULSE_PERIOD_NS:
            raise ValueError(
                f"times of flight up to {offsets.max() / 1e6:.1f} ms exceed the pulse period"
            )
        detector_number = ids + lbp.FIRST_PIXEL_ID
        written = 0
        for bank_id in range(c.NUMBER_OF_BANKS):
            name = f"{INSTRUMENT}/loki_detector_{bank_id}"
            if name not in dst:
                continue
            numbers = dst[f"{name}/detector_number"][()]
            in_bank = (detector_number >= numbers.min()) & (detector_number <= numbers.max())
            write_event_data(
                dst[name].create_group("detector_events"),
                detector_number[in_bank],
                offsets[in_bank],
                pulses[in_bank],
                time_zero,
            )
            written += int(in_bank.sum())
            if verbose:
                print(f"  bank {bank_id}: {int(in_bank.sum())} events")
        if written != len(ids):
            raise ValueError(
                f"{len(ids) - written} events have detector numbers outside the banks of the file "
                "(another number of pixels per straw in the simulation?)"
            )
        for node in dst[
            INSTRUMENT
        ].values():  # monitors: no simulated monitor events (empty, as the template)
            if hasattr(node, "keys") and _nx_class(node) == "NXmonitor":
                write_event_data(
                    node.create_group("monitor_events"),
                    [],
                    [],
                    np.zeros(0, np.int64),
                    np.zeros(0, np.uint64),
                )
        # the detector carriage position
        for log in (f"{CARRIAGE}/transformations/motion", f"{CARRIAGE}/value"):
            if log in dst and hasattr(dst[log], "keys"):
                _set_log(dst[log], carriage_mm, "mm", start_ns)
        _set_text(entry, "title", title or "LOKI simulation")
        _set_text(entry, "start_time", _iso(start_ns))
        _set_text(entry, "end_time", _iso(end_ns))
        _set_text(entry, "experiment_identifier", "simulation")
        _set_text(
            entry,
            "experiment_description",
            "Simulated detection events (loki-geometry loki-nexus-sim)",
        )
        note = entry.create_group("simulation")
        note.attrs["NX_class"] = "NXnote"
        info = {
            "written_by": f"loki-geometry {__version__}",
            "template": os.path.basename(template),
            "event_files": [os.path.basename(e) for e in event_files],
            "events": int(len(ids)),
            "calibration": calibration,
            "duration_s": duration_s,
            "pulses": int(len(time_zero)),
            "carriage_mm": carriage_mm,
            "time_of_flight_wrapped": bool(wrap),
            "uniform_time_offset": bool(uniform_time_offset),
            "seed": seed,
            "simulation_metadata": meta,
        }
        _set_text(note, "description", json.dumps(info, indent=1))
    return {
        "events": int(len(ids)),
        "pulses": int(len(time_zero)),
        "geometry_deviation_mm": deviations,
    }


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _calibration_arg(parser, default):
    parser.add_argument(
        "-c",
        "--calibration",
        default=default,
        help=f"bank calibration (see loki-info --list-calibrations; default: {default})",
    )


def main_geometry(argv=None):
    """loki-nexus-geometry: a copy of a NeXus file with the detector geometry of a calibration."""
    from . import geometry_info as gi

    parser = argparse.ArgumentParser(
        description="Copy a LOKI NeXus file (measured or simulated) with the detector geometry of a bank "
        "calibration (pixel offsets, pixel shape, bank positions through the transformation chains). The "
        "events and everything else are kept."
    )
    parser.add_argument("input")
    parser.add_argument("output")
    _calibration_arg(parser, gi.DEFAULT_CALIBRATION)
    args = parser.parse_args(argv)
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        parser.error("the output must be another file")
    if args.calibration not in gi.available_calibrations():
        parser.error(f"unknown calibration {args.calibration!r}")
    h5py = _h5py()
    shutil.copyfile(args.input, args.output)
    with h5py.File(args.output, "r+") as f:
        deviations = replace_geometry(f, args.calibration)
        note = f["/entry"].require_group("geometry_replacement")
        note.attrs["NX_class"] = "NXnote"
        _set_text(
            note,
            "description",
            json.dumps(
                {
                    "written_by": f"loki-geometry {__version__}",
                    "calibration": args.calibration,
                    "input": os.path.basename(args.input),
                },
                indent=1,
            ),
        )
    worst = max(deviations.values()) if deviations else 0.0
    print(
        f"Written {args.output}: {len(deviations)} banks, largest bank position deviation {worst:.2e} mm"
    )
    return 0


def main_sim(argv=None):
    """loki-nexus-sim: simulated detection events as a NeXus file in the format of the measured files."""
    from . import geometry_info as gi

    parser = argparse.ArgumentParser(
        description="Write simulated LOKI detection events (MCPL detection event files, or .npz with pixel_id "
        "and time_ms) as a NeXus file with the structure of a measured file (the template), for the data "
        "reduction of the measured data."
    )
    parser.add_argument("events", nargs="+", help="detection event files (.mcpl, .mcpl.gz or .npz)")
    parser.add_argument("-t", "--template", required=True, help="a measured NeXus file of LOKI")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument(
        "-c",
        "--calibration",
        default=None,
        help="the bank calibration of the simulation (default: from the MCPL metadata "
        "bank_calibration, otherwise nominal-geant4-geometry)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        required=True,
        help="the measuring time [s] over which the events are distributed (14 Hz pulses)",
    )
    parser.add_argument(
        "--carriage-mm",
        type=float,
        default=None,
        help="the detector carriage position [mm] from its zero position (default: from the "
        "MCPL metadata rear_detector_distance_m: the distance of the rear bank front face minus its "
        "distance at the carriage zero in the calibration; otherwise 0)",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="ISO time of the first pulse (default 2026-01-01T00:00:00Z)",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--seed", type=int, default=1, help="random seed of the pulse assignment")
    parser.add_argument(
        "--no-wrap",
        action="store_true",
        help="error for times of flight beyond the pulse period (default: modulo the period)",
    )
    parser.add_argument(
        "--uniform-time-offset",
        action="store_true",
        help="event_time_offset uniform in the pulse period instead of the simulated time of flight (sources "
        "without pulses, e.g. AmBe source tests; as the measured data of such tests)",
    )
    args = parser.parse_args(argv)
    _, _, meta = read_events(args.events[:1])
    calibration = args.calibration or meta.get("bank_calibration") or gi.NOMINAL_GEANT4
    if calibration not in gi.available_calibrations():
        parser.error(
            f"unknown calibration {calibration!r} (a calibration file of a simulation must be a known "
            "calibration of loki-geometry)"
        )
    carriage = args.carriage_mm
    if carriage is None:
        rear = meta.get("rear_detector_distance_m")
        carriage = (
            1000.0 * float(rear) - gi.LokiGeometry(calibration, 1).bank(0).front_face_centre[2]
            if rear
            else 0.0
        )
    result = write_simulation_file(
        args.template,
        args.output,
        args.events,
        calibration,
        args.duration,
        carriage,
        args.start_time,
        args.title,
        args.seed,
        wrap=not args.no_wrap,
        uniform_time_offset=args.uniform_time_offset,
    )
    worst = (
        max(result["geometry_deviation_mm"].values()) if result["geometry_deviation_mm"] else 0.0
    )
    print(
        f"Written {args.output}: {result['events']} events in {result['pulses']} pulses, calibration "
        f"{calibration}, detector carriage {carriage:g} mm, largest bank position deviation {worst:.2e} mm"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main_sim())
