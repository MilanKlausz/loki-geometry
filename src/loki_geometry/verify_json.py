"""
Verify the pixel positions of a LOKI NeXus-JSON file against the expected pixel
positions of a bank calibration (geometry_info.py):
  - a file written by loki-pixel-json records its calibration in "info": it is
    compared with that calibration (e.g. the surveyed geometry), which the tests
    show to agree with the Geant4 geometry using the same calibration;
  - another file, e.g. the full instrument file loki.json, is compared with the
    nominal Geant4 geometry (nominal-geant4-geometry, the Python port geant4.py),
    as scripts/verifyJSON of the LOKI Geant4 geometry package G4GeoLoki does
    (https://github.com/ess-dg/dgcode_projects/tree/main/LOKI/G4GeoLoki);
  - --calibration NAME compares with any calibration.

Works with the output of nexus_json.py and with a full instrument file
(e.g. loki.json): the loki_detector_<id> groups of /entry/instrument are read by
name. Pixel centre = bank position + pixel offset + centre of the pixel shape
cylinder (the midpoint of its two face centres), so both pixel reference
conventions (pixel_centre / pixel_base) are handled, and an inconsistent pixel
shape shows up as a deviation. The bank position is the sum of the translations
of the depends_on chain of the group, followed by path to the end ("."): for
bank 0 the detector carriage position (its "value": 0 if it has no value, e.g. a
stream in the instrument file) and detector_carriage_zero. A broken chain (a
missing node or depends_on, a loop), a transformation that is not a translation
along a unit vector, lengths without a length unit, and two groups with the same
path are errors. --ignore-pixel-shape uses bank position + pixel offset only.

Assumptions:
  - bank 0 (rear bank) is compared with its position in the calibration
    (its front face at carriage zero: nominal Geant4 5 m, CAD 4999.8 mm, a survey
    calibration its surveyed position, e.g. 4985.65 mm for SAM-606),
    i.e. detector_carriage_zero must correspond to that (as in loki.json);
  - vertices 0 and 2 of the pixel shape are the centres of the two cylinder
    faces, in the units of the pixel offsets (m).
Differences from verifyJSON: the default tolerance is 1.0 mm (verifyJSON:
0.1 mm), and the pixel shape centre is used unless --ignore-pixel-shape is
given (verifyJSON uses translation + offset only).

Usage:
  loki-verify-json FILE [--calibration NAME] [--tolerance 1.0] [--pixels-per-straw N]
                        [--ignore-pixel-shape]
"""

import argparse
import json
import sys

import numpy as np

from . import geant4 as g4
from .constants import BANK_IDS, STRAWS_PER_TUBE, TUBES_PER_BANK
from .nexus_json import INSTRUMENT_PATH, bank_name

M_TO_MM = 1e3


def _config(node):
    return node.get("config", {})


def _child(group_node, name):
    for child in group_node.get("children", []):
        if child.get("name") == name or _config(child).get("name") == name:
            return child
    return None


def _attribute(node, name):
    for attr in node.get("attributes", []):
        if attr.get("name") == name:
            return attr.get("values")
    return None


def _required_child(group_node, name, context):
    """The child `name` of a group; ValueError if the group or the child is missing."""
    child = _child(group_node, name) if group_node is not None else None
    if child is None:
        raise ValueError(f"{context}: no '{name}'")
    return child


def _values(dataset_node, context):
    if "values" not in _config(dataset_node):
        raise ValueError(f"{context}: no values")
    return _config(dataset_node)["values"]


LENGTH_UNITS_IN_MM = {"m": 1e3, "cm": 10.0, "mm": 1.0, "um": 1e-3, "nm": 1e-6}
VECTOR_NORM_TOLERANCE = 1e-6


def _translation_mm(node, magnitude, units, context):
    """magnitude [units] along the vector of a translation node, in mm; ValueError if the node
    is not a translation, has no (unit) vector or the units are not a length."""
    transformation_type = _attribute(node, "transformation_type")
    if transformation_type != "translation":
        raise ValueError(
            f"{context}: transformation_type {transformation_type!r}: only translations are "
            "supported"
        )
    vector = _attribute(node, "vector")
    vector = np.asarray(vector if vector is not None else [], dtype=float)
    if vector.shape != (3,) or abs(np.linalg.norm(vector) - 1.0) > VECTOR_NORM_TOLERANCE:
        raise ValueError(f"{context}: the vector {vector.tolist()} is not a unit vector")
    if units not in LENGTH_UNITS_IN_MM:
        raise ValueError(f"{context}: units {units!r} are not a length unit")
    return vector * float(magnitude) * LENGTH_UNITS_IN_MM[units]


def translation_mm(translation_node):
    """Position [mm] given by a translation dataset (magnitude * unit vector, in its units)."""
    name = _config(translation_node).get("name", "translation")
    return _translation_mm(
        translation_node,
        _values(translation_node, name),
        _attribute(translation_node, "units"),
        name,
    )


def _node_name(node):
    return node.get("name") or _config(node).get("name")


def node_index(data):
    """{absolute path: node} of all named nodes of a NeXus-JSON tree; ValueError for two nodes
    with the same path."""
    index = {}

    def walk(node, path):
        for child in node.get("children", []):
            name = _node_name(child)
            if name is None:
                continue
            child_path = f"{path}/{name}"
            if child_path in index:
                raise ValueError(f"two nodes {child_path}")
            index[child_path] = child
            walk(child, child_path)

    walk(data, "")
    return index


def depends_on_translation_mm(path, index, context):
    """The position [mm] given by a depends_on chain: the sum of its translations, followed until
    "." (a dataset: its values; a group, like the NXlog of the detector carriage position: its
    "value" dataset, 0 if it has none, e.g. a stream in the instrument file). ValueError for a
    missing node, a missing depends_on, a loop, or a node that is not a translation in a length
    unit along a unit vector."""
    total = np.zeros(3)
    visited = set()
    while path != ".":
        if path in visited:
            raise ValueError(f"{context}: the depends_on chain loops at {path}")
        visited.add(path)
        node = index.get(path)
        if node is None:
            raise ValueError(f"{context}: depends_on {path} not found in the file")
        if node.get("module") == "dataset":
            total += _translation_mm(node, _values(node, path), _attribute(node, "units"), path)
        else:
            value = index.get(f"{path}/value")
            if value is not None:
                magnitude, units = _values(value, f"{path}/value"), _attribute(value, "units")
                magnitude = magnitude[0] if isinstance(magnitude, list) else magnitude
            else:  # a stream (no value in a static file): the zero position
                magnitude, units = 0.0, "m"
            total += _translation_mm(node, magnitude, units, path)
        depends_on = _attribute(node, "depends_on")
        if depends_on is None:
            raise ValueError(f"{context}: {path} has no depends_on")
        if depends_on != "." and not depends_on.startswith("/"):  # relative to its group
            depends_on = path.rsplit("/", 1)[0] + "/" + depends_on
        path = depends_on
    return total


def _group_name(group_node):
    return group_node.get("name", "bank group")


def pixel_shape_centre_mm(group_node):
    """Centre of the pixel cylinder relative to the pixel offset point [mm]:
    the midpoint of the centres of its two faces (vertices 0 and 2)."""
    name = _group_name(group_node)
    shape = _required_child(group_node, "pixel_shape", name)
    vertices = np.asarray(
        _values(_required_child(shape, "vertices", f"{name}/pixel_shape"), f"{name}/vertices"),
        dtype=float,
    )
    return (vertices[0] + vertices[2]) / 2 * M_TO_MM


def bank_pixels_from_group(group_node, translation, use_pixel_shape=True):
    """(pixel ids, absolute pixel centre positions [mm]) of a loki_detector_<id> group, whose
    depends_on gives the position translation [mm]. Raises ValueError if a part of the geometry
    is missing."""
    name = _group_name(group_node)

    def dataset_values(dataset):
        return _values(_required_child(group_node, dataset, name), f"{name}/{dataset}")

    ids = np.asarray(dataset_values("detector_number"))
    offsets = np.column_stack([dataset_values(f"{axis}_pixel_offset") for axis in "xyz"])
    offsets = offsets * LENGTH_UNITS_IN_MM[_offset_units(group_node, name)]
    positions = np.asarray(translation, dtype=float) + offsets
    if use_pixel_shape:
        positions = positions + pixel_shape_centre_mm(group_node)
    return ids, positions


def _offset_units(group_node, name):
    units = {
        _attribute(_required_child(group_node, f"{axis}_pixel_offset", name), "units")
        for axis in "xyz"
    }
    if len(units) != 1 or next(iter(units)) not in LENGTH_UNITS_IN_MM:
        raise ValueError(f"{name}: the units of the pixel offsets are {sorted(map(str, units))}")
    return units.pop()


def pixel_positions_from_tree(data, use_pixel_shape=True):
    """{bank_id: (pixel ids, pixel centre positions [mm])} of the loki_detector_<id> groups of
    /entry/instrument of a NeXus-JSON tree (the instrument file, or a loki-pixel-json output),
    each placed by its depends_on chain. Raises ValueError if there is none."""
    index = node_index(data)
    banks = {}
    for bank_id in BANK_IDS:
        path = f"{INSTRUMENT_PATH}/{bank_name(bank_id)}"
        group = index.get(path)
        if group is None:
            continue
        depends_on = _values(_required_child(group, "depends_on", path), f"{path}/depends_on")
        translation = depends_on_translation_mm(depends_on, index, path)
        banks[bank_id] = bank_pixels_from_group(group, translation, use_pixel_shape)
    if not banks:
        raise ValueError(f"no LOKI detector groups ({INSTRUMENT_PATH}/loki_detector_<id>) found")
    return banks


def read_pixel_positions(filename, use_pixel_shape=True):
    """{bank_id: (pixel ids, pixel centre positions [mm])} of all LOKI banks in a file (see
    pixel_positions_from_tree; the layout of older loki-pixel-json versions, the nodes at the
    top level, is converted to the instrument tree first). Raises ValueError if the file has
    no LOKI bank groups or their geometry is incomplete or inconsistent."""
    with open(filename) as file:
        data = json.load(file)
    if any(bank_name(b) in data for b in BANK_IDS):  # the nodes at the top level
        from .nexus_json import instrument_tree

        data = instrument_tree({"info": data.get("info", {}), **data})
    return pixel_positions_from_tree(data, use_pixel_shape)


def compare_with_geant4(banks, pixels_per_straw):
    """{bank_id: (largest deviation [mm], pixel id of it)} against the nominal Geant4 geometry."""
    return compare_with_calibration(banks, pixels_per_straw, None)


def compare_with_calibration(banks, pixels_per_straw, calibration):
    """{bank_id: (largest deviation [mm], pixel id of it)} against the pixel centres of a
    calibration (geometry_info.py; None: the nominal Geant4 geometry of geant4.py)."""
    geometry = None
    if calibration is not None:
        from .geometry_info import LokiGeometry

        geometry = LokiGeometry(calibration, pixels_per_straw)
    results = {}
    for bank_id, (ids, positions) in banks.items():
        if geometry is None:
            expected = g4.bank_pixel_centres(bank_id, pixels_per_straw)
        else:
            expected = geometry.bank_pixel_centres(bank_id)
        expected_ids = (
            np.arange(len(expected)) + g4.bank_pixel_offset(bank_id, pixels_per_straw) + 1
        )
        if len(ids) != len(expected_ids) or np.any(ids != expected_ids):
            raise ValueError(f"bank {bank_id}: pixel ids do not match the Geant4 numbering")
        deviation = np.linalg.norm(positions - expected, axis=1)
        worst = int(np.argmax(deviation))
        results[bank_id] = (float(deviation[worst]), int(ids[worst]))
    return results


def file_calibration(filename):
    """The calibration recorded in the "info" of a loki-pixel-json output (None if there is none)."""
    with open(filename) as file:
        info = json.load(file).get("info", {})
    return info.get("calibration") if isinstance(info, dict) else None


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, not {text}")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare the LOKI pixel positions of a NeXus-JSON file with the expected ones."
    )
    parser.add_argument("filename", help="NeXus-JSON file (loki-pixel-json output or loki.json)")
    parser.add_argument(
        "-c",
        "--calibration",
        default=None,
        help="compare with this bank calibration (default: the calibration recorded in a "
        "loki-pixel-json output, otherwise the nominal Geant4 geometry, nominal-geant4-geometry)",
    )
    parser.add_argument(
        "--pixels-per-straw",
        type=_positive_int,
        default=None,
        help="default: deduced from the number of pixels in the banks",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1.0, help="[mm] largest accepted deviation"
    )
    parser.add_argument(
        "--ignore-pixel-shape",
        action="store_true",
        help="pixel position = translation + offset (do not use the pixel shape centre)",
    )
    args = parser.parse_args(argv)

    try:
        banks = read_pixel_positions(args.filename, use_pixel_shape=not args.ignore_pixel_shape)
        pixels_per_straw = args.pixels_per_straw
        if pixels_per_straw is None:
            bank_id, (ids, _) = next(iter(banks.items()))
            pixels_per_straw = len(ids) // (TUBES_PER_BANK[bank_id] * STRAWS_PER_TUBE)
        calibration = args.calibration or file_calibration(args.filename)
        from .geometry_info import NOMINAL_GEANT4

        calibration = calibration or NOMINAL_GEANT4
        results = compare_with_calibration(
            banks, pixels_per_straw, None if calibration == NOMINAL_GEANT4 else calibration
        )
    except OSError as error:
        parser.error(f"cannot read {args.filename}: {error.strerror or error}")
    except ValueError as error:
        parser.error(f"{args.filename}: {error}")
    print(
        f"{args.filename}: {pixels_per_straw} pixels per straw, deviation from the "
        f"{calibration} pixel centres [mm]:"
    )
    for bank_id, (deviation, pixel) in results.items():
        flag = "" if deviation <= args.tolerance else "  > tolerance!"
        print(f"  bank {bank_id}: largest {deviation:.4f} (pixel {pixel}){flag}")
    return 0 if all(d <= args.tolerance for d, _ in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
