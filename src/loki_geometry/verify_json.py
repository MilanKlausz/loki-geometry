"""
Verify the pixel positions of a LOKI NeXus-JSON file against the Geant4
geometry (via geant4.py, the Python port of the Geant4 pixel positions).
Python counterpart of scripts/verifyJSON of the LOKI Geant4 geometry package
G4GeoLoki (https://github.com/ess-dg/dgcode_projects/tree/main/LOKI/G4GeoLoki).

Works with the output of nexus_json.py and with a full instrument file
(e.g. loki.json). Pixel centre = bank translation + pixel offset + centre of
the pixel shape cylinder (the midpoint of its two face centres), so both pixel
reference conventions (pixel_centre / pixel_base) are handled, and an
inconsistent pixel shape shows up as a deviation. The bank translation is the
transformation the group depends on (for bank 0 the detector_carriage_zero
translation, with the carriage motion taken as zero).
--ignore-pixel-shape uses bank translation + pixel offset only.

Assumptions:
  - bank 0 (rear bank) depends on the detector carriage, and it is compared
    with the Geant4 geometry at the default rear bank distance of 5 m, i.e.
    detector_carriage_zero must correspond to 5 m (as in loki.json);
  - all lengths are in m (the units attributes are not read), the translations
    have no further transformations (their depends_on is "."), and vertices 0
    and 2 of the pixel shape are the centres of the two cylinder faces.
Differences from verifyJSON: the default tolerance is 1.0 mm (verifyJSON:
0.1 mm), and the pixel shape centre is used unless --ignore-pixel-shape is
given (verifyJSON uses translation + offset only).

Usage:
  loki-verify-json FILE [--tolerance 1.0] [--pixels-per-straw N] [--ignore-pixel-shape]
"""

import argparse
import json
import sys

import numpy as np

from . import geant4 as g4
from .constants import BANK_IDS, STRAWS_PER_TUBE, TUBES_PER_BANK
from .nexus_json import CARRIAGE_BANK_ID, CARRIAGE_ZERO_NAME, bank_name

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


def translation_mm(translation_node):
    """Position [mm] given by a translation dataset (magnitude [m] * unit vector)."""
    return (
        np.asarray(_attribute(translation_node, "vector"), dtype=float)
        * _config(translation_node)["values"]
        * M_TO_MM
    )


def _group_name(group_node):
    return group_node.get("name", "bank group")


def _bank_translation_node(group_node, carriage_zero):
    name = _group_name(group_node)
    depends_on = _values(_required_child(group_node, "depends_on", name), f"{name}/depends_on")
    if depends_on.endswith("/detector_carriage/value"):
        if carriage_zero is None:
            raise ValueError(
                f"{name} depends on the detector carriage, but no {CARRIAGE_ZERO_NAME} "
                "translation was found"
            )
        return carriage_zero
    transformations = _required_child(group_node, "transformations", name)
    return _required_child(transformations, depends_on.rsplit("/", 1)[1], f"{name}/transformations")


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


def bank_pixels_from_group(group_node, carriage_zero=None, use_pixel_shape=True):
    """(pixel ids, absolute pixel centre positions [mm]) of a loki_detector_<id> group.
    Raises ValueError if a part of the geometry is missing."""
    name = _group_name(group_node)

    def dataset_values(dataset):
        return _values(_required_child(group_node, dataset, name), f"{name}/{dataset}")

    ids = np.asarray(dataset_values("detector_number"))
    offsets = np.column_stack([dataset_values(f"{axis}_pixel_offset") for axis in "xyz"]) * M_TO_MM
    positions = translation_mm(_bank_translation_node(group_node, carriage_zero)) + offsets
    if use_pixel_shape:
        positions = positions + pixel_shape_centre_mm(group_node)
    return ids, positions


def _iter_nodes(filename):
    """Yield (name, node) of the nodes of interest (bank groups, carriage zero) of a
    file: the top level keys of our own output, or the /entry/instrument children
    of a full instrument file."""
    with open(filename) as file:
        data = json.load(file)
    wanted = {bank_name(b) for b in BANK_IDS} | {CARRIAGE_ZERO_NAME, "detector_carriage"}
    if any(key in data for key in wanted):
        yield from ((key, value) for key, value in data.items() if key in wanted)
        return
    for entry in data.get("children", []):
        for instrument in entry.get("children", []):
            for node in instrument.get("children", []):
                if node.get("name") in wanted:
                    yield node["name"], node


def read_pixel_positions(filename, use_pixel_shape=True):
    """{bank_id: (pixel ids, pixel centre positions [mm])} of all LOKI banks in a file.
    Raises ValueError if the file has no LOKI bank groups."""
    banks, pending, carriage_zero = {}, {}, None
    for name, node in _iter_nodes(filename):
        if name == CARRIAGE_ZERO_NAME:
            carriage_zero = node
        elif name == "detector_carriage":
            transformations = _child(node, "transformations")
            if transformations is not None:
                carriage_zero = _child(transformations, CARRIAGE_ZERO_NAME)
        else:
            pending[int(name.rsplit("_", 1)[1])] = node
    if not pending:
        raise ValueError(f"{filename}: no LOKI detector groups (loki_detector_<id>) found")
    for bank_id, node in sorted(pending.items()):
        if bank_id == CARRIAGE_BANK_ID and carriage_zero is None:
            raise ValueError(f"{CARRIAGE_ZERO_NAME} translation not found for {bank_name(bank_id)}")
        banks[bank_id] = bank_pixels_from_group(node, carriage_zero, use_pixel_shape)
    return banks


def compare_with_geant4(banks, pixels_per_straw):
    """{bank_id: (largest deviation [mm], pixel id of it)} against Geant4."""
    results = {}
    for bank_id, (ids, positions) in banks.items():
        geant4 = g4.bank_pixel_centres(bank_id, pixels_per_straw)
        expected_ids = np.arange(len(geant4)) + g4.bank_pixel_offset(bank_id, pixels_per_straw) + 1
        if len(ids) != len(expected_ids) or np.any(ids != expected_ids):
            raise ValueError(f"bank {bank_id}: pixel ids do not match the Geant4 numbering")
        deviation = np.linalg.norm(positions - geant4, axis=1)
        worst = int(np.argmax(deviation))
        results[bank_id] = (float(deviation[worst]), int(ids[worst]))
    return results


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, not {text}")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare LOKI pixel positions in a NeXus-JSON file with Geant4."
    )
    parser.add_argument("filename", help="NeXus-JSON file (loki-pixel-json output or loki.json)")
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
        results = compare_with_geant4(banks, pixels_per_straw)
    except OSError as error:
        parser.error(f"cannot read {args.filename}: {error.strerror or error}")
    except ValueError as error:
        parser.error(f"{args.filename}: {error}")
    print(f"{args.filename}: {pixels_per_straw} pixels per straw, deviation from Geant4 [mm]:")
    for bank_id, (deviation, pixel) in results.items():
        flag = "" if deviation <= args.tolerance else "  > tolerance!"
        print(f"  bank {bank_id}: largest {deviation:.4f} (pixel {pixel}){flag}")
    return 0 if all(d <= args.tolerance for d, _ in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
