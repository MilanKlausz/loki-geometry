"""
Verify the pixel positions of a LOKI NeXus-JSON file against the Geant4
geometry (via the Python port geant4.py; no dgcode needed).
Python counterpart of dgcode_projects/LOKI/G4GeoLoki/scripts/verifyJSON.

Works with the output of nexus_json.py and with a full instrument file
(e.g. loki.json). Pixel centre = bank translation + pixel offset + centre of
the pixel shape cylinder (the midpoint of its two face centres), so both pixel
reference conventions (pixel_centre / pixel_base) are handled, and an
inconsistent pixel shape shows up as a deviation. The bank translation is the
transformation the group depends on (for bank 0 the detector_carriage_zero
translation, with the carriage motion taken as zero).
--ignore-pixel-shape uses bank translation + pixel offset only.
Large files are streamed if the ijson package is installed.

Usage:
  loki-verify-json loki_detector_pixels.json [--tolerance 0.001]
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


def translation_mm(translation_node):
    """Position [mm] given by a translation dataset (magnitude [m] * unit vector)."""
    return np.asarray(_attribute(translation_node, "vector"), dtype=float) * _config(translation_node)["values"] * M_TO_MM


def _bank_translation_node(group_node, carriage_zero):
    depends_on = _config(_child(group_node, "depends_on"))["values"]
    if depends_on.endswith("/detector_carriage/value"):
        return carriage_zero
    return _child(_child(group_node, "transformations"), depends_on.rsplit("/", 1)[1])


def pixel_shape_centre_mm(group_node):
    """Centre of the pixel cylinder relative to the pixel offset point [mm]:
    the midpoint of the centres of its two faces (vertices 0 and 2)."""
    vertices = np.asarray(_config(_child(_child(group_node, "pixel_shape"), "vertices"))["values"], dtype=float)
    return (vertices[0] + vertices[2]) / 2 * M_TO_MM


def bank_pixels_from_group(group_node, carriage_zero=None, use_pixel_shape=True):
    """(pixel ids, absolute pixel centre positions [mm]) of a loki_detector_<id> group."""
    ids = np.asarray(_config(_child(group_node, "detector_number"))["values"])
    offsets = np.column_stack([_config(_child(group_node, f"{axis}_pixel_offset"))["values"] for axis in "xyz"]) * M_TO_MM
    positions = translation_mm(_bank_translation_node(group_node, carriage_zero)) + offsets
    if use_pixel_shape:
        positions = positions + pixel_shape_centre_mm(group_node)
    return ids, positions


def _iter_nodes_ijson(filename):
    """Yield the nodes of interest (bank groups, carriage zero) of a file, streaming."""
    import ijson
    wanted = {bank_name(b) for b in BANK_IDS} | {CARRIAGE_ZERO_NAME, "detector_carriage"}
    with open(filename, "rb") as file:
        # our own output: top level keys
        parser = ijson.kvitems(file, "", use_float=True)
        found = False
        for key, value in parser:
            if key in wanted:
                found = True
                yield key, value
        if found:
            return
    with open(filename, "rb") as file:
        # full instrument file: /entry/instrument children
        for node in ijson.items(file, "children.item.children.item.children.item", use_float=True):
            if node.get("name") in wanted:
                yield node["name"], node


def _iter_nodes_json(filename):
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
    """{bank_id: (pixel ids, pixel centre positions [mm])} of all LOKI banks in a file."""
    try:
        import ijson  # noqa: F401
        nodes = _iter_nodes_ijson(filename)
    except ImportError:
        nodes = _iter_nodes_json(filename)
    banks, pending, carriage_zero = {}, {}, None
    for name, node in nodes:
        if name == CARRIAGE_ZERO_NAME:
            carriage_zero = node
        elif name == "detector_carriage":
            carriage_zero = _child(_child(node, "transformations"), CARRIAGE_ZERO_NAME)
        else:
            pending[int(name.rsplit("_", 1)[1])] = node
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare LOKI pixel positions in a NeXus-JSON file with Geant4.")
    parser.add_argument("filename")
    parser.add_argument("--pixels-per-straw", type=int, default=None,
                        help="default: deduced from the number of pixels in the banks")
    parser.add_argument("--tolerance", type=float, default=1.0, help="[mm] largest accepted deviation")
    parser.add_argument("--ignore-pixel-shape", action="store_true",
                        help="pixel position = translation + offset (do not use the pixel shape centre)")
    args = parser.parse_args(argv)

    banks = read_pixel_positions(args.filename, use_pixel_shape=not args.ignore_pixel_shape)
    pixels_per_straw = args.pixels_per_straw
    if pixels_per_straw is None:
        bank_id, (ids, _) = next(iter(banks.items()))
        pixels_per_straw = len(ids) // (TUBES_PER_BANK[bank_id] * STRAWS_PER_TUBE)
    results = compare_with_geant4(banks, pixels_per_straw)
    print(f"{args.filename}: {pixels_per_straw} pixels per straw, deviation from Geant4 [mm]:")
    for bank_id, (deviation, pixel) in results.items():
        flag = "" if deviation <= args.tolerance else "  > tolerance!"
        print(f"  bank {bank_id}: largest {deviation:.4f} (pixel {pixel}){flag}")
    return 0 if all(d <= args.tolerance for d, _ in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
