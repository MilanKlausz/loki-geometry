"""
Export the LOKI detector pixel geometry as NeXus-JSON nodes, ready to be copied
into the instrument's JSON file (the file-writer configuration, e.g. loki.json):
the main output of the package. The pixel positions are those of a bank
calibration (--calibration, see geometry_info.py): by default the calibration of
the default survey (the ACTUAL, surveyed geometry, e.g. 2026-September-SAM-606;
banks that were not surveyed, like the rear bank of SAM-606, keep the intended
CAD geometry), or nominal-cad-geometry / nominal-geant4-geometry.

For each bank ("loki_detector_<id>") the output contains the geometry parts of
its NXdetector group, in the same structure as in loki.json:
  - detector_number   pixel ids (1-based, ICD numbering)
  - pixel_shape       NXcylindrical_geometry of one pixel, a cylinder of the straw
                      radius and the pixel length, placed consistently with the
                      pixel reference point (see below)
  - x/y/z_pixel_offset  offsets of the pixel reference points from the bank position [m]
  - transformations   one translation from the sample to the bank position [m]
                      (banks 1-8; bank 0 has none, see below)
  - depends_on
Bank position (--bank-origin), i.e. the bank translation, to which the pixel
offsets are relative. The absolute pixel positions are the same for both:
  first_pixel_end (default): the corner A[0] of the bank, i.e. the FIRST PIXEL
      END of the centre straw of tube 0 (front layer, top tube, or leftmost tube
      in vertical banks) on the tube axis, as in the current loki.json. It is
      not a pixel position: the first pixel centre is half a pixel further along
      the straw;
  centre_of_mass: the mean of all pixel centres of the bank (the centre of the
      tube grid, halfway along the straws; independent of the pixels per straw).

Pixel reference point (--pixel-reference):
  pixel_centre (default): the offsets point to the pixel centres, and the pixel
      shape is centred on them (faces at -/+ half a pixel along the straw);
  pixel_base: the offsets point to the centre of the pixel's first face (towards
      the first pixel end), and the pixel shape starts there (faces at 0 and +1
      pixel along the straw).
NOTE: the current loki.json mixes the two: its offsets point to the pixel
centres, but its pixel shape starts at the offset point (pixel_base shape), so
the pixel volumes are shifted by half a pixel along the straws.

Bank 0 (the rear bank) is on the moving detector carriage: its group depends on
/entry/instrument/detector_carriage/value, and its translation is given as the
"detector_carriage_zero" dataset of /entry/instrument/detector_carriage/transformations.
Other children of the groups in the instrument file (e.g. detector_events)
are not geometry and are not written.

Output file layout: the instrument tree of the instrument file, with only the
geometry nodes of the detector banks in it (the structure of loki.json, so that
the file can be read like the instrument file, e.g. by the verification scripts
and NeXus-JSON geometry viewers):
  {"info": {...},
   "children": [entry (NXentry) -> instrument (NXinstrument) -> [
       loki_detector_0, ..., loki_detector_8   (the written banks),
       detector_carriage (NXpositioner, if bank 0 is written):
           value (NXlog: the carriage position, 0 here; a translation along z that
                  depends on detector_carriage_zero),
           transformations/detector_carriage_zero (the translation of bank 0),
           depends_on]]}
To update the instrument file, copy the loki_detector_<id> groups and the
detector_carriage_zero dataset into it (see the README).

--visualize: a file for a look at the geometry with NeXus-JSON geometry viewers
(e.g. https://github.com/chribe/jsondisplay): 4 pixels per straw unless
--pixels-per-straw is given (the pixel offsets are drawn as points). Bank 0 is placed
by its own transformations/detector_carriage_zero (the same positions) instead of the
detector carriage, which is not written: viewers that follow only the transformations
inside a group (as jsondisplay) place it correctly. Not for the instrument file.

Usage:
  loki-pixel-json -o loki_detector_pixels.json [--calibration NAME] [--pixels-per-straw 256]
                  [--pixel-reference pixel_centre|pixel_base]
                  [--bank-origin first_pixel_end|centre_of_mass] [--banks 1 2 ...] [--decimals 8]
  loki-pixel-json --visualize -o view.json --banks 1 2 3 4 5 6 7 8

The output records the calibration and the source of the corners of each bank in
"info" (calibration, bank_corner_sources).
"""

import argparse
import json
import sys

import numpy as np

from . import __version__
from .bank_pixels import (
    BANK_ORIGINS,
    CENTRE_OF_MASS,
    FIRST_PIXEL_END,
    PIXEL_CENTRE,
    PIXEL_REFERENCES,
    compute_instrument_pixels,
    pixel_reference_fraction,
)
from .constants import BANK_IDS, DEFAULT_PIXELS_PER_STRAW, STRAW_RADIUS

MM_TO_M = 1e-3
LENGTH_UNIT = "m"
DEFAULT_DECIMALS = 8  # rounding of the coordinates in [m] (1e-8 m = 10 nm)
# coarsest accepted rounding (1e-4 m = 0.1 mm per value; a pixel position adds several rounded values: the
# translation, the offsets and the pixel shape, so it can be off by somewhat more, ~0.11 mm)
MIN_DECIMALS = 4
INSTRUMENT_PATH = "/entry/instrument"
CARRIAGE_BANK_ID = 0  # the bank on the moving detector carriage
CARRIAGE_ZERO_NAME = "detector_carriage_zero"
CARRIAGE_PATH = f"{INSTRUMENT_PATH}/detector_carriage"
CARRIAGE_VALUE_PATH = f"{CARRIAGE_PATH}/value"
CARRIAGE_ZERO_PATH = f"{CARRIAGE_PATH}/transformations/{CARRIAGE_ZERO_NAME}"
# what the bank translation points to, for each bank origin (info of the output)
BANK_ORIGIN_DESCRIPTIONS = {
    FIRST_PIXEL_END: "from the sample to corner A[0]: the first pixel end of the centre straw of "
    "tube 0 (front layer, top/left) on the tube axis",
    CENTRE_OF_MASS: "from the sample to the centre of mass of the bank: the mean of all its "
    "pixel centres",
}


# ---------------------------------------------------------------------------
# Generic NeXus-JSON node builders
# ---------------------------------------------------------------------------


def attribute(name, values, dtype=None):
    node = {"name": name, "values": values}
    if dtype is not None:
        node["dtype"] = dtype
    return node


def units_attribute():
    return attribute("units", LENGTH_UNIT, "string")


def nx_class_attribute(nx_class):
    return attribute("NX_class", nx_class)


def dataset(name, values, dtype, attributes=None):
    node = {"module": "dataset", "config": {"name": name, "values": values, "dtype": dtype}}
    if attributes:
        node["attributes"] = attributes
    return node


def group(name, children, nx_class):
    return {
        "name": name,
        "type": "group",
        "children": children,
        "attributes": [nx_class_attribute(nx_class)],
    }


def _rounded_list(array, decimals):
    return np.round(np.asarray(array, dtype=float), decimals).tolist()


# ---------------------------------------------------------------------------
# LOKI specific nodes
# ---------------------------------------------------------------------------


def bank_name(bank_id):
    return f"loki_detector_{bank_id}"


def translation_name(bank_id):
    """Name of the translation of a bank (as in loki.json: trans_<id + 1>)."""
    return CARRIAGE_ZERO_NAME if bank_id == CARRIAGE_BANK_ID else f"trans_{bank_id + 1}"


def translation_dataset(name, position_mm, decimals=DEFAULT_DECIMALS):
    """Translation from the sample to position_mm, as magnitude [m] + unit vector."""
    position = np.asarray(position_mm, dtype=float) * MM_TO_M
    magnitude = float(np.linalg.norm(position))
    vector = position / magnitude if magnitude > 0 else np.array([0.0, 0.0, 1.0])
    return dataset(
        name,
        round(magnitude, decimals),
        "float",
        [
            attribute("transformation_type", "translation"),
            attribute("depends_on", ".", "string"),
            attribute("vector", vector.tolist(), "float"),
            units_attribute(),
        ],
    )


def pixel_shape_vertices(bank_pixels):
    """Vertices of the pixel cylinder relative to the pixel reference point [mm]
    (as NeXus defines them): centre of the first face, a point on the edge of
    the first face, centre of the second face. The first face is the one towards
    the first pixel end; the reference point is at pixel_reference_fraction of
    the pixel length from it (0.5: centred shape, 0: shape starting there)."""
    frame = bank_pixels.frame
    pixel_vector = frame.along * bank_pixels.pixel_length
    first_face = -pixel_reference_fraction(bank_pixels.pixel_reference) * pixel_vector
    radius_vector = frame.row_axis * STRAW_RADIUS  # any direction perpendicular to the straw
    return np.array([first_face, first_face + radius_vector, first_face + pixel_vector])


def pixel_shape_group(bank_pixels, decimals=DEFAULT_DECIMALS):
    """NXcylindrical_geometry of one pixel (see pixel_shape_vertices), in [m]."""
    vertices = pixel_shape_vertices(bank_pixels) * MM_TO_M
    return group(
        "pixel_shape",
        [
            dataset("cylinders", [[0, 1, 2]], "int32"),
            dataset("vertices", _rounded_list(vertices, decimals), "float", [units_attribute()]),
        ],
        "NXcylindrical_geometry",
    )


def pixel_offset_datasets(bank_pixels, decimals=DEFAULT_DECIMALS):
    """x/y/z_pixel_offset datasets [m]."""
    offsets = bank_pixels.offsets * MM_TO_M
    return [
        dataset(
            f"{axis}_pixel_offset",
            _rounded_list(offsets[:, index], decimals),
            "float",
            [units_attribute()],
        )
        for index, axis in enumerate("xyz")
    ]


def bank_group(bank_pixels, decimals=DEFAULT_DECIMALS):
    """Geometry part of the NXdetector group of a bank."""
    bank_id = bank_pixels.bank_id
    children = [
        dataset("detector_number", bank_pixels.pixel_ids.tolist(), "int32"),
        pixel_shape_group(bank_pixels, decimals),
        *pixel_offset_datasets(bank_pixels, decimals),
    ]
    if bank_id == CARRIAGE_BANK_ID:
        depends_on = CARRIAGE_VALUE_PATH
    else:
        name = translation_name(bank_id)
        children.append(
            group(
                "transformations",
                [translation_dataset(name, bank_pixels.translation, decimals)],
                "NXtransformations",
            )
        )
        depends_on = f"{INSTRUMENT_PATH}/{bank_name(bank_id)}/transformations/{name}"
    children.append(dataset("depends_on", depends_on, "string"))
    return group(bank_name(bank_id), children, "NXdetector")


def build_output(instrument_pixels, corners_source, decimals=DEFAULT_DECIMALS, calibration=None):
    """The whole output dictionary (see the module docstring). corners_source: where the
    corners come from (a text, or {bank_id: text}); calibration: its name (recorded in info)."""
    pixels_per_straw = {bp.pixels_per_straw for bp in instrument_pixels.values()}
    pixel_references = {bp.pixel_reference for bp in instrument_pixels.values()}
    bank_origins = {bp.bank_origin for bp in instrument_pixels.values()}
    if isinstance(corners_source, dict):
        sources = {str(b): corners_source[b] for b in instrument_pixels}
        corners_source = "; ".join(sorted(set(sources.values())))
    else:
        sources = {str(b): corners_source for b in instrument_pixels}
    output = {
        "info": {
            "description": "LOKI detector pixel geometry (geometry parts of the loki_detector_<id> groups)",
            "written_by": f"loki-geometry {__version__}",
            "calibration": calibration,
            "corners_source": corners_source,
            "bank_corner_sources": sources,
            "pixels_per_straw": pixels_per_straw.pop()
            if len(pixels_per_straw) == 1
            else sorted(pixels_per_straw),
            "length_unit": LENGTH_UNIT,
            "pixel_reference": pixel_references.pop()
            if len(pixel_references) == 1
            else sorted(pixel_references),
            "bank_origin": bank_origins.pop() if len(bank_origins) == 1 else sorted(bank_origins),
            "bank_translation": "; ".join(
                BANK_ORIGIN_DESCRIPTIONS[origin]
                for origin in sorted({bp.bank_origin for bp in instrument_pixels.values()})
            ),
            "decimals": decimals,
        }
    }
    if CARRIAGE_BANK_ID in instrument_pixels:
        output["info"]["carriage_bank"] = (
            f"{bank_name(CARRIAGE_BANK_ID)} translation = {CARRIAGE_ZERO_NAME} "
            f"(in {INSTRUMENT_PATH}/detector_carriage/transformations)"
        )
    for bank_id, bank_pixels in instrument_pixels.items():
        output[bank_name(bank_id)] = bank_group(bank_pixels, decimals)
        if bank_id == CARRIAGE_BANK_ID:
            output[CARRIAGE_ZERO_NAME] = translation_dataset(
                CARRIAGE_ZERO_NAME, bank_pixels.translation, decimals
            )
    return output


VISUALIZE_PIXELS_PER_STRAW = 4  # --visualize: few points per straw for the viewers


def carriage_group(carriage_zero):
    """The detector carriage of bank 0 (as in loki.json): its position (value, 0 here: in the
    instrument file it comes from the motor), along z, depending on detector_carriage_zero."""
    value = group(
        "value",
        [dataset("value", 0.0, "double", [attribute("units", "mm", "string")])],
        "NXlog",
    )
    value["attributes"] += [
        attribute("vector", [0.0, 0.0, 1.0], "double"),
        attribute("depends_on", CARRIAGE_ZERO_PATH, "string"),
        attribute("transformation_type", "translation", "string"),
    ]
    return group(
        "detector_carriage",
        [
            value,
            group("transformations", [carriage_zero], "NXtransformations"),
            dataset("depends_on", CARRIAGE_ZERO_PATH, "string"),
        ],
        "NXpositioner",
    )


def carriage_bank_for_viewers(bank, carriage_zero):
    """Bank 0 placed by its own copy of detector_carriage_zero (--visualize, see the module
    docstring) instead of depending on the detector carriage."""
    name = bank["name"]
    children = [
        child for child in bank["children"] if child.get("config", {}).get("name") != "depends_on"
    ]
    children.append(group("transformations", [carriage_zero], "NXtransformations"))
    children.append(
        dataset(
            "depends_on", f"{INSTRUMENT_PATH}/{name}/transformations/{CARRIAGE_ZERO_NAME}", "string"
        )
    )
    return {**bank, "children": children}


def instrument_tree(output, visualize=False):
    """The nodes of an output dictionary (build_output) in the instrument tree that is written
    to the file (see the module docstring); visualize: bank 0 for viewers (--visualize)."""
    info = dict(output["info"])
    carriage_bank = bank_name(CARRIAGE_BANK_ID)
    if visualize:
        info["layout"] = (
            f"the loki_detector_<id> groups in {INSTRUMENT_PATH}; for viewers (--visualize): "
            f"{carriage_bank} is placed by its own {CARRIAGE_ZERO_NAME}, without the detector carriage"
        )
        if carriage_bank in output:
            info["carriage_bank"] = (
                f"{carriage_bank} translation = {CARRIAGE_ZERO_NAME} (in its own transformations)"
            )
    else:
        info["layout"] = (
            f"the loki_detector_<id> groups in {INSTRUMENT_PATH}, and the detector carriage of bank 0 "
            f"({CARRIAGE_ZERO_PATH}), as in the instrument file"
        )
    children = []
    for name, node in output.items():
        if name == carriage_bank and visualize:
            children.append(carriage_bank_for_viewers(node, output[CARRIAGE_ZERO_NAME]))
        elif name.startswith("loki_detector_"):
            children.append(node)
    if CARRIAGE_ZERO_NAME in output and not visualize:
        children.append(carriage_group(output[CARRIAGE_ZERO_NAME]))
    instrument = group("instrument", children, "NXinstrument")
    return {"info": info, "children": [group("entry", [instrument], "NXentry")]}


def tree_nodes(tree):
    """The inverse of instrument_tree: {"info": ..., "loki_detector_<id>": group, ...,
    "detector_carriage_zero": dataset} of a file written by loki-pixel-json."""
    nodes = {"info": tree.get("info")}
    for entry in tree.get("children", []):
        for instrument in entry.get("children", []):
            for node in instrument.get("children", []):
                if node.get("name", "").startswith("loki_detector_"):
                    nodes[node["name"]] = node
                elif node.get("name") == "detector_carriage":
                    for child in node.get("children", []):
                        if child.get("name") == "transformations":
                            for dataset_node in child.get("children", []):
                                nodes[dataset_node["config"]["name"]] = dataset_node
    return nodes


def write_json(output, filename):
    with open(filename, "w") as file:
        json.dump(output, file, separators=(",", ":"))


def reference_corners():
    """Corners of all banks from REFERENCE_BANKS_DATA (the intended geometry)."""
    from .reference_data import REFERENCE_BANKS_DATA

    return dict(REFERENCE_BANKS_DATA)


def calibration_corners(calibration):
    """(corners of all banks, {bank_id: source description}) of a calibration (by name,
    see geometry_info.available_calibrations)."""
    from . import geometry_info as gi

    corners = gi.calibration_corners(calibration)
    return corners, gi.calibration_bank_sources(calibration)


def _decimals(text):
    value = int(text)
    if value < MIN_DECIMALS:
        raise argparse.ArgumentTypeError(
            f"must be at least {MIN_DECIMALS} (0.1 mm per value), not {text}"
        )
    return value


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, not {text}")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Write the LOKI detector pixel geometry as NeXus-JSON nodes."
    )
    parser.add_argument(
        "-o", "--output", default="loki_detector_pixels.json", help="output JSON file"
    )
    from . import geometry_info as gi

    parser.add_argument(
        "-c",
        "--calibration",
        default=gi.DEFAULT_CALIBRATION,
        help=f"bank calibration (default: {gi.DEFAULT_CALIBRATION}, the actual geometry from the "
        f"default survey); also {gi.NOMINAL_CAD} (the intended CAD geometry) or {gi.NOMINAL_GEANT4}; "
        "see loki-info --list-calibrations",
    )
    parser.add_argument(
        "--pixels-per-straw",
        type=_positive_int,
        default=None,
        help=f"pixels along each straw (default: {DEFAULT_PIXELS_PER_STRAW}, with --visualize "
        f"{VISUALIZE_PIXELS_PER_STRAW})",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="a file for NeXus-JSON geometry viewers (e.g. https://github.com/chribe/jsondisplay): "
        f"{VISUALIZE_PIXELS_PER_STRAW} pixels per straw by default",
    )
    parser.add_argument(
        "--decimals",
        type=_decimals,
        default=DEFAULT_DECIMALS,
        help=f"rounding of the values in [m] (decimals), at least {MIN_DECIMALS} (default: {DEFAULT_DECIMALS}); "
        "a pixel position adds several rounded values, so it can be off by somewhat more than the rounding",
    )
    parser.add_argument(
        "--banks",
        type=int,
        nargs="+",
        choices=BANK_IDS,
        default=None,
        help="only these banks (default: all)",
    )
    parser.add_argument(
        "--pixel-reference",
        choices=PIXEL_REFERENCES,
        default=PIXEL_CENTRE,
        help="what the pixel offsets point to: pixel_centre (default) or pixel_base "
        "(centre of the pixel's first face); the pixel shape follows",
    )
    parser.add_argument(
        "--bank-origin",
        choices=BANK_ORIGINS,
        default=FIRST_PIXEL_END,
        help="the bank position (the bank translation), to which the pixel offsets are "
        f"relative: {FIRST_PIXEL_END} (default) = the first pixel end of the centre straw of "
        f"tube 0; {CENTRE_OF_MASS} = the mean of "
        "all pixel centres of the bank. The absolute pixel positions are the same for both",
    )
    args = parser.parse_args(argv)
    if args.pixels_per_straw is None:
        args.pixels_per_straw = (
            VISUALIZE_PIXELS_PER_STRAW if args.visualize else DEFAULT_PIXELS_PER_STRAW
        )

    try:
        corners, sources = calibration_corners(args.calibration)
    except ValueError as error:
        parser.error(str(error))
    if args.banks is not None:
        corners = {bank_id: corners[bank_id] for bank_id in args.banks}
    try:
        instrument_pixels = compute_instrument_pixels(
            corners,
            args.pixels_per_straw,
            pixel_reference=args.pixel_reference,
            bank_origin=args.bank_origin,
        )
    except ValueError as error:
        parser.error(f"calibration {args.calibration}: {error}")
    output = build_output(instrument_pixels, sources, args.decimals, calibration=args.calibration)
    try:
        write_json(instrument_tree(output, args.visualize), args.output)
    except OSError as error:
        parser.error(f"cannot write {args.output}: {error.strerror or error}")
    total = sum(len(bp.pixel_ids) for bp in instrument_pixels.values())
    print(
        f"Written {args.output}: calibration {args.calibration}, {len(instrument_pixels)} banks, {total} pixels, "
        f"{args.pixels_per_straw} pixels per straw, pixel reference: {args.pixel_reference}, "
        f"bank origin: {args.bank_origin}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
