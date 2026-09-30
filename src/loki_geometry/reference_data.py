"""
The intended corner positions of the LOKI detector banks (exact CAD values):
REFERENCE_BANKS_DATA, the "intended" geometry of the comparisons in
calibration.py and the corners currently used by nexus_json.py.

Format, per bank id [mm]:
  'A': [front layer top tube, back layer top tube, front layer bottom tube,
        back layer bottom tube] at the FIRST pixel end of the straws, on the tube
        axis ("top/bottom" = lowest/highest local tube id),
  'B': the same 4 tubes at the LAST pixel end,
  'num_tubes': tubes in the bank.
convert_reference_data_to_calibration_data turns it into calibration data.
"""

REFERENCE_BANKS_DATA = {
    0: {
        "A": [
            (500, 781, 5012.5),
            (500, 799.84, 5091.28),
            (500, -781, 5012.5),
            (500, -762.16, 5091.28),
        ],
        "B": [
            (-500, 781, 5012.5),
            (-500, 799.84, 5091.28),
            (-500, -781, 5012.5),
            (-500, -762.16, 5091.28),
        ],
        "num_tubes": 224,
    },
    1: {
        "A": [
            (500, 710.24, 2899.18),
            (500, 699.24, 2979.43),
            (500, 286.34, 2941.42),
            (500, 275.35, 3021.67),
        ],
        "B": [
            (-500, 710.24, 2899.18),
            (-500, 699.24, 2979.43),
            (-500, 286.34, 2941.42),
            (-500, 275.35, 3021.67),
        ],
        "num_tubes": 64,
    },
    2: {
        "A": [
            (535.94, -250, 3328.75),
            (523.26, -250, 3408.75),
            (224.49, -250, 3353.18),
            (211.82, -250, 3433.19),
        ],
        "B": [
            (535.94, 250, 3328.75),
            (523.26, 250, 3408.75),
            (224.49, 250, 3353.18),
            (211.82, 250, 3433.19),
        ],
        "num_tubes": 48,
    },
    3: {
        "A": [
            (500, -286.33, 2941.34),
            (500, -275.34, 3021.59),
            (500, -710.23, 2899.11),
            (500, -699.24, 2979.36),
        ],
        "B": [
            (-500, -286.33, 2941.34),
            (-500, -275.34, 3021.59),
            (-500, -710.23, 2899.11),
            (-500, -699.24, 2979.36),
        ],
        "num_tubes": 64,
    },
    4: {
        "A": [
            (-224.49, -250, 3353.11),
            (-211.82, -250, 3433.11),
            (-535.93, -250, 3328.67),
            (-523.26, -250, 3408.67),
        ],
        "B": [
            (-224.49, 250, 3353.11),
            (-211.82, 250, 3433.11),
            (-535.93, 250, 3328.67),
            (-523.26, 250, 3408.67),
        ],
        "num_tubes": 48,
    },
    5: {
        "A": [
            (700, 1096.67, 1051.39),
            (700, 1102.32, 1132.19),
            (700, 365.34, 1281.9),
            (700, 370.99, 1362.7),
        ],
        "B": [
            (-500, 1096.67, 1051.39),
            (-500, 1102.32, 1132.19),
            (-500, 365.34, 1281.9),
            (-500, 370.99, 1362.7),
        ],
        "num_tubes": 112,
    },
    6: {
        "A": [
            (1191.15, -615, 1509.59),
            (1187.05, -615, 1590.49),
            (325.76, -615, 1671.47),
            (321.66, -615, 1752.37),
        ],
        "B": [
            (1191.15, 585, 1509.59),
            (1187.05, 585, 1590.49),
            (325.76, 585, 1671.47),
            (321.66, 585, 1752.37),
        ],
        "num_tubes": 128,
    },
    7: {
        "A": [
            (500, -365.36, 1281.97),
            (500, -371.01, 1362.77),
            (500, -880, 1119.78),
            (500, -885.65, 1200.58),
        ],
        "B": [
            (-700, -365.36, 1281.97),
            (-700, -371.01, 1362.77),
            (-700, -880, 1119.78),
            (-700, -885.65, 1200.58),
        ],
        "num_tubes": 80,
    },
    8: {
        "A": [
            (-325.62, -550, 1670.71),
            (-321.52, -550, 1751.61),
            (-1191.14, -550, 1509.52),
            (-1187.04, -550, 1590.41),
        ],
        "B": [
            (-325.62, 650, 1670.71),
            (-321.52, 650, 1751.61),
            (-1191.14, 650, 1509.52),
            (-1187.04, 650, 1590.41),
        ],
        "num_tubes": 128,
    },
}


def convert_reference_data_to_calibration_data(reference_data):
    """
    Converts REFERENCE_BANKS_DATA into the format used by calibration_data.
    Uses the first(top) and last(bottom) tubes of the first layer (indices 0
    and 2 in 'A' and 'B'). 'A' is the first pixel end (local pixel ids increase
    from +x to -x in horizontal banks, from -y to +y in vertical banks, in the
    NeXus coordinate system), 'B' is the last pixel end.
    """
    calib_data = []
    number_of_layers = 4
    for bank_id, data in reference_data.items():
        num_tubes_in_layer = int(data["num_tubes"] / number_of_layers)
        calib_data.append(
            {
                "bank_id": bank_id,
                "positions": [
                    {
                        "tube_id": 0,  # zero-indexed local tube id
                        "first_pixel_end": [x for x in data["A"][0]],
                        "last_pixel_end": [x for x in data["B"][0]],
                    },
                    {
                        "tube_id": num_tubes_in_layer - 1,  # zero-indexed local tube id
                        "first_pixel_end": [x for x in data["A"][2]],
                        "last_pixel_end": [x for x in data["B"][2]],
                    },
                ],
            }
        )
    return calib_data
