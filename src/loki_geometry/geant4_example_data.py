# Calibration data taken from the Geant4 geometry (for testing the calibration
# and the Geant4 geometry itself).
# Generated with geant4.py (a validated Python port of the
# AimHelper::getPixelCentreCoordinates of G4GeoLoki), 256 pixels per straw, rear
# bank at 5 m, positions of the CENTRE straw (straw 3, on the tube axis) [mm].
# Format (same as produced by survey.py):
#   "tube_id": zero-indexed local tube id in the FRONT layer (horizontal banks:
#              0 = topmost tube, vertical banks: 0 = leftmost tube, beam's view)
#              (= the Geant4 tube id, for the front layer rows)
#   "first_pixel_end" / "last_pixel_end": the two ends of that tube on its axis.
#              The local pixel ids of a straw increase from +x to -x (horizontal
#              banks) and from -y to +y (vertical banks) in the NeXus coordinate
#              system, so first_pixel_end is the +x (horizontal) / -y (vertical) end.
#   "distance": |first_pixel_end - last_pixel_end|
# NOTE the end positions are the centres of the first and last pixels (Geant4
# pixel 0 and 255 of the straw), not the real straw ends: they are half a pixel
# inside at both ends. The calibration only uses the tube midpoints and
# directions, so this does not change its result (use
# survey.adjust_calibration_data_to_effective_tube_lengths
# to get the straw ends if needed).
example_calibration_data = [
    {
        "bank_id": 0,
        "positions": [
            {
                "tube_id": 7,  # zero-indexed local tube id (Geant4 pixels 13312 and 13567)
                "first_pixel_end": [498.046875, 582.200000, 5012.700000],
                "last_pixel_end": [-498.046875, 582.200000, 5012.700000],
                "distance": 996.093750,
            },
            {
                "tube_id": 9,  # zero-indexed local tube id (Geant4 pixels 16896 and 17151)
                "first_pixel_end": [498.046875, 525.400000, 5012.700000],
                "last_pixel_end": [-498.046875, 525.400000, 5012.700000],
                "distance": 996.093750,
            },
        ],
    },
    {
        "bank_id": 1,
        "positions": [
            {
                "tube_id": 5,  # zero-indexed local tube id (Geant4 pixels 411136 and 411391)
                "first_pixel_end": [498.046875, 568.953976, 2913.410737],
                "last_pixel_end": [-498.046875, 568.953976, 2913.410737],
                "distance": 996.093750,
            },
            {
                "tube_id": 8,  # zero-indexed local tube id (Geant4 pixels 416512 and 416767)
                "first_pixel_end": [498.046875, 484.175241, 2921.872780],
                "last_pixel_end": [-498.046875, 484.175241, 2921.872780],
                "distance": 996.093750,
            },
        ],
    },
    {
        "bank_id": 2,
        "positions": [
            {
                "tube_id": 5,  # zero-indexed local tube id (Geant4 pixels 525824 and 526079)
                "first_pixel_end": [394.383423, -249.023438, 3340.012447],
                "last_pixel_end": [394.383423, 249.023438, 3340.012447],
                "distance": 498.046875,
            },
            {
                "tube_id": 8,  # zero-indexed local tube id (Geant4 pixels 531200 and 531455)
                "first_pixel_end": [309.446066, -249.023438, 3346.697162],
                "last_pixel_end": [309.446066, 249.023438, 3346.697162],
                "distance": 498.046875,
            },
        ],
    },
    {
        "bank_id": 3,
        "positions": [
            {
                "tube_id": 5,  # zero-indexed local tube id (Geant4 pixels 611840 and 612095)
                "first_pixel_end": [498.046875, -427.656085, 2927.514142],
                "last_pixel_end": [-498.046875, -427.656085, 2927.514142],
                "distance": 996.093750,
            },
            {
                "tube_id": 8,  # zero-indexed local tube id (Geant4 pixels 617216 and 617471)
                "first_pixel_end": [498.046875, -512.434820, 2919.052099],
                "last_pixel_end": [-498.046875, -512.434820, 2919.052099],
                "distance": 996.093750,
            },
        ],
    },
    {
        "bank_id": 4,
        "positions": [
            {
                "tube_id": 5,  # zero-indexed local tube id (Geant4 pixels 726528 and 726783)
                "first_pixel_end": [-366.070970, -249.023438, 3342.240686],
                "last_pixel_end": [-366.070970, 249.023438, 3342.240686],
                "distance": 498.046875,
            },
            {
                "tube_id": 8,  # zero-indexed local tube id (Geant4 pixels 731904 and 732159)
                "first_pixel_end": [-451.008327, -249.023438, 3335.555971],
                "last_pixel_end": [-451.008327, 249.023438, 3335.555971],
                "distance": 498.046875,
            },
        ],
    },
    {
        "bank_id": 5,
        "positions": [
            {
                "tube_id": 5,  # zero-indexed local tube id (Geant4 pixels 812544 and 812799)
                "first_pixel_end": [697.656250, 961.251832, 1094.304715],
                "last_pixel_end": [-497.656250, 961.251832, 1094.304715],
                "distance": 1195.312500,
            },
            {
                "tube_id": 8,  # zero-indexed local tube id (Geant4 pixels 817920 and 818175)
                "first_pixel_end": [697.656250, 879.995148, 1119.924849],
                "last_pixel_end": [-497.656250, 879.995148, 1119.924849],
                "distance": 1195.312500,
            },
        ],
    },
    {
        "bank_id": 6,
        "positions": [
            {
                "tube_id": 4,  # zero-indexed local tube id (Geant4 pixels 1011456 and 1011711)
                "first_pixel_end": [1079.512928, -612.656250, 1530.611689],
                "last_pixel_end": [1079.512928, 582.656250, 1530.611689],
                "distance": 1195.312500,
            },
            {
                "tube_id": 6,  # zero-indexed local tube id (Geant4 pixels 1015040 and 1015295)
                "first_pixel_end": [1023.682200, -612.656250, 1541.060126],
                "last_pixel_end": [1023.682200, 582.656250, 1541.060126],
                "distance": 1195.312500,
            },
        ],
    },
    {
        "bank_id": 7,
        "positions": [
            {
                "tube_id": 4,  # zero-indexed local tube id (Geant4 pixels 1240832 and 1241087)
                "first_pixel_end": [497.656250, -473.591518, 1247.676968],
                "last_pixel_end": [-697.656250, -473.591518, 1247.676968],
                "distance": 1195.312500,
            },
            {
                "tube_id": 6,  # zero-indexed local tube id (Geant4 pixels 1244416 and 1244671)
                "first_pixel_end": [497.656250, -527.762640, 1230.596878],
                "last_pixel_end": [-697.656250, -527.762640, 1230.596878],
                "distance": 1195.312500,
            },
        ],
    },
    {
        "bank_id": 8,
        "positions": [
            {
                "tube_id": 9,  # zero-indexed local tube id (Geant4 pixels 1393152 and 1393407)
                "first_pixel_end": [-577.036377, -547.656250, 1624.647619],
                "last_pixel_end": [-577.036377, 647.656250, 1624.647619],
                "distance": 1195.312500,
            },
            {
                "tube_id": 12,  # zero-indexed local tube id (Geant4 pixels 1398528 and 1398783)
                "first_pixel_end": [-660.782469, -547.656250, 1608.974964],
                "last_pixel_end": [-660.782469, 647.656250, 1608.974964],
                "distance": 1195.312500,
            },
        ],
    },
]
