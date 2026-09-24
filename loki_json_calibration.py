

import numpy as np
import sys
import pprint

from example_calibration_data import example_calibration_data
from reference_banks_data import REFERENCE_BANKS_DATA, convert_reference_data_to_calibration_data

# The script will receive 4 points for each bank
# From these, the 2 vectors of the front plane should be calculated:
#  1) vector along the tubes
#  3) vector perpendicular to the tubes
# From these, all 4 corner positions in the Front should be calculated by knowing the
#  1) 28.40 [mm] tube-tube distance multiplied by the number of steps up / down to the top/bottom
#  2) tube length in the bank
# After the 4 Front corner positions are known, the vector perpendicular to the layers
# (with positive Z direction) should be derived
# Then all 4 Back corners can be calculated by from the Front corners by
#  1) moving 18.84 mm "UP" (in the front plane)
#  2) moving 78.78 mm "BACK" (along the perpendicular direction)

class Detectors:
    # bank_ids = []
    banks = {}

    tubes_per_layer_by_bank = np.array([
        2 * 28, # 0 - rear
        2 * 8,  # 1 - mid top
        2 * 6,  # 2 - mid left
        2 * 8,  # 3 - mid bottom
        2 * 6,  # 4 - mid right
        2 * 14, # 5 - front top
        2 * 16, # 6 - front left
        2 * 10, # 7 - front bottom
        2 * 16, # 8 - front right
    ])
    tube_length_by_bank = np.array([ #[mm]
        1000.0,  # 0 - rear
        1000.0,  # 1 - mid top
        500.0,   # 2 - mid left
        1000.0,  # 3 - mid bottom
        500.0,   # 4 - mid right
        1200.0,  # 5 - front top
        1200.0,  # 6 - front left
        1200.0,  # 7 - front bottom
        1200.0,  # 8 - front right
    ])
    intended_bank_tilt_angles = np.array([ #[degrees]
        0.0,  # 0 - rear
        5.7,  # 1 - mid top
        4.5,  # 2 - mid left
        5.7,  # 3 - mid bottom
        4.5,  # 4 - mid right
        17.5, # 5 - front top
        10.6, # 6 - front left
        17.5, # 7 - front bottom
        10.6, # 8 - front right
    ])
    intended_bank_distances = np.array([ #[mm]
        5000.0,  # 0 - rear #TODO this should depend on an input value
        2950.0,  # 1 - mid top
        3350.0,  # 2 - mid left
        2950.0,  # 3 - mid bottom
        3350.0,  # 4 - mid right
        1364.68, # 5 - front top
        1750.0,  # 6 - front left
        1340.0,  # 7 - front bottom
        1750.0,  # 8 - front right
    ])

    vertical_bank_ids = [2, 4, 6, 8] # ids of banks which consist of vertical tubes
    upside_down_bank_ids = [1, 2, 5, 6] # ids of banks which consist of detector packs placed upside down
    straw_pixel_number = 256 # Number of pixels in each straw

    REFERENCE_BANKS_DATA = REFERENCE_BANKS_DATA # reference data from the generate-nexus repo (also serving as template for output)

    def __init__(self, calibration_data, use_pixel_centre_position):
        for calibration_bank_data in calibration_data:
            bank_id = calibration_bank_data["bank_id"]
            self.banks[bank_id] = Bank(
                calibration_bank_data,
                is_upside_down_bank = bank_id in self.upside_down_bank_ids,
                tubes_per_layer = self.tubes_per_layer_by_bank[bank_id],
                tube_length = self.tube_length_by_bank[bank_id],
                straw_pixel_number = self.straw_pixel_number,
                use_pixel_centre_position = use_pixel_centre_position)

        self._check_vector_orthogonality()
        self._verify_bank_tilt_angle()
        self._verify_bank_distance()

    @property
    def bank_ids(self):
        """Sorted list of the bank ids"""
        return sorted([id for id in self.banks.keys()])

    def print_formatted_data(self):
        """Returns the detector data as a dictionary matching the reference format."""
        bank_data = {}
        for bank_id, bank_instance in self.banks.items():
            # Use existing data as template, make a copy to avoid modifying the original
            bank_output = self.REFERENCE_BANKS_DATA.get(bank_id, {}).copy()

            # Round coordinates and convert to tuples to match reference formatting
            bank_output["A"] = [tuple(pos) for pos in bank_instance.positionA]
            bank_output["B"] = [tuple(pos) for pos in bank_instance.positionB]

            bank_data[bank_id] = bank_output

        formatted_output = {bank_id: bank_data[bank_id] for bank_id in self.bank_ids}

        # Use pprint to preserve tuples () and integer keys in the console output
        pprint.pprint(formatted_output, sort_dicts=False, indent=4)

    def verify_positions_against_reference_data(self):
        for bank_id in self.bank_ids:
            tolerance = 1 #[mm] # NOTE this could be a input variable
            print(f"\nComparing bank {bank_id} position A to reference values:")
            self.compare_arrays(self.banks[bank_id].positionA,
                                self.REFERENCE_BANKS_DATA[bank_id]["A"],
                                tolerance)

            print(f"\nComparing bank {bank_id} position B to reference values:")
            self.compare_arrays(self.banks[bank_id].positionB,
                                self.REFERENCE_BANKS_DATA[bank_id]["B"],
                                tolerance)

    @staticmethod
    def compare_arrays(arr1, arr2, tolerance=1.0):
        """Utility function to check if two arrays are the same within a given tolerance."""
        arr1, arr2 = np.array(arr1), np.array(arr2)
        diff = np.abs(arr1 - arr2)
        if np.any(diff > tolerance):
            print(f"Arrays differ! Max difference: {np.round(np.max(diff), 2)}")
            print(f"Indices exceeding tolerance: {np.argwhere(diff > tolerance).tolist()}")
            print(f"Array 1: {arr1}")
            print(f"Array 2: {arr2}")
        else:
            print(f"Arrays match within tolerance ({tolerance} mm).")

    def _check_vector_orthogonality(self):
        """Verify that the base vectors are perpendicular in all banks"""
        all_vectors_are_orthogonal = True
        for bank_id, bank in self.banks.items():
            if not np.isclose(np.dot(bank.vector_along_tubes, bank.vector_perpendicular_tubes), 0):
                print(f"Warning: Bank {bank_id} vector_along_tubes and vector_perpendicular_tubes are not perpendicular!")
                all_vectors_are_orthogonal = False
            if not np.isclose(np.dot(bank.vector_along_tubes, bank.vector_perpendicular_layers), 0):
                print(f"Warning: Bank {bank_id} vector_along_tubes and vector_perpendicular_layers are not perpendicular!")
                all_vectors_are_orthogonal = False
            if not np.isclose(np.dot(bank.vector_perpendicular_tubes, bank.vector_perpendicular_layers), 0):
                print(f"Warning: Bank {bank_id} vector_perpendicular_tubes and vector_perpendicular_layers are not perpendicular!")
                all_vectors_are_orthogonal = False
        if all_vectors_are_orthogonal:
            print("All bank base vectors are orthogonal.")

    def _verify_bank_tilt_angle(self):
        angle_tolerance = 0.1
        all_tilt_angles_agree = True
        for bank_id, bank in self.banks.items():
            if bank_id in self.vertical_bank_ids:
                # Calculate tilt_angle as the angle between vector_perpendicular_tubes and [1.0, 0.0, 0.0]
                dot_product = np.dot(bank.vector_perpendicular_tubes, np.array([1.0, 0.0, 0.0]))
            else:
                # Calculate tilt_angle as the angle between vector_perpendicular_tubes and [0.0, 1.0, 0.0]
                dot_product = np.dot(bank.vector_perpendicular_tubes, np.array([0.0, 1.0, 0.0]))
            tilt_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

            if not np.isclose(np.degrees(tilt_angle), self.intended_bank_tilt_angles[bank_id], atol=angle_tolerance):
                print(f"Warning: Bank {bank_id} tilt angle {np.round(np.degrees(tilt_angle), 3)} degrees differs from intended {self.intended_bank_tilt_angles[bank_id]} degrees.")
                all_tilt_angles_agree = False
        if all_tilt_angles_agree:
            print(f"All bank tilt angles agree with the intended values within tolerance {angle_tolerance} degrees.")

    def _verify_bank_distance(self):
        tolerance = 1.0 # [mm]
        all_distances_agree = True
        for bank_id, bank in self.banks.items():
            if not np.isclose(bank.sample_bank_distance, self.intended_bank_distances[bank_id], atol=tolerance):
                print(f"Warning: Bank {bank_id} sample bank distance {int(np.round(bank.sample_bank_distance))} mm differs from intended {self.intended_bank_distances[bank_id]} mm.")
                all_distances_agree = False
        if all_distances_agree:
            print(f"All bank sample-bank distances agree with the intended values within tolerance {tolerance} mm.")


class Bank:
    tube_tube_distance = 28.40 #[mm]
    front_to_back_up_distance = 18.84 #[mm]
    front_to_back_in_distance = 78.78 #[mm]

    def __init__(self, calibration_bank_data, is_upside_down_bank, tubes_per_layer, tube_length,  straw_pixel_number, use_pixel_centre_position=True):
        self.is_upside_down_bank = is_upside_down_bank
        self.tubes_per_layer = tubes_per_layer
        self.tube_length = tube_length #could also be called straw_length
        self.straw_pixel_number = straw_pixel_number
        self.pixel_size = self.tube_length / self.straw_pixel_number

        self._calculate_corner_positions(calibration_bank_data, use_pixel_centre_position)
        self._calculate_sample_bank_distance(use_pixel_centre_position)

    def _calculate_corner_positions(self, calibration_bank_data, use_pixel_centre_position):
        """Calculates the 8 corner positions of the detector bank based on calibration data.
        The corner positions are the 2 ends of the 4 tubes in the corners of the bank:
        front layer top tube, back layer top tube, front layer bottom tube, back layer bottom tube.
        positionA denotes the 4 positions at the tube ends with the lower pixel ids.
        positionB denotes the 4 positions at the tube ends with the higher pixel ids.
        The end of a tube means the centre of the last pixel in the centre of a tube
        (in the central straw) if use_pixel_centre_position flag is True, otherwise
        the end of the tube.
        """

        pos0 = calibration_bank_data["positions"][0]
        pos1 = calibration_bank_data["positions"][1]

        # 'top' is the one with the lower tube_id
        upper_pos, lower_pos = (pos0, pos1) if pos0["tube_id"] < pos1["tube_id"] else (pos1, pos0)

        upper_tube_end = np.array(upper_pos["end_position"])
        upper_tube_in = np.array(upper_pos["inside_position"])
        lower_tube_end = np.array(lower_pos["end_position"])
        # lower_tube_in = np.array(lower_pos["inside_position"]) #use for validation?

        # Calculate perpendicular unit vectors
        v1 = upper_tube_end - upper_tube_in
        v2 = upper_tube_end - lower_tube_end
        self.vector_along_tubes = v1 / np.linalg.norm(v1)
        self.vector_perpendicular_tubes = v2 / np.linalg.norm(v2)
        v3 = np.cross(self.vector_along_tubes, self.vector_perpendicular_tubes)
        self.vector_perpendicular_layers = v3 / np.linalg.norm(v3)

        top_to_bottom_vector = - self.vector_perpendicular_tubes * (self.tubes_per_layer -1) * self.tube_tube_distance
        front_to_back_vector = self.vector_perpendicular_layers * self.front_to_back_in_distance + (self.vector_perpendicular_tubes * self.front_to_back_up_distance if not self.is_upside_down_bank else -self.vector_perpendicular_tubes * self.front_to_back_up_distance) #TODO clean this!

        front_top_tube_end = upper_tube_end + self.vector_perpendicular_tubes * upper_pos["tube_id"] * self.tube_tube_distance
        back_top_tube_end = front_top_tube_end + front_to_back_vector
        front_bottom_tube_end = front_top_tube_end + top_to_bottom_vector
        back_bottom_tube_end = front_bottom_tube_end + front_to_back_vector

        self.positionA = np.array([
                front_top_tube_end,  # top tube in first layer
                back_top_tube_end,  # top tube in last (4th) layer
                front_bottom_tube_end, # bottom tube in first layer
                back_bottom_tube_end  # bottom tube in last (4th) layer
                ]
        )
        self.positionB = self.positionA - self.vector_along_tubes * (self.tube_length - self.pixel_size)

        use_pixel_centre_position = True
        if use_pixel_centre_position:
            self.positionA -= self.vector_along_tubes * (self.pixel_size / 2)
            self.positionB -= self.vector_along_tubes * (self.pixel_size / 2)

    def _calculate_sample_bank_distance(self, use_pixel_centre_position):
        """Calculates the distance from the sample to the bank.
        The reference position is at the centre of the front face of the bank,
        defining the front face as the plane that could be placed at the front
        layer of tube, not the centre of these tubes.
        (see in DETAIL D in SI-7615-463 (LOKI DETECTOR GEOMETRY SKETCH)"""

        if not use_pixel_centre_position:
            return #TODO implement if needed

        # calculate as the average of the 4 front corner positions
        centre_at_front_layer_tube_centre = (self.positionA[0] + self.positionA[2] + self.positionB[0] + self.positionB[2]) / 4

        #move to the front face from the tube centre plane
        tube_radius = 12.7 # [mm]
        bank_front_face_position = centre_at_front_layer_tube_centre - self.vector_perpendicular_layers * tube_radius #DEBUG TEMP OFF
        self.sample_bank_distance = np.linalg.norm(bank_front_face_position)



def main():

    print("Using reference data from generate-nexus repo as calibration data")
    calibration_data = convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA)
    detectors = Detectors(calibration_data, use_pixel_centre_position=True)

    print("Using example calibration data from Geant4 model")
    calibration_data = example_calibration_data
    detectors = Detectors(calibration_data, use_pixel_centre_position=True)
    # detectors.print_formatted_data()

    # detectors.verify_positions_against_reference_data() #TODO temp commented


if __name__ == "__main__":
    main()
