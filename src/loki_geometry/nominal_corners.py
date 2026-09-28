# This script should get the distance to the bank centres (detector face at the
# centre of a given bank) and output the position of each banks corner positions
# that can be used to generate the loki json geometry.

import numpy as np
import math as m
import sys
import pprint
from .reference_data import REFERENCE_BANKS_DATA

# Bank represented by the end points of the 4 corner tubes.
class Bank:
    BANK_DISTANCES = np.array([
        5000.0,  # 0 - rear #TODO fix this
        2950.0,  # 1 - mid top
        3350.0,  # 2 - mid left
        2950.0,  # 3 - mid bottom
        3350.0,  # 4 - mid right
        1364.68, # 5 - front top
        1750.0,  # 6 - front left
        1340.0,  # 7 - front bottom
        1750.0,  # 8 - front right
    ])

    STRAW_LENGTHS = np.array([
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

    BANK_TILT_ANGLES = np.array([ # counterclockwise rotation angles
        m.radians(0.0),  # 0 - rear
        m.radians(-9.7 + (94 - 90)),  # 1 - mid top 
        m.radians(6.5 - (92 - 90)),   # 2 - mid left
        m.radians(9.7 - (94 - 90)),   # 3 - mid bottom
        m.radians(-6.5 + (92 - 90)),  # 4 - mid right
        m.radians(-32.20 + (104.7 - 90)), # 5 - front top
        m.radians(25.6 - (105.0 - 90)),   # 6 - front left
        m.radians(27.5 - (100 - 90)),     # 7 - front bottom #note 100=180-80, 80 deg is indicated in the tech drawing
        m.radians(-25.6 + (105.0 - 90)),  # 8 - front right
    ])

    BANK_POSITION_ANGLES = np.array([ # counterclockwise rotation angles
        m.radians(0.0),   # 0 - rear
        m.radians(9.7),   # 1 - mid top
        m.radians(6.5),   # 2 - mid left
        m.radians(-9.7),  # 3 - mid bottom
        m.radians(-6.5),  # 4 - mid right
        m.radians(32.20), # 5 - front top
        m.radians(25.6),  # 6 - front left
        m.radians(-27.5), # 7 - front bottom
        m.radians(-25.6), # 8 - front right
    ])

    BANK_OFFSETS = np.array([
        [0.0, 0.0, 0.0],    # 0 - rear
        [0.0, 0.0, 0.0],    # 1 - mid top
        [0.0, 0.0, 0.0],    # 2 - mid left
        [0.0, 0.0, 0.0],    # 3 - mid bottom
        [0.0, 0.0, 0.0],    # 4 - mid right
        [100.0, 0.0, 0.0],  # 5 - front top
        [0.0, -15.0, 0.0],  # 6 - front left
        [-100.0, 0.0, 0.0], # 7 - front bottom
        [0.0, 50.0, 0.0],   # 8 - front right
    ])

    PACKS_PER_BANK = np.array([
        28, # 0 - rear
        8,  # 1 - mid top
        6,  # 2 - mid left
        8,  # 3 - mid bottom
        6,  # 4 - mid right
        14, # 5 - front top
        16, # 6 - front left
        10, # 7 - front bottom
        16, # 8 - front right
    ])

    TUBE_GRID_BASE = 27.00  # [mm]
    TUBE_GRID_SIDE = 28.40  # [mm]
    TUBE_GRID_ANGLE = 13.45 # [mm] 90-76.55=13.45
    TUBE_RADIUS = 12.7 # [mm]
    BANK_DEPTH = 3 * TUBE_GRID_BASE # distance between first and last layer centers

    def __init__(self, bankId):
        self.bankId = bankId
        self.isVerticalBank = bankId in [2, 4, 6, 8]
        
        # 1. Start with baseline coordinates calculated from the bank dimensions
        self._initialize_corner_positions()
        
        # 2. Apply geometric transformations
        self._apply_vertical_bank_rotation()
        self._apply_bank_tilt()
        self._apply_bank_translation()
        self._apply_bank_offset()
        
        # Round all values to 2 decimal places
        self.positionA = np.round(self.positionA, 2)
        self.positionB = np.round(self.positionB, 2)

    def _initialize_corner_positions(self):
        """Calculates the initial A,B corner positions based on bank dimensions."""

        firstLayerToLastLayerVector = [0, 18.84, 78.78]
        topToBottomVector = [0, -self._get_bank_height(), 0]
        positionAtoBvector = [-self.STRAW_LENGTHS[self.bankId], 0, 0]

        firstTubePosition = [self.STRAW_LENGTHS[self.bankId] / 2, self._get_bank_height() / 2, 0]

        if self.bankId in [1, 2, 5, 6]:
            # These banks consist of packs mounted upside-down
            firstLayerToLastLayerVector[1] *= -1

        self.positionA = np.array([
                firstTubePosition,  # top tube in first layer
                np.add(firstTubePosition, firstLayerToLastLayerVector),  # top tube in last (4th) layer
                np.add(firstTubePosition, topToBottomVector),  # bottom tube in first layer
                np.add(np.add(firstTubePosition, firstLayerToLastLayerVector),topToBottomVector) # bottom tube in last (4th) lay
            ]
        )
        self.positionB = np.add(self.positionA, positionAtoBvector)

    def _apply_vertical_bank_rotation(self):
        """Rotates the bank by -90 degrees around the Z-axis if it's a vertical bank."""
        if self.isVerticalBank:
            # Standard CCW 90-degree rotation around Z-axis
            # For row-vector multiplication (v @ M), we need M.T.
            z_rotation_matrix_standard = Bank.rotation_matrix(-m.radians(90), "z")
            self.positionA = np.dot(self.positionA, z_rotation_matrix_standard.T)
            self.positionB = np.dot(self.positionB, z_rotation_matrix_standard.T)

    def _apply_bank_tilt(self):
        """Applies the tilt rotation to the bank."""
        if self.isVerticalBank:
            # Rotate anticlockwise around the -Y axis (equivalent to clockwise around +Y)
            # This is a clockwise rotation around +Y by bankTiltAngle[bankId].
            # A clockwise rotation by angle 'a' is a CCW rotation by '-a'.
            y_tilt_matrix_standard = Bank.rotation_matrix(self.BANK_TILT_ANGLES[self.bankId], "y")
            rotation_matrix_to_apply = y_tilt_matrix_standard.T
        else:
            # Rotate counterclockwise around the +X axis
            x_tilt_matrix_standard = Bank.rotation_matrix(self.BANK_TILT_ANGLES[self.bankId], "x")
            rotation_matrix_to_apply = x_tilt_matrix_standard.T

        self.positionA = np.dot(self.positionA, rotation_matrix_to_apply)
        self.positionB = np.dot(self.positionB, rotation_matrix_to_apply)

    def _apply_bank_translation(self):
        """Move the bank to it's place.
        The bank distance is the distance from the sample to the centre of the
        bank's FRONT FACE (a sheet laid on the front of the front layer tubes),
        measured in the section plane of the drawing (SI-7615-463) along the
        bank position angle. The corner points are tube centres, so the front
        face centre (at distance * direction) is moved by one tube radius along
        the bank NORMAL (not along the position direction, which differs from
        the normal by the tilt angle)."""
        angle = self.BANK_POSITION_ANGLES[self.bankId]
        if self.isVerticalBank:
            direction = np.array([np.sin(angle), 0, np.cos(angle)])
            # bank normal: +z rotated in the same way as the corner points
            normal = np.array([0.0, 0.0, 1.0]) @ Bank.rotation_matrix(-m.radians(90), "z").T @ Bank.rotation_matrix(self.BANK_TILT_ANGLES[self.bankId], "y").T
        else:
            direction = np.array([0, np.sin(angle), np.cos(angle)])
            normal = np.array([0.0, 0.0, 1.0]) @ Bank.rotation_matrix(self.BANK_TILT_ANGLES[self.bankId], "x").T

        translation = direction * self.BANK_DISTANCES[self.bankId] + normal * self.TUBE_RADIUS
        self.positionA = np.add(self.positionA, translation)
        self.positionB = np.add(self.positionB, translation)

    def _apply_bank_offset(self):
        """Apply bank offsets (e.g. asymmetric offset of the front banks)"""
        self.positionA = np.add(self.positionA, self.BANK_OFFSETS[self.bankId])
        self.positionB = np.add(self.positionB, self.BANK_OFFSETS[self.bankId])

    def _get_bank_height(self):
        numberOfTubesInALayer = self.PACKS_PER_BANK[self.bankId] * 2
        return (numberOfTubesInALayer - 1) * self.TUBE_GRID_SIDE

    @staticmethod
    def rotation_matrix(angle, axis):
        """
        Generates a 3x3 rotation matrix for a counter-clockwise rotation
        around the given axis.
        """
        c = np.cos(angle)
        s = np.sin(angle)
        if axis == "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == "y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif axis == "z":
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

class Detectors:
    BANK_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    def __init__(self):
        self.banks = {}
        for i in self.BANK_IDS:
            self.banks[i] = Bank(i)

    def print_formatted_data(self):
        """Returns the detector data as a dictionary matching the reference format."""
        bank_data = {}
        for bank_id, bank_instance in self.banks.items():
            # Use existing data as template, make a copy to avoid modifying the original
            bank_output = self._REFERENCE_BANKS_DATA.get(bank_id, {}).copy() 
            
            # Round coordinates and convert to tuples to match reference formatting
            bank_output["A"] = [tuple(pos) for pos in bank_instance.positionA]
            bank_output["B"] = [tuple(pos) for pos in bank_instance.positionB]
            
            bank_data[bank_id] = bank_output
        
        formatted_output = {bank_id: bank_data[bank_id] for bank_id in self.BANK_IDS}
    
        # Use pprint to preserve tuples () and integer keys in the console output
        pprint.pprint(formatted_output, sort_dicts=False, indent=4)
    
    # reference data from the generate-nexus repo (also serving as template for output)
    _REFERENCE_BANKS_DATA = REFERENCE_BANKS_DATA


def compare_arrays(arr1, arr2, tolerance=1.0):
    arr1, arr2 = np.array(arr1), np.array(arr2)
    diff = np.abs(arr1 - arr2)
    if np.any(diff > tolerance):
        # print(f"Arrays differ! Max difference: {np.max(diff)}")
        print(f"Arrays differ! Max difference: {np.round(np.max(diff), 2)}")
        print(f"Indices exceeding tolerance: {np.argwhere(diff > tolerance).tolist()}")
        print(f"Array 1: {arr1}")
        print(f"Array 2: {arr2}")
    else:
        print(f"Arrays match within tolerance ({tolerance} mm).")

def main(bankIds_to_process):
    detectors = Detectors()
    
    #TODO this comparison could be part of the Detector class
    for bankId in bankIds_to_process:
        positionA = detectors.banks[bankId].positionA
        positionB = detectors.banks[bankId].positionB
        
        reference_banks_data = detectors._REFERENCE_BANKS_DATA

        tolerance = 1 #[mm]
        print(f"\nComparing bank {bankId} position A to reference values:")
        compare_arrays(positionA, reference_banks_data[bankId]["A"], tolerance)

        # # TODO temp commented out
        # print(f"\nComparing bank {bankId} position B to reference values:")
        # compare_arrays(positionB, reference_banks_data[bankId]["B"], tolerance)
    
    # Generate and print data structure matching the reference template
    # detectors.print_formatted_data()


def cli():
    """Command line entry point (loki-nominal-corners [bank_id])."""
    bankIds = [int(sys.argv[1])] if len(sys.argv) == 2 else [0, 1, 2, 3, 4, 5, 6, 7, 8]
    main(bankIds)


if __name__ == "__main__":
    cli()