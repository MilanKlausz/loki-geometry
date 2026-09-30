"""
Rebuild the intended corner positions of the banks from the nominal bank
distances, angles and offsets (drawing SI-7615-463 / Geant4, see
constants.py), and compare them with REFERENCE_BANKS_DATA.
"""

import argparse
import math as m
import pprint
import sys

import numpy as np

from . import constants as c
from .reference_data import REFERENCE_BANKS_DATA


def _signed_tilt(bank_id):
    """Counterclockwise tilt rotation of a bank [rad] (around x for horizontal,
    around -y for vertical banks)."""
    sign = c.bank_position_side(bank_id) * (1 if bank_id in c.VERTICAL_BANK_IDS else -1)
    return m.radians(sign * c.bank_tilt_angle(bank_id))


# Bank represented by the end points of the 4 corner tubes.
class Bank:
    BANK_DISTANCES = np.array(c.BANK_DISTANCES)
    STRAW_LENGTHS = np.array(c.STRAW_LENGTHS)
    BANK_TILT_ANGLES = np.array(
        [_signed_tilt(b) for b in c.BANK_IDS]
    )  # counterclockwise rotation angles
    BANK_POSITION_ANGLES = np.radians(c.BANK_POSITION_ANGLES)  # counterclockwise rotation angles
    BANK_OFFSETS = np.array(c.BANK_OFFSETS)
    PACKS_PER_BANK = np.array(c.PACKS_PER_BANK)

    TUBE_GRID_BASE = c.TUBE_GRID_BASE  # [mm]
    TUBE_GRID_SIDE = c.TUBE_GRID_SIDE  # [mm]
    TUBE_GRID_ANGLE = c.TUBE_GRID_ANGLE  # [deg]
    TUBE_RADIUS = c.TUBE_OUTER_RADIUS  # [mm]
    BANK_DEPTH = (
        c.LAYERS_PER_BANK - 1
    ) * TUBE_GRID_BASE  # distance between first and last layer centers

    def __init__(self, bankId):
        self.bankId = bankId
        self.isVerticalBank = bankId in c.VERTICAL_BANK_IDS

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

        firstLayerToLastLayerVector = [0, c.FRONT_TO_BACK_IN_PLANE, c.FRONT_TO_BACK_NORMAL]
        topToBottomVector = [0, -self._get_bank_height(), 0]
        positionAtoBvector = [-self.STRAW_LENGTHS[self.bankId], 0, 0]

        firstTubePosition = [self.STRAW_LENGTHS[self.bankId] / 2, self._get_bank_height() / 2, 0]

        if self.bankId in c.UPSIDE_DOWN_BANK_IDS:
            # These banks consist of packs mounted upside-down
            firstLayerToLastLayerVector[1] *= -1

        self.positionA = np.array(
            [
                firstTubePosition,  # top tube in first layer
                np.add(
                    firstTubePosition, firstLayerToLastLayerVector
                ),  # top tube in last (4th) layer
                np.add(firstTubePosition, topToBottomVector),  # bottom tube in first layer
                np.add(
                    np.add(firstTubePosition, firstLayerToLastLayerVector), topToBottomVector
                ),  # bottom tube in last (4th) lay
            ]
        )
        self.positionB = np.add(self.positionA, positionAtoBvector)

    def _apply_vertical_bank_rotation(self):
        """Rotates the bank by -90 degrees around the Z-axis if it's a vertical bank."""
        if self.isVerticalBank:
            # Clockwise 90-degree rotation around the Z-axis (-90 degrees)
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
            normal = (
                np.array([0.0, 0.0, 1.0])
                @ Bank.rotation_matrix(-m.radians(90), "z").T
                @ Bank.rotation_matrix(self.BANK_TILT_ANGLES[self.bankId], "y").T
            )
        else:
            direction = np.array([0, np.sin(angle), np.cos(angle)])
            normal = (
                np.array([0.0, 0.0, 1.0])
                @ Bank.rotation_matrix(self.BANK_TILT_ANGLES[self.bankId], "x").T
            )

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
    BANK_IDS = list(c.BANK_IDS)

    def __init__(self):
        self.banks = {}
        for i in self.BANK_IDS:
            self.banks[i] = Bank(i)

    def print_formatted_data(self):
        """Print the corner data in the REFERENCE_BANKS_DATA format."""
        corners = {
            bank_id: {
                "A": [tuple(float(v) for v in pos) for pos in bank.positionA],
                "B": [tuple(float(v) for v in pos) for pos in bank.positionB],
                "num_tubes": c.TUBES_PER_BANK[bank_id],
            }
            for bank_id, bank in self.banks.items()
        }
        # pprint keeps the tuples and the integer keys
        pprint.pprint(corners, sort_dicts=False, indent=4)

    # the intended corners
    _REFERENCE_BANKS_DATA = REFERENCE_BANKS_DATA


def compare_arrays(arr1, arr2, tolerance=1.0):
    """Print whether two arrays agree within tolerance [mm]; returns True if they do."""
    arr1, arr2 = np.array(arr1), np.array(arr2)
    diff = np.abs(arr1 - arr2)
    if np.any(diff > tolerance):
        # print(f"Arrays differ! Max difference: {np.max(diff)}")
        print(f"Arrays differ! Max difference: {np.round(np.max(diff), 2)}")
        print(f"Indices exceeding tolerance: {np.argwhere(diff > tolerance).tolist()}")
        print(f"Array 1: {arr1}")
        print(f"Array 2: {arr2}")
        return False
    print(f"Arrays match within tolerance ({tolerance} mm).")
    return True


def main(bankIds_to_process):
    """Compare the rebuilt corners of the given banks with REFERENCE_BANKS_DATA;
    returns True if all agree within 1 mm."""
    detectors = Detectors()
    all_match = True

    # TODO this comparison could be part of the Detector class
    for bankId in bankIds_to_process:
        positionA = detectors.banks[bankId].positionA
        positionB = detectors.banks[bankId].positionB

        reference_banks_data = detectors._REFERENCE_BANKS_DATA

        tolerance = 1  # [mm]
        print(f"\nComparing bank {bankId} position A to reference values:")
        all_match &= compare_arrays(positionA, reference_banks_data[bankId]["A"], tolerance)

        print(f"\nComparing bank {bankId} position B to reference values:")
        all_match &= compare_arrays(positionB, reference_banks_data[bankId]["B"], tolerance)
    return all_match


def cli(argv=None):
    """Command line entry point (loki-nominal-corners [bank_id ...])."""
    parser = argparse.ArgumentParser(
        description="Rebuild the intended bank corners from the nominal distances, angles and "
        "offsets, and compare them with REFERENCE_BANKS_DATA."
    )
    # (no choices=: with nargs="*" argparse < 3.12 rejects the empty default)
    parser.add_argument(
        "bank_ids", type=int, nargs="*", metavar="bank_id", help="banks to compare (default: all)"
    )
    args = parser.parse_args(argv)
    unknown = sorted(set(args.bank_ids) - set(c.BANK_IDS))
    if unknown:
        parser.error(f"unknown bank ids {unknown} (choose from {list(c.BANK_IDS)})")
    return 0 if main(args.bank_ids or list(c.BANK_IDS)) else 1


if __name__ == "__main__":
    sys.exit(cli())
