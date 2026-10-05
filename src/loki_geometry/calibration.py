"""
LOKI detector bank calibration: calculate the 8 corner positions of each
detector bank (the input of the pixel geometry, bank_pixels.py / nexus_json.py)
from the positions of two tubes of the bank.

Usage
-----
  loki-calibrate [--data measured|reference|example] [--survey-file FILE]
                 [--verbose] [--compare-to-intended] [--print]
                 [--geant4-calibration-file PATH [--geant4-calibration-name NAME]]
  --data measured   a survey file, via survey.py (default; the SAM-606 survey
                    unless --survey-file is given)
  --data reference  the intended geometry (REFERENCE_BANKS_DATA) -> must
                    reproduce it (self test of the algorithm, the corners are
                    compared with REFERENCE_BANKS_DATA)
  --data example    Geant4 positions from geant4_example_data.py
  --verbose         print the quality report (see Detectors.print_quality_report)
  --compare-to-intended  print the corner shifts from the intended banks
  --print           print the corners (straw ends) in the REFERENCE_BANKS_DATA format
  --geant4-calibration-file PATH  write the bank calibration file of the Geant4
                    geometry (see geant4_calibration.py) to PATH; if PATH is a
                    directory (e.g. G4GeoLoki/data of dgcode_projects), the
                    file bank_calibration_<NAME>.txt is written in it; all 9
                    banks, those missing from the input (e.g. bank 0 of
                    SAM-606) from REFERENCE_BANKS_DATA
  --geant4-calibration-name NAME  the calibration name (the bank_calibration
                    value of the Geant4 geometry); default: the calibration_name
                    of the survey file entry (survey_files.py), required for
                    --data reference / example
The "intended" geometry = the banks built from REFERENCE_BANKS_DATA (exact CAD
values, also used by nexus_json.py). The nominal Geant4 table values in
Detectors (tilt angles, distances) are rounded CAD values; they are printed
for information.

Input (calibration data), per bank
----------------------------------
Two tubes of the FRONT layer (the layer facing the sample), each given by
  "tube_id":         zero-indexed local tube id; horizontal banks (0,1,3,5,7):
                     0 = topmost tube, vertical banks (2,4,6,8): 0 = leftmost
                     tube (looking along the beam)
  "first_pixel_end": point on the tube axis at the straw end on the side of
                     the first pixel
  "last_pixel_end":  point on the tube axis at the straw end on the side of
                     the last pixel
The local pixel ids of a straw increase from +x to -x (horizontal banks) and
from -y to +y (vertical banks) in the NeXus coordinate system (x left, y up,
z along the beam, origin at the sample), [mm].
The points must be on the tube CENTRE LINE (axis), not on the tube surface.
The two end points may be wrong by the same amount at both ends (e.g. the
measured physical tube ends include ~15 mm of end electronics at both ends, the
Geant4 end pixel centres are half a pixel inside): the algorithm only uses the
midpoint and the direction of each tube, which such errors do not change. The
nominal (effective) straw length of the bank is used for the tube length.
The input is checked (ValueError): exactly 2 tubes per bank and each bank
once, the two tube ids must differ, the two tubes must point the same way, and
the layer normal must point away from the sample (swapped first / last pixel
ends, or survey data in a mirrored frame, would put the back layers in front).

Output
------
positionA / positionB of each bank: the first / last pixel ends of the 4
corner tubes, in the order [front layer top tube, back layer top tube, front
layer bottom tube, back layer bottom tube] ("top/bottom" = lowest/highest tube
id). With use_pixel_centre_position=True (as used by loki-calibrate) they are
pixel CENTRE positions (moved half a pixel inwards from the straw ends),
otherwise the straw ends. formatted_data() gives the straw ends in both cases.

Algorithm (Bank._calculate_corner_positions)
--------------------------------------------
 1. Order the two input tubes by tube id (a = lower id, closer to tube 0).
 2. Direction along the tubes u: the average of the two tubes' unit direction
    vectors (last -> first pixel end). The angle between them is recorded
    (the tubes of a bank are expected to be parallel).
 3. Least-squares plane through the 4 input points (numpy SVD of the centred
    points: the normal is the singular vector with the smallest singular
    value). The 4 measured points are never exactly coplanar; their distances
    from the plane are recorded.
 4. u is projected into the fitted plane.
 5. In-plane vector perpendicular to the tubes w = plane normal x u, oriented
    from the higher-id tube towards the lower-id tube (towards tube 0, "up").
 6. Layer normal n = u x w. (u, w, n) is an exact orthonormal basis.
 7. Each input tube predicts the midpoint of tube 0 (its own midpoint + w *
    tube_id * 28.40 mm); the average of the two predictions is used. The
    difference of the predictions is recorded (along u: the two tubes are
    shifted along their length; along w: the tube spacing is not tube id
    difference * 28.40 mm).
 8. Corners: front top first pixel end = tube-0 midpoint + u * L/2 (L =
    effective straw length); front bottom = front top - w * (N-1) * 28.40 mm
    (N = tubes per layer); back layer = front layer + 78.78 mm * n +- 18.84 mm
    * w (the sign depends on whether the packs are mounted upside down); the
    last pixel ends = first pixel ends - u * L; finally everything is moved
    half a pixel inwards to get pixel centres.
 9. End-to-end check (Bank._check_reproduction_of_input): the input tubes
    are recomputed from the final corners and their midpoints compared with
    the input.
Then Detectors checks the orthogonality of the basis, the tilt angles and the
sample-bank distances against the intended values.
"""

import os
import pprint
import sys
import textwrap

import numpy as np

from . import __version__, geant4_calibration
from . import constants as c
from .geant4_example_data import example_calibration_data
from .reference_data import REFERENCE_BANKS_DATA, convert_reference_data_to_calibration_data


class Detectors:
    # Bank constants (single source: constants.py)
    tubes_per_layer_by_bank = np.array(c.TUBES_PER_LAYER)
    tube_length_by_bank = np.array(c.STRAW_LENGTHS)  # [mm] effective straw length
    # Nominal (Geant4) placement, used when no intended geometry is given (see
    # constants.py for how the distances are defined: in the section plane,
    # to the front face centre, with the offsets along the tubes on top).
    intended_bank_tilt_angles = np.array(c.BANK_TILT_ANGLES)  # [deg]
    intended_bank_distances = np.array(
        c.BANK_DISTANCES
    )  # [mm] #TODO the rear bank distance should be an input value
    intended_bank_position_angles = np.array(
        c.BANK_POSITION_ANGLES
    )  # [deg], + is up (horizontal) / left (vertical)
    intended_bank_offsets = np.array(c.BANK_OFFSETS)  # [mm] (x, y, z)
    vertical_bank_ids = c.VERTICAL_BANK_IDS
    upside_down_bank_ids = c.UPSIDE_DOWN_BANK_IDS
    straw_pixel_number = c.DEFAULT_PIXELS_PER_STRAW

    REFERENCE_BANKS_DATA = REFERENCE_BANKS_DATA  # the intended corners

    def __init__(
        self,
        calibration_data,
        use_pixel_centre_position,
        verbose=False,
        intended="reference",
        run_checks=True,
    ):
        """calibration_data: list of per bank dictionaries (see the module docstring).
        intended: what the banks are compared with (tilt angles, sample-bank
          distances, front face centres, corner shifts):
          "reference" (default) - the geometry built from REFERENCE_BANKS_DATA
                                  (exact CAD values, used by nexus_json.py),
          a Detectors object    - that geometry,
          None                  - the nominal Geant4 table values below
                                  (intended_bank_tilt_angles, intended_bank_distances, ...).
        run_checks: print the orthogonality / tilt / distance checks."""
        # Instance attribute (not a class attribute), so that several Detectors
        # objects (e.g. measured and intended geometry) don't share their banks
        self.banks = {}
        for calibration_bank_data in calibration_data:
            bank_id = calibration_bank_data["bank_id"]
            if bank_id in self.banks:
                raise ValueError(f"bank {bank_id} appears twice in the calibration data")
            self.banks[bank_id] = Bank(
                calibration_bank_data,
                is_upside_down_bank=bank_id in self.upside_down_bank_ids,
                tubes_per_layer=self.tubes_per_layer_by_bank[bank_id],
                tube_length=self.tube_length_by_bank[bank_id],
                straw_pixel_number=self.straw_pixel_number,
                use_pixel_centre_position=use_pixel_centre_position,
                bank_id=bank_id,
            )

        if isinstance(intended, str) and intended == "reference":
            intended = Detectors(
                convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA),
                use_pixel_centre_position,
                verbose=False,
                intended=None,
                run_checks=False,
            )
        self.intended = intended
        self.intended_label = (
            "REFERENCE_BANKS_DATA" if intended is not None else "Geant4 nominal values"
        )

        if run_checks:
            self._check_vector_orthogonality()
            self._verify_bank_tilt_angle()
            self._verify_bank_distance()
        if verbose:
            self.print_quality_report()

    @property
    def bank_ids(self):
        """Sorted list of the bank ids"""
        return sorted([id for id in self.banks.keys()])

    def formatted_data(self, as_straw_ends=True, decimals=2):
        """The corner data as a dictionary in the REFERENCE_BANKS_DATA format.

        as_straw_ends: bank_pixels.py expects the 'A'/'B' points at the
          STRAW ENDS (it places the pixel centres at (j + 0.5) * L / N from 'A'),
          like the values in REFERENCE_BANKS_DATA, while positionA/B are pixel
          centres. With True (default) the points are moved back half a pixel
          outwards along the tubes."""
        bank_data = {}
        for bank_id in self.bank_ids:
            bank = self.banks[bank_id]
            positionA, positionB = bank.positionA, bank.positionB
            if as_straw_ends and bank.positions_are_pixel_centres:
                half_pixel = bank.vector_along_tubes * bank.pixel_size / 2
                positionA, positionB = positionA + half_pixel, positionB - half_pixel
            # rounded, as tuples of plain floats (the REFERENCE_BANKS_DATA format)
            bank_data[bank_id] = {
                "A": [tuple(round(float(v), decimals) for v in pos) for pos in positionA],
                "B": [tuple(round(float(v), decimals) for v in pos) for pos in positionB],
                "num_tubes": c.TUBES_PER_BANK[bank_id],
            }
        return bank_data

    def print_formatted_data(self, as_straw_ends=True):
        """Print the corner data in the REFERENCE_BANKS_DATA format (see formatted_data)."""
        # Use pprint to preserve tuples () and integer keys in the console output
        pprint.pprint(self.formatted_data(as_straw_ends), sort_dicts=False, indent=4)

    def verify_positions_against_reference_data(self, tolerance=1.0):
        """Compare the corners (as straw ends, like REFERENCE_BANKS_DATA) with
        REFERENCE_BANKS_DATA; prints the result per bank and returns the largest
        difference of a coordinate [mm]."""
        corners = self.formatted_data(decimals=6)
        worst = 0.0
        for bank_id in self.bank_ids:
            for end in ("A", "B"):
                print(f"\nComparing bank {bank_id} position {end} to reference values:")
                self.compare_arrays(
                    corners[bank_id][end], self.REFERENCE_BANKS_DATA[bank_id][end], tolerance
                )
                difference = np.abs(
                    np.subtract(corners[bank_id][end], self.REFERENCE_BANKS_DATA[bank_id][end])
                )
                worst = max(worst, float(difference.max()))
        return worst

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

    def print_quality_report(self):
        """Print the quality indicators of each bank (see Bank.quality):
        - plane fit: the largest distance of the 4 input points from the
          least-squares plane (and which point it is),
        - the angle between the 2 input tubes (parallelism),
        - the spacing of the 2 input tubes in pitches vs the id difference,
        - how differently the 2 tubes predict the position of tube 0,
        - end-to-end: how well the final 8 corners reproduce the input
          tube ends (residuals along / across the tubes / out of plane)."""
        point_names = ["tube a first", "tube a last", "tube b first", "tube b last"]
        print("\n=== Quality report ===")
        for bank_id in self.bank_ids:
            bank = self.banks[bank_id]
            q = bank.quality
            ta, tb = bank.input_tubes
            worst = int(np.argmax(np.abs(q["plane_residuals"])))
            print(f"\nBank {bank_id} (input tubes {ta['tube_id']} and {tb['tube_id']}):")
            print(
                f"  plane fit: largest outlier {q['plane_residuals'][worst]:+.3f} mm ({point_names[worst]}), "
                f"rms {np.sqrt(np.mean(q['plane_residuals'] ** 2)):.3f} mm"
            )
            print(f"  angle between the 2 input tubes: {q['tube_angle_deg']:.3f} deg")
            print(
                f"  tube spacing: {q['spacing_in_pitches']:.3f} pitches (id difference {tb['tube_id'] - ta['tube_id']}, "
                f"deviation {(q['spacing_in_pitches'] - (tb['tube_id'] - ta['tube_id'])) * bank.tube_tube_distance:+.2f} mm)"
            )
            along, across, normal = q["tube0_prediction_difference"]
            print(
                f"  tube 0 predicted by the 2 tubes differs by: along {along:+.2f}, across {across:+.2f}, normal {normal:+.2f} mm"
            )
            print(
                "  reproduction of the input tubes by the final corners (input - predicted midpoint) [mm]:"
            )
            for tube_id, (r_along, r_across, r_normal), length_difference in q[
                "reproduction_residuals"
            ]:
                print(
                    f"    tube {tube_id:2d}: along {r_along:+6.2f}  across {r_across:+6.2f}  normal {r_normal:+6.2f}"
                    f"   (input length - straw length: {length_difference:+.2f})"
                )
            max_residual = max(np.linalg.norm(r) for _, r, _ in q["reproduction_residuals"])
            print(f"    largest 3D midpoint residual: {max_residual:.2f} mm")
            tilt, intended_tilt = self.tilt_angle(bank_id), self.intended_tilt(bank_id)
            print(
                f"  tilt angle: {tilt:.3f} deg, intended {intended_tilt:.3f} deg ({self.intended_label}; Geant4 nominal "
                f"{self.intended_bank_tilt_angles[bank_id]:.3f}), difference {tilt - intended_tilt:+.3f} deg"
            )
            intended_centre = self.intended_centre(bank_id)
            intended_distance = np.linalg.norm(intended_centre)
            shift = bank.front_face_centre - intended_centre
            print(
                f"  sample - front face centre distance: {bank.sample_bank_distance:.2f} mm, intended {intended_distance:.2f} mm "
                f"({self.intended_label}; Geant4 nominal {np.linalg.norm(self.intended_front_face_centre(bank_id)):.2f}), "
                f"difference {bank.sample_bank_distance - intended_distance:+.2f} mm"
            )
            print(
                f"  front face centre shift from intended: x {shift[0]:+.2f}  y {shift[1]:+.2f}  z {shift[2]:+.2f} mm"
            )
            (d, a, o), (di, ai, oi), (dn, an, on) = (
                self.section_parameters(bank_id),
                self.intended_section_parameters(bank_id),
                self.nominal_section_parameters(bank_id),
            )
            print(
                "  as in the drawing / Geant4 (section plane distance, position angle, offset along the tubes):"
            )
            print(
                f"    distance {d:8.2f} mm, intended {di:8.2f} (Geant4 nominal {dn}), difference {d - di:+6.2f} mm"
            )
            print(
                f"    angle    {a:8.3f} deg, intended {ai:8.3f} (Geant4 nominal {an}), difference {a - ai:+6.3f} deg"
            )
            print(
                f"    offset   {o:8.2f} mm, intended {oi:8.2f} (Geant4 nominal {on}), difference {o - oi:+6.2f} mm"
            )

        print(
            f"\nBank placement as in the drawing / Geant4 (measured - intended, intended = {self.intended_label}):"
        )
        print(
            "  bank | section distance [mm]      diff | position angle [deg]    diff | offset along tubes [mm]    diff"
        )
        for bank_id in self.bank_ids:
            (d, a, o), (di, ai, oi) = (
                self.section_parameters(bank_id),
                self.intended_section_parameters(bank_id),
            )
            print(
                f"  {bank_id:4d} |      {d:8.2f} {di:8.2f} {d - di:+7.2f} |      {a:7.3f} {ai:7.3f} {a - ai:+6.3f} |       {o:7.2f} {oi:7.2f} {o - oi:+6.2f}"
            )

        print(f"\nSummary (measured - intended, intended = {self.intended_label}):")
        print(
            "  bank | tilt [deg]  measured intended   diff | front face distance [mm] measured intended    diff | largest midpoint residual [mm]"
        )
        for bank_id in self.bank_ids:
            bank = self.banks[bank_id]
            tilt, intended_tilt = self.tilt_angle(bank_id), self.intended_tilt(bank_id)
            intended_distance = np.linalg.norm(self.intended_centre(bank_id))
            max_residual = max(
                np.linalg.norm(r) for _, r, _ in bank.quality["reproduction_residuals"]
            )
            print(
                f"  {bank_id:4d} |             {tilt:8.3f} {intended_tilt:8.3f} {tilt - intended_tilt:+6.3f} |"
                f"                          {bank.sample_bank_distance:8.2f} {intended_distance:8.2f} {bank.sample_bank_distance - intended_distance:+7.2f} | {max_residual:.2f}"
            )

    def compare_to(self, other, label="other"):
        """Compare the corner positions with another Detectors object (e.g. the
        intended geometry built from REFERENCE_BANKS_DATA). Prints, per common
        bank, the mean corner shift vector (self - other) and the largest
        corner distance [mm]."""
        print(f"\n=== Corner positions compared to {label} (self - {label}) [mm] ===")
        for bank_id in self.bank_ids:
            if bank_id not in other.banks:
                continue
            mine = np.concatenate([self.banks[bank_id].positionA, self.banks[bank_id].positionB])
            theirs = np.concatenate(
                [other.banks[bank_id].positionA, other.banks[bank_id].positionB]
            )
            diff = mine - theirs
            mean = diff.mean(axis=0)
            print(
                f"  bank {bank_id}: mean shift x {mean[0]:+7.2f} y {mean[1]:+7.2f} z {mean[2]:+7.2f}, "
                f"largest corner distance {np.max(np.linalg.norm(diff, axis=1)):6.2f}"
            )

    def _check_vector_orthogonality(self):
        """Verify that the base vectors are perpendicular in all banks"""
        all_vectors_are_orthogonal = True
        for bank_id, bank in self.banks.items():
            if not np.isclose(np.dot(bank.vector_along_tubes, bank.vector_perpendicular_tubes), 0):
                print(
                    f"Warning: Bank {bank_id} vector_along_tubes and vector_perpendicular_tubes are not perpendicular!"
                )
                all_vectors_are_orthogonal = False
            if not np.isclose(np.dot(bank.vector_along_tubes, bank.vector_perpendicular_layers), 0):
                print(
                    f"Warning: Bank {bank_id} vector_along_tubes and vector_perpendicular_layers are not perpendicular!"
                )
                all_vectors_are_orthogonal = False
            if not np.isclose(
                np.dot(bank.vector_perpendicular_tubes, bank.vector_perpendicular_layers), 0
            ):
                print(
                    f"Warning: Bank {bank_id} vector_perpendicular_tubes and vector_perpendicular_layers are not perpendicular!"
                )
                all_vectors_are_orthogonal = False
        if all_vectors_are_orthogonal:
            print("All bank base vectors are orthogonal.")

    def intended_tilt(self, bank_id):
        """Intended tilt angle [deg]: from the intended geometry if given, else the nominal table."""
        if self.intended is not None and bank_id in self.intended.banks:
            return self.intended.tilt_angle(bank_id)
        return self.intended_bank_tilt_angles[bank_id]

    def intended_centre(self, bank_id):
        """Intended front face centre [mm]: from the intended geometry if given, else nominal."""
        if self.intended is not None and bank_id in self.intended.banks:
            return self.intended.banks[bank_id].front_face_centre
        return self.intended_front_face_centre(bank_id)

    def section_parameters(self, bank_id, centre=None):
        """Decompose a front face centre into the quantities the drawing
        (SI-7615-463) and the Geant4 geometry use:
          distance - in the section plane perpendicular to the tubes (the y-z
                     plane for horizontal banks, the x-z plane for vertical banks)
          angle    - bank position angle in that plane [deg], signed like
                     intended_bank_position_angles
          offset   - the component along the tube axis direction (x for
                     horizontal, y for vertical banks) = the bank offset
        Returns (distance, angle, offset)."""
        c = self.banks[bank_id].front_face_centre if centre is None else centre
        if bank_id in self.vertical_bank_ids:
            offset, transverse = c[1], c[0]
        else:
            offset, transverse = c[0], c[1]
        return np.hypot(transverse, c[2]), np.degrees(np.arctan2(transverse, c[2])), offset

    def nominal_section_parameters(self, bank_id):
        """Geant4 nominal (distance, position angle, offset along the tubes)."""
        axis = 1 if bank_id in self.vertical_bank_ids else 0
        return (
            self.intended_bank_distances[bank_id],
            self.intended_bank_position_angles[bank_id],
            self.intended_bank_offsets[bank_id][axis],
        )

    def intended_section_parameters(self, bank_id):
        """Intended (distance, position angle, offset): from the intended geometry if given, else nominal."""
        if self.intended is not None and bank_id in self.intended.banks:
            return self.intended.section_parameters(bank_id)
        return self.nominal_section_parameters(bank_id)

    def intended_front_face_centre(self, bank_id):
        """Intended position of the bank's front face centre from the nominal
        placement (see BANK_DISTANCES in constants.py)."""
        angle = np.radians(self.intended_bank_position_angles[bank_id])
        distance = self.intended_bank_distances[bank_id]
        if bank_id in self.vertical_bank_ids:
            direction = np.array([np.sin(angle), 0.0, np.cos(angle)])
        else:
            direction = np.array([0.0, np.sin(angle), np.cos(angle)])
        return distance * direction + self.intended_bank_offsets[bank_id]

    def tilt_angle(self, bank_id):
        """Tilt angle of a bank [degrees]: the angle between the in-plane vector
        perpendicular to the tubes and the y axis (horizontal banks) or the
        x axis (vertical banks). Only the magnitude is returned."""
        bank = self.banks[bank_id]
        axis = (
            np.array([1.0, 0.0, 0.0])
            if bank_id in self.vertical_bank_ids
            else np.array([0.0, 1.0, 0.0])
        )
        dot_product = np.dot(bank.vector_perpendicular_tubes, axis)
        return np.degrees(np.arccos(np.clip(abs(dot_product), -1.0, 1.0)))

    def _verify_bank_tilt_angle(self):
        angle_tolerance = 0.1
        all_tilt_angles_agree = True
        for bank_id in self.banks:
            tilt_angle, intended = self.tilt_angle(bank_id), self.intended_tilt(bank_id)
            if not np.isclose(tilt_angle, intended, atol=angle_tolerance):
                print(
                    f"Warning: Bank {bank_id} tilt angle {tilt_angle:.3f} degrees differs from intended {intended:.3f} degrees ({self.intended_label})."
                )
                all_tilt_angles_agree = False
        if all_tilt_angles_agree:
            print(
                f"All bank tilt angles agree with the intended values ({self.intended_label}) within tolerance {angle_tolerance} degrees."
            )

    def _verify_bank_distance(self):
        """Compare the sample - front face centre distance with the intended one
        (see intended_centre / intended_front_face_centre)."""
        tolerance = 1.0  # [mm]
        all_distances_agree = True
        for bank_id, bank in self.banks.items():
            intended = np.linalg.norm(self.intended_centre(bank_id))
            if not np.isclose(bank.sample_bank_distance, intended, atol=tolerance):
                print(
                    f"Warning: Bank {bank_id} sample bank distance {bank.sample_bank_distance:.1f} mm differs from intended {intended:.1f} mm ({self.intended_label})."
                )
                all_distances_agree = False
        if all_distances_agree:
            print(
                f"All bank sample-bank distances agree with the intended values ({self.intended_label}) within tolerance {tolerance} mm."
            )


class Bank:
    """One detector bank, described by its 8 corner positions.

    positionA / positionB: the first / last pixel ends of the 4 corner tubes
    (order: front layer top tube, back layer top tube, front layer bottom tube,
    back layer bottom tube; "top/bottom" = lowest/highest local tube id, i.e.
    left/right for vertical banks). See the module docstring for the algorithm.
    """

    tube_tube_distance = c.TUBE_GRID_SIDE  # [mm] distance of neighbouring tubes in a layer
    front_to_back_up_distance = (
        c.FRONT_TO_BACK_IN_PLANE
    )  # [mm] front->back layer shift in the tube plane (18.84)
    front_to_back_in_distance = (
        c.FRONT_TO_BACK_NORMAL
    )  # [mm] front->back layer shift along the normal (78.78)

    def __init__(
        self,
        calibration_bank_data,
        is_upside_down_bank,
        tubes_per_layer,
        tube_length,
        straw_pixel_number,
        use_pixel_centre_position=True,
        bank_id=None,
    ):
        self.bank_id = bank_id
        self.is_upside_down_bank = is_upside_down_bank
        self.tubes_per_layer = tubes_per_layer
        self.tube_length = tube_length  # could also be called straw_length
        self.straw_pixel_number = straw_pixel_number
        self.pixel_size = self.tube_length / self.straw_pixel_number
        self.quality = {}  # quality indicators of the input data and of the fit (see print_quality_report)

        self._calculate_corner_positions(calibration_bank_data, use_pixel_centre_position)
        self._check_reproduction_of_input()
        self._calculate_sample_bank_distance(use_pixel_centre_position)

    def _calculate_corner_positions(self, calibration_bank_data, use_pixel_centre_position):
        """Calculates the 8 corner positions of the detector bank from the two
        calibration tubes (see the module docstring for the full description).
        positionA denotes the 4 positions at the tube ends with the lower pixel ids.
        positionB denotes the 4 positions at the tube ends with the higher pixel ids.
        The end of a tube means the centre of the last pixel in the centre of a tube
        (in the central straw) if use_pixel_centre_position flag is True, otherwise
        the end of the tube.
        """
        # --- Step 1: the input tubes, ordered by local tube id --------------
        # tube_a = the tube with the lower id (closer to tube 0), tube_b = the other
        if len(calibration_bank_data["positions"]) != 2:
            raise ValueError(
                f"bank {self.bank_id}: exactly 2 input tubes are needed, "
                f"got {len(calibration_bank_data['positions'])}"
            )
        tube_a, tube_b = sorted(calibration_bank_data["positions"], key=lambda pos: pos["tube_id"])
        if tube_a["tube_id"] == tube_b["tube_id"]:
            raise ValueError(
                f"bank {self.bank_id}: the two input tubes have the same id {tube_a['tube_id']}"
            )
        self.input_tubes = []  # kept for the quality checks
        for tube in (tube_a, tube_b):
            first = np.array(tube["first_pixel_end"], dtype=float)
            last = np.array(tube["last_pixel_end"], dtype=float)
            self.input_tubes.append(
                {
                    "tube_id": tube["tube_id"],
                    "first": first,
                    "last": last,
                    "mid": (first + last) / 2,
                }
            )
        ta, tb = self.input_tubes

        # --- Step 2: direction along the tubes = average of the 2 tubes ------
        # Points from the last pixel end to the first pixel end (as before:
        # first - last). Averaging the two measured directions halves the
        # effect of the measurement uncertainty of the end points.
        dir_a = _unit(ta["first"] - ta["last"])
        dir_b = _unit(tb["first"] - tb["last"])
        if np.dot(dir_a, dir_b) <= 0:
            raise ValueError(
                f"bank {self.bank_id}: the input tubes {ta['tube_id']} and {tb['tube_id']} point in "
                "opposite directions (first / last pixel ends swapped in one of them?)"
            )
        self.quality["tube_angle_deg"] = np.degrees(
            np.arccos(np.clip(np.dot(dir_a, dir_b), -1.0, 1.0))
        )
        u = _unit(dir_a + dir_b)

        # --- Step 3: least-squares plane through the 4 points ----------------
        # The singular vector with the smallest singular value of the centred
        # points is the normal of the best fitting plane (minimal sum of squared
        # orthogonal distances). The 4 measured points are never exactly
        # coplanar; the residuals are a measure of the input quality.
        points = np.array([ta["first"], ta["last"], tb["first"], tb["last"]])
        centroid = points.mean(axis=0)
        _, _, vt = np.linalg.svd(points - centroid)
        plane_normal = vt[2]
        self.quality["plane_residuals"] = (points - centroid) @ plane_normal  # signed [mm]

        # --- Step 4: put the tube direction exactly into the fitted plane ----
        u = _unit(u - np.dot(u, plane_normal) * plane_normal)

        # --- Step 5: in-plane direction perpendicular to the tubes -----------
        # Orthogonal to u within the plane; oriented to point from the tube with
        # the higher id towards the one with the lower id, i.e. towards tube 0
        # ("up" in the bank, same convention as before).
        w = np.cross(plane_normal, u)
        if np.dot(w, ta["mid"] - tb["mid"]) < 0:
            w = -w

        # --- Step 6: normal of the layers, completing the orthonormal basis --
        # Same convention as before: along x perpendicular_tubes.
        n = np.cross(u, w)
        # n must point away from the sample (towards the back layers); it
        # points towards it if the first / last pixel ends are swapped (u flipped)
        if np.dot(n, centroid) <= 0:
            raise ValueError(
                f"bank {self.bank_id}: the layer normal points towards the sample: first / last "
                "pixel ends swapped in the input, or a mirrored frame (determinant -1) in the "
                "frame conversion of the survey file?"
            )

        self.vector_along_tubes = u
        self.vector_perpendicular_tubes = w
        self.vector_perpendicular_layers = n

        # --- Step 7: position of tube 0 (front layer), from both tubes -------
        # The midpoint of a tube does not depend on the uncertain end
        # positions as long as the ends are wrong symmetrically (e.g. the
        # ~15 mm end electronics at both ends of a measured tube, or half a
        # pixel at both ends when pixel centres are given). Each tube predicts
        # the midpoint of tube 0 by stepping tube_id pitches towards tube 0;
        # the average of the two predictions is used.
        prediction_a = ta["mid"] + w * ta["tube_id"] * self.tube_tube_distance
        prediction_b = tb["mid"] + w * tb["tube_id"] * self.tube_tube_distance
        tube0_mid = (prediction_a + prediction_b) / 2
        # Disagreement of the two predictions (decomposed in the bank frame):
        # along = offset of the 2 tubes along the tube direction, across = the
        # tube spacing does not match N*pitch, normal = out of plane (0 by construction)
        difference = prediction_a - prediction_b
        self.quality["tube0_prediction_difference"] = np.array(
            [np.dot(difference, u), np.dot(difference, w), np.dot(difference, n)]
        )
        # measured tube spacing in units of the pitch (should be integer = id difference)
        d = tb["mid"] - ta["mid"]
        self.quality["spacing_in_pitches"] = (
            np.linalg.norm(d - np.dot(d, u) * u) / self.tube_tube_distance
        )

        # --- Step 8: the 8 corners (unchanged logic from here) ---------------
        front_top_tube_end = (
            tube0_mid + u * self.tube_length / 2
        )  # first pixel end of tube 0, front layer

        top_to_bottom_vector = (
            -self.vector_perpendicular_tubes * (self.tubes_per_layer - 1) * self.tube_tube_distance
        )
        front_to_back_vector = self.vector_perpendicular_layers * self.front_to_back_in_distance + (
            self.vector_perpendicular_tubes * self.front_to_back_up_distance
            if not self.is_upside_down_bank
            else -self.vector_perpendicular_tubes * self.front_to_back_up_distance
        )  # TODO clean this!

        back_top_tube_end = front_top_tube_end + front_to_back_vector
        front_bottom_tube_end = front_top_tube_end + top_to_bottom_vector
        back_bottom_tube_end = front_bottom_tube_end + front_to_back_vector

        self.positionA = np.array(
            [
                front_top_tube_end,  # top tube in first layer
                back_top_tube_end,  # top tube in last (4th) layer
                front_bottom_tube_end,  # bottom tube in first layer
                back_bottom_tube_end,  # bottom tube in last (4th) layer
            ]
        )
        # last pixel ends of the 4 corner tubes (straw ends)
        self.positionB = self.positionA - self.vector_along_tubes * self.tube_length

        # optionally move both ends half a pixel inwards, to the first / last pixel centres
        self.positions_are_pixel_centres = use_pixel_centre_position
        if use_pixel_centre_position:
            half_pixel = self.vector_along_tubes * (self.pixel_size / 2)
            self.positionA = self.positionA - half_pixel
            self.positionB = self.positionB + half_pixel

    def _check_reproduction_of_input(self):
        """End-to-end test: recompute the input tubes from the FINAL corner
        positions and compare them with the input.

        Tube k of the front layer lies on the line between the front top and
        front bottom corners, k/(N-1) of the way from the top one. For each input
        tube the residual of its MIDPOINT (input - predicted) is decomposed into
        along the tubes / across the tubes in the plane / along the normal; it
        shows how well the regular tube grid of the final bank reproduces the
        measured positions. The midpoint is used (instead of the two ends)
        because the input ends may be off symmetrically by design (pixel centres
        instead of straw ends, or the end electronics of the measured tubes);
        the length difference (input length - effective straw length) is
        reported separately."""
        half_pixel = (
            self.vector_along_tubes * self.pixel_size / 2
            if self.positions_are_pixel_centres
            else 0.0
        )
        top_first = self.positionA[0] + half_pixel  # back to the straw end
        bottom_first = self.positionA[2] + half_pixel
        step = (bottom_first - top_first) / (self.tubes_per_layer - 1)
        basis = np.array(
            [
                self.vector_along_tubes,
                self.vector_perpendicular_tubes,
                self.vector_perpendicular_layers,
            ]
        )
        residuals = []
        for tube in self.input_tubes:
            predicted_first = top_first + step * tube["tube_id"]
            predicted_mid = predicted_first - self.vector_along_tubes * self.tube_length / 2
            input_length = np.linalg.norm(tube["first"] - tube["last"])
            residuals.append(
                (
                    tube["tube_id"],
                    basis @ (tube["mid"] - predicted_mid),
                    input_length - self.tube_length,
                )
            )
        self.quality["reproduction_residuals"] = (
            residuals  # [(tube_id, [along, across, normal], length difference)]
        )

    def _calculate_sample_bank_distance(self, use_pixel_centre_position):
        """Calculates the distance from the sample to the bank.
        The reference position is at the centre of the front face of the bank,
        defining the front face as the plane that could be placed at the front
        layer of tube, not the centre of these tubes.
        (see in DETAIL D in SI-7615-463 (LOKI DETECTOR GEOMETRY SKETCH)"""

        # calculate as the average of the 4 front corner positions (the same
        # for pixel centres and straw ends, as both are symmetric along the tubes)
        centre_at_front_layer_tube_centre = (
            self.positionA[0] + self.positionA[2] + self.positionB[0] + self.positionB[2]
        ) / 4

        # move to the front face from the tube centre plane
        bank_front_face_position = (
            centre_at_front_layer_tube_centre
            - self.vector_perpendicular_layers * c.TUBE_OUTER_RADIUS
        )  # normal points away from the sample
        self.front_face_centre = bank_front_face_position
        self.sample_bank_distance = np.linalg.norm(bank_front_face_position)


def _unit(vector):
    """Normalised copy of a vector"""
    return vector / np.linalg.norm(vector)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate the 8 corner positions of the LOKI detector banks from calibration data."
    )
    parser.add_argument(
        "--data",
        choices=["measured", "reference", "example"],
        default="measured",
        help="calibration input: a survey measurement (see --survey-file), the intended geometry "
        "(REFERENCE_BANKS_DATA) or the Geant4 example data (default: measured)",
    )
    parser.add_argument(
        "--survey-file",
        default=None,
        help="survey file for --data measured (a path, or the name of a packaged file); it must "
        "have an entry in survey_files.py (default: the default survey of survey_files.py)",
    )
    parser.add_argument("--verbose", action="store_true", help="print the quality report")
    parser.add_argument(
        "--compare-to-intended",
        action="store_true",
        help="compare the corners with the intended geometry (REFERENCE_BANKS_DATA)",
    )
    parser.add_argument(
        "--print",
        dest="print_data",
        action="store_true",
        help="print the corner data (straw ends) in the REFERENCE_BANKS_DATA format",
    )
    parser.add_argument(
        "--geant4-calibration-file",
        metavar="PATH",
        default=None,
        help="write the bank calibration file of the Geant4 geometry (G4GeoLoki) to PATH; if PATH "
        "is a directory (e.g. LOKI/G4GeoLoki/data of dgcode_projects), the file "
        "bank_calibration_<NAME>.txt is written in it",
    )
    parser.add_argument(
        "--geant4-calibration-name",
        metavar="NAME",
        default=None,
        help="name of the calibration (the bank_calibration value of the Geant4 geometry); default: "
        "the calibration_name of the survey file entry in survey_files.py",
    )
    args = parser.parse_args(argv)
    if args.survey_file is not None and args.data != "measured":
        parser.error("--survey-file only applies to --data measured")
    if args.geant4_calibration_name is not None and args.geant4_calibration_file is None:
        parser.error("--geant4-calibration-name needs --geant4-calibration-file")

    calibration_name = args.geant4_calibration_name
    if args.data == "reference":
        print("Using the intended geometry (REFERENCE_BANKS_DATA) as calibration data")
        calibration_data = convert_reference_data_to_calibration_data(REFERENCE_BANKS_DATA)
        provenance = ["Input: the intended geometry (REFERENCE_BANKS_DATA of loki-geometry)."]
    elif args.data == "example":
        print("Using example calibration data from Geant4 model")
        calibration_data = example_calibration_data
        provenance = ["Input: the Geant4 example data (geant4_example_data.py of loki-geometry)."]
    else:
        from .survey import get_measured_calibration_data
        from .survey_files import DEFAULT_SURVEY_FILE

        survey_file = args.survey_file or DEFAULT_SURVEY_FILE
        print(f"Using the survey {survey_file} as calibration data")

        try:
            calibration_data = get_measured_calibration_data(survey_file, verbose=args.verbose)
        except OSError as error:
            parser.error(f"cannot read {survey_file}: {error.strerror or error}")
        except ValueError as error:
            parser.error(str(error))
        from .survey_files import SURVEY_FILES

        entry = SURVEY_FILES[os.path.basename(os.fspath(survey_file))]
        if calibration_name is None:
            calibration_name = entry.calibration_name
        provenance = [f"Input: the survey file {entry.file_name}"]
        if entry.checksum is not None:
            provenance.append(f"  (content SHA-256 {entry.checksum})")
        if entry.description:
            provenance += textwrap.wrap(entry.description, 88)

    if args.geant4_calibration_file is not None:
        if calibration_name is None:
            parser.error(
                "--geant4-calibration-file needs --geant4-calibration-name (no default name for "
                "this input)"
            )
        try:
            geant4_calibration.check_name(calibration_name)
        except ValueError as error:
            parser.error(str(error))

    try:
        detectors = Detectors(
            calibration_data, use_pixel_centre_position=True, verbose=args.verbose
        )
    except ValueError as error:
        parser.error(str(error))

    if args.compare_to_intended:
        detectors.compare_to(detectors.intended, label="intended (REFERENCE_BANKS_DATA)")

    if args.print_data:
        detectors.print_formatted_data()

    if args.geant4_calibration_file is not None:
        comments = [f"Written by loki-geometry {__version__} (loki-calibrate --data {args.data})."]
        comments += provenance
        try:
            # banks missing from the input (e.g. bank 0 of SAM-606): the intended geometry
            path = geant4_calibration.write_calibration_file(
                detectors,
                args.geant4_calibration_file,
                calibration_name,
                comments,
                fallback=detectors.intended,
                fallback_label="REFERENCE_BANKS_DATA (the intended geometry)",
            )
        except OSError as error:
            parser.error(f"cannot write {args.geant4_calibration_file}: {error.strerror or error}")
        except ValueError as error:
            parser.error(str(error))
        print(f"\nGeant4 bank calibration '{calibration_name}' written to {path}")
        if path.name != geant4_calibration.file_name_for(calibration_name):
            print(
                f"  (to select it by name, the file must be G4GeoLoki/data/"
                f"{geant4_calibration.file_name_for(calibration_name)}; otherwise use "
                f"bank_calibration=<path of the file>)"
            )

    if args.data == "reference":
        # self test: the corners must reproduce REFERENCE_BANKS_DATA (the
        # differences are the rounding of the CAD values, < 0.1 mm)
        tolerance = 0.1
        worst = detectors.verify_positions_against_reference_data(tolerance)
        print(f"\nSelf test: largest difference from REFERENCE_BANKS_DATA {worst:.3f} mm")
        return 0 if worst <= tolerance else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
