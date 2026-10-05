"""
loki_geometry: geometry of the LOKI detector banks (ESS).

- Calibration of the 8 bank corner positions from survey data
  (loki_geometry.survey, loki_geometry.calibration); the known survey files
  and what is specific to each of them are in loki_geometry.survey_files.
- Pixel geometry (detector numbers and pixel offsets) from the bank corners,
  and its export as NeXus-JSON nodes for the instrument file
  (loki_geometry.bank_pixels, loki_geometry.nexus_json), with a verification
  against the Geant4 geometry (loki_geometry.verify_json).
- Geometry information (positions and ids of banks, tubes, straws and pixels)
  for a named bank calibration (loki_geometry.geometry_info, loki-info).
- A Python port of the Geant4 pixel positions (loki_geometry.geant4).
- The intended bank corners (loki_geometry.reference_data).
- The shared geometry constants (loki_geometry.constants).

Coordinates: x horizontal (positive to the left, looking along the beam), y up,
z along the beam, origin at the sample; mm unless stated otherwise.
"""

__version__ = "0.1.0"

from . import constants  # noqa: F401
from .bank_pixels import (  # noqa: F401
    PIXEL_BASE,
    PIXEL_CENTRE,
    BankFrame,
    BankPixels,
    compute_bank_pixels,
    compute_instrument_pixels,
)
from .reference_data import REFERENCE_BANKS_DATA  # noqa: F401
