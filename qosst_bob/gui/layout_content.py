# qosst-bob - Bob module of the Quantum Open Software for Secure Transmissions.
# Copyright (C) 2021-2024 Yoann Piétri

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Enumerations of content in the GUI, and definition of some constants.
"""
import os
from pathlib import Path
from enum import Enum

from qosst_bob import __version__

DEFAULT_CONFIG_LOCATION = Path(os.getcwd()) / "config.toml"
LOGO_LOCATION = Path(__file__).parent / "logo.png"

THEME = "DarkGrey14"  #: The used theme.

SEQUENCE = [38, 38, 40, 40, 37, 39, 37, 39, 98, 97]

LICENSE_TEXT = """
qosst-bob - Bob module of the Quantum Open Software for Secure Transmissions.
Copyright (C) 2021-2024 Yoann Piétri

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


class QOSSTGUIContent(Enum):
    """
    A generic object for enumeration of content in the GUI.
    """


class QOSSTGUIActions(QOSSTGUIContent):
    """
    Enumeration of actions in the GUI (i.e. buttons).
    """

    EXIT = "-EXIT-"
    BROWSE = "-BROWSE-"
    READ_CONFIGURATION = "-READ-CONFIGURATION-"
    INITIALIZE_HARDWARE = "-INITIALIZE-HARDWARE-"
    CONNECT = "-CONNECT-"
    IDENTIFICATION = "-IDENTIFICATION-"
    INITIALIZATION = "-INITIALIZATION-"
    QIE = "-QIE-"
    DSP = "-DSP-"
    PARAMETERS_ESTIMATION = "-PARAMETERS-ESTIMATION-"
    ERROR_CORRECTION = "-ERROR-CORRECTION-"
    PRIVACY_AMPLITICATION = "-PRIVACY-AMPLIFICATION-"
    ACQUISITION_ELEC_NOISE = "-ACQUISITION-ELEC-NOISE-"
    LOAD_ELEC_NOISE = "-LOAD-ELEC-NOISE-"
    SAVE_ELEC_NOISE = "-SAVE-ELEC-NOISE-"
    ACQUISITION_SHOT_NOISE = "-ACQUISITION-SHOT-NOISE-"
    LOAD_SHOT_NOISE = "-LOAD-SHOT-NOISE-"
    SAVE_SHOT_NOISE = "-SAVE-SHOT-NOISE-"
    EXPORT_ELEC_NOISE = "-EXPORT-ELEC-NOISE-"
    EXPORT_SHOT_NOISE = "-EXPORT-SHOT-NOISE-"
    EXPORT_SIGNAL = "-EXPORT-SIGNAL-"
    ABOUT = "-ABOUT-"


class QOSSTGUIText(QOSSTGUIContent):
    """
    Enumerator of updatable text content in the GUI.
    """

    CONNECTION_STATUS = "-CONNECTION-STATUS-"
    IDENTIFICATION_STATUS = "-IDENTIFICATION-STATUS-"
    QIE_STATUS = "-QIE-STATUS-"
    DSP_STATUS = "-DSP-STATUS-"
    PARAMETERS_ESTIMATION_STATUS = "-PARAMETERS-ESTIMATION-STATUS-"
    ERROR_CORRECTION_STATUS = "-ERROR-CORRECTION-STATUS-"
    PRIVACY_AMPLIFICATION_STATUS = "-PRIVACY-AMPLITICATION-STATUS-"
    ELEC_DATA_STATUS = "-ELEC-DATA-STATUS-"
    SHOT_DATA_STATUS = "-SHOT-DATA-STATUS-"
    ALICE_ADDRESS = "-ALICE-ADDRESS-"
    FRAME_UUID = "-FRAME-UUID-"
    CONFIGURATION_QI_NUM_SYMBOLS = "-CONFIGURATION-QI-NUM-SYMBOLS-"
    CONFIGURATION_QI_SHIFT = "-CONFIGURATION-QI-SHIFT-"
    CONFIGURATION_QI_SYMBOL_RATE = "-CONFIGURATION-QI-SYMBOL-RATE-"
    CONFIGURATION_QI_ROLL_OFF = "-CONFIGURATION-QI-ROLL-OFF"
    CONFIGURATION_QI_MODULATION = "-CONFIGURATION-QI-MODULATION-"
    CONFIGURATION_QI_MODULATION_SIZE = "-CONFIGURATION-QI-MODULATION-SIZE-"
    CONFIGURATION_ZC_ROOT = "-CONFIGURATION-ZC-ROOT-"
    CONFIGURATION_ZC_LENGTH = "-CONFIGURATION-ZC-LENGTH-"
    CONFIGURATION_ZC_RATE = "-CONFIGURATION-ZC-RATE-"
    CONFIGURATION_PILOTS_FREQUENCIES = "-CONFIGURATION-PILOTS-FREQUENCIES-"
    CONFIGURATION_DSP_TONE_CUTOFF = "-CONFIGURATION-DSP-TONE-CUTOFF-"
    CONFIGURATION_DSP_SUBFRAMES_SIZE = "-CONFIGURATION-DSP-SUBFRAMES-SIZE-"
    CONFIGURATION_DSP_ABORT_CLOCK_RECOVERY = "-CONFIGURATION-DSP-ABORT-CLOCK-RECOVERY-"
    CONFIGURATION_DSP_ALICE_DAC_RATE = "-CONFIGURATION-DSP-ALICE-DAC-RATE-"
    CONFIGURATION_DSP_EXCLUSION_ZONE = "-CONFIGURATION-DSP-EXCLUSION-ZONE-"
    CONFIGURATION_DSP_PHASE_FILTERING = "-CONFIGURATION-DSP-PHASE-FILTERING-"
    PE_ETA = "-PE-ETA-"
    PE_PHOTON_NUMBER = "-PE-PHOTON-NUMBER-"
    PE_ETA_T = "-PE-ETA-T-"
    PE_T = "-PE-T-"
    PE_DISTANCE = "-PE-DISTANCE-"
    PE_SHOT = "-PE-SHOT-"
    PE_VEL = "-PE-VEL-"
    PE_EXCESS_NOISE_BOB = "-PE-EXCESS-NOISE-BOB-"
    PE_EXCESS_NOISE_ALICE = "-PE-EXCESS-NOISE-ALICE-"
    PE_SKR = "-PE-SKR-"
    LAST_EXPORT = "-LAST-EXPORT-"
    LOGGER = "-LOGGER-"
    EXPORT_DIRECTORY = "-EXPORT-DIRECTORY"


class QOSSTGUIInput(QOSSTGUIContent):
    """
    Enumeration of inputs (i.e. text input, checkboxes and selects) in the GUI.
    """

    CONFIGURATION_PATH = "-CONFIGURATION-PATH-"
    AUTO_EXPORT_DATA = "-AUTO-EXPORT-DATA-"
    SELECT_PLOT_STYLE = "-SELECT-PLOPT-STYLE-"
