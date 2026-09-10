# qosst-bob - Bob module of the Quantum Open Software for Secure Transmissions.
# Copyright (C) 2021-2026 Yoann Piétri
# Copyright (C) 2021-2024 Valentina Marulanda Acosta
# Copyright (C) 2021-2024 Matteo Schiavon
# Copyright (C) 2021-2026 Thomas Liege

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
Base code for the DSP. Implements the common special DSP for electronic and shot noise.
"""

import logging
from typing import Tuple, List, Optional

import numpy as np
from scipy.signal import oaconvolve

from qosst_core.schema.detection import SINGLE_POLARISATION_RF_HETERODYNE
from qosst_core.comm.filters import root_raised_cosine_filter
from qosst_core.dsp.base import BaseDSP

logger = logging.getLogger(__name__)


def _subsample(data: np.ndarray, ratio: float, position: str) -> np.ndarray:
    """
    Extract a contiguous subsample of an array, the size of which is a
    ratio of the original array length.

    The position string describes from where the samples should be taken:
    * 'h' or 'head' for the beginning of the data.
    * 'm' or 'middle' for the middle of the data.
    * 't' or 'tail' for the end of the data.

    Args:
        data (np.ndarray): array from which to extract the subsample
        ratio (float): ratio of sizes between the subsample and original array
        position (str): 'head', 'middle' or 'tail'
    """
    n = len(data)

    if position in ["m", "middle"]:
        index_from = int(n / 2 * (1 - ratio))
        index_to = int(n / 2 * (1 + ratio))
    elif position in ["h", "head"]:
        index_from = 0
        index_to = int(n * ratio)
    elif position in ["t", "tail"]:
        index_from = int(n * (1 - ratio))
        index_to = n
    else:
        raise ValueError(
            f"position must be one of 'head', 'middle' or 'tail' (got '{ position }')"
        )
    return data[index_from:index_to]


class DSPWithSpecial(BaseDSP):
    special_adc_rate: float  #: Equivalent ADC rate.
    special_frequency_shift: float  #: Actual frequency shift.

    def _run_special_dsp(
        self,
        electronic_noise_data: List[np.ndarray],
        electronic_shot_noise_data: List[np.ndarray],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        logger.info("Starting DSP on electronic noise and electronic and shot noise.")

        if self.schema != SINGLE_POLARISATION_RF_HETERODYNE:
            logger.critical(
                "General DSP can be only be used with detection scheme %s",
                str(SINGLE_POLARISATION_RF_HETERODYNE),
            )
            return None

        # We can then use electronic_noise_data[0] and electronic_shot_noise_data[0]
        # as the data.

        sps = self.special_adc_rate / self.symbol_rate

        # For efficiency reasons, the DSP on this section is performed
        # on float32s/complex64s.
        elec_noise_data = _subsample(
            electronic_noise_data[0], self.elec_noise_estimation_ratio, "tail"
        )
        elec_noise_data = elec_noise_data.astype(np.complex64)
        n_elec_noise_data = len(elec_noise_data)

        elec_shot_noise_data = _subsample(
            electronic_shot_noise_data[0], self.elec_shot_noise_estimation_ratio, "tail"
        )
        elec_shot_noise_data = elec_shot_noise_data.astype(np.complex64)
        n_elec_shot_noise_data = len(elec_shot_noise_data)

        # Precompute the filter and complex exponential for shifting.
        _, rrc_filter = root_raised_cosine_filter(
            int(10 * sps + 2),
            self.roll_off,
            1 / self.symbol_rate,
            self.special_adc_rate,
        )
        rrc_filter = rrc_filter[1:].astype("f")

        n_shift = max(n_elec_noise_data, n_elec_shot_noise_data)
        shift = np.exp(
            -1j
            * 2
            * np.pi
            * np.arange(n_shift)
            * self.special_frequency_shift
            / self.special_adc_rate
        ).astype(np.complex64)

        logger.info("Starting DSP on elec noise.")
        elec_noise_bb = elec_noise_data * shift[:n_elec_noise_data]
        elec_noise_filtered = (
            1 / np.sqrt(sps) * oaconvolve(elec_noise_bb, rrc_filter, "same")
        )
        logger.info("Starting DSP on elec+shot noise.")

        elec_shot_noise_bb = elec_shot_noise_data * shift[:n_elec_shot_noise_data]
        elec_shot_noise_filtered = (
            1 / np.sqrt(sps) * oaconvolve(elec_shot_noise_bb, rrc_filter, "same")
        )
        logger.info("DSP on elec and elec+shot noise finished.")

        return elec_noise_filtered, elec_shot_noise_filtered
