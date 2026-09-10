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
Specific DSP for the shared clock and unshared LO case. Is untested.
"""

import logging
from typing import List, Optional

import numpy as np
from scipy.signal import oaconvolve

from qosst_core.schema.detection import (
    SINGLE_POLARISATION_RF_HETERODYNE,
)
from qosst_core.comm.filters import root_raised_cosine_filter

from qosst_bob.dsp.synchro import synchronize
from qosst_bob.dsp.pilots import (
    recover_tone,
    find_one_pilot,
    correct_noise,
)
from qosst_bob.dsp.resample import (
    downsample,
    best_sampling_point,
)
from qosst_bob.dsp.dsp.base import DSPWithSpecial

logger = logging.getLogger(__name__)


class SharedClockUnsharedLODSP(DSPWithSpecial):
    """
    DSP in the case of a shared clock and an unshared local oscillator.

    This simplifies the DSP, since there is no clock difference.

    The procedure is the following:
        - Estimation of f_beat (to find the Zadoff-Chu sequence)
        - Recovery of the Zadoff-Chu sequence
        - Recovery of the pilot (per subframe)
        - Estimation of f_beat (per subframe)
        - Unshift signal (per subframe)
        - Apply match filter (per subframe)
        - Downsample (per subframe)
        - Correct relative phase noise (per subframe)

    The output has still a global phase noise.
    """

    unsafe = True

    # pylint: disable=too-many-locals, too-many-statements
    def _run_dsp(
        self,
        data: np.ndarray,
        electronic_noise_data: Optional[List[np.ndarray]] = None,
        electronic_shot_noise_data: Optional[List[np.ndarray]] = None,
    ) -> Optional[List[np.ndarray]]:
        logger.info("Starting DSP with shared clock and unshared local oscillator.")
        logger.warning(
            "This is specialized DSP that was less tested than the general DSP."
        )

        if self.schema != SINGLE_POLARISATION_RF_HETERODYNE:
            logging.critical(
                "This specialized DSP was not intended for another schema than SINGLE_POLARISATION_RF_HETERODYNE. Aborting."
            )
            return None

        # For single polarization and RF heterodyne, there is only one BHD detector, using the data from the first channel.
        data = data[0]

        # Find pilot frequency
        if self.num_pilots > 1:
            logger.warning(
                "More than 1 pilot was given but only one is necessary for recovery with shared clock and unshared LO. Taking the first pilot (%.2f MHz)",
                self.pilots_frequencies[0] * 1e-6,
            )
        f_pilot = self.pilots_frequencies[0]

        sps = self.adc_rate / self.symbol_rate

        # First find the pilot, shift and find the synchronization sequence
        f_pilot_real = find_one_pilot(
            data, self.adc_rate, excl=self.pilots_exclusion_zones
        )
        f_beat = f_pilot_real - f_pilot

        if self.debug:
            self.debug_object["real_pilot_frequencies"] = [f_pilot_real]
            self.debug_object["beat_frequency"] = f_beat

        # Create the synchronization sequence object
        synchro_obj = self.synchronization_cls(
            root=self.zc_root,
            length=self.zc_length,
            nbits=self.mls_nbits,
        )

        if synchro_rate == 0:
            synchro_rate = self.dac_rate
        begin_synchro, end_synchro = synchronize(
            data
            * np.exp(-1j * 2 * np.pi * np.arange(len(data)) * f_beat / self.adc_rate),
            synchro_obj,
            resample=self.adc_rate / synchro_rate,
        )

        begin_data = end_synchro
        end_data = int(begin_data + self.num_symbols * sps)
        useful_data = data[begin_data:end_data]

        if self.debug:
            self.debug_object["begin_synchro"] = begin_synchro
            self.debug_object["end_synchro"] = end_synchro
            self.debug_object["begin_data"] = begin_data
            self.debug_object["end_data"] = end_data

        begin_subframe = 0
        end_subframe = len(useful_data)

        process_subframes = self.subframe_length not in (0, self.num_symbols)

        if process_subframes:
            end_subframe = self.subframe_length

        if self.debug:
            self.debug_object["tones"] = []
            self.debug_object["uncorrected_data"] = []

        result = []
        f_shift_mean = 0.0
        num_subframes = 0
        while begin_subframe < len(useful_data):
            subframe_data = useful_data[begin_subframe:end_subframe]

            # Find beat frequency
            f_pilot_real = find_one_pilot(
                subframe_data, self.adc_rate, excl=self.pilots_exclusion_zones
            )
            f_beat = f_pilot_real - f_pilot

            tone_data = recover_tone(
                subframe_data,
                f_pilot_real,
                self.adc_rate,
                self.fir_size,
                cutoff=self.tone_filtering_cutoff,
            )

            if self.debug:
                self.debug_object["tones"].append(tone_data)

            # Now unshift signal taking the beat into account, apply RRC filter and downsample

            subframe_data = subframe_data * np.exp(
                -1j
                * 2
                * np.pi
                * np.arange(len(subframe_data))
                * (self.frequency_shift + f_beat)
                / self.adc_rate
            )

            f_shift_mean += self.frequency_shift + f_beat

            _, filtre = root_raised_cosine_filter(
                int(10 * sps + 2),
                self.roll_off,
                1 / self.symbol_rate,
                self.adc_rate,
            )

            subframe_data = (
                1 / np.sqrt(sps) * oaconvolve(subframe_data, filtre[1:], "same")
            )

            max_t = best_sampling_point(subframe_data, sps)

            subframe_data = downsample(subframe_data, max_t, sps)

            if self.debug:
                self.debug_object["uncorrected_data"].append(subframe_data)

            # Correct phase noise
            subframe_data = correct_noise(
                subframe_data,
                max_t,
                sps,
                tone_data,
                f_pilot,
                self.adc_rate,
                filter_size=self.pilot_phase_filtering_size,
            )

            result.append(subframe_data)
            begin_subframe = end_subframe

            if process_subframes:
                end_subframe = begin_subframe + self.subframe_length

            num_subframes += 1

        # Set special DSP parameters
        self.special_frequency_shift = f_shift_mean / num_subframes
        self.special_adc_rate = self.adc_rate
        self.special_parameters_set = True

        return result
