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
General implementation of the DSP with some optimizations.
"""

import logging
from typing import List, Optional
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import oaconvolve

from qosst_core.schema.detection import SINGLE_POLARISATION_RF_HETERODYNE
from qosst_core.comm.filters import root_raised_cosine_filter
from qosst_bob.dsp.synchro import synchronize
from qosst_bob.dsp.pilots import (
    recover_tone,
    find_one_pilot,
    find_two_pilots,
    correct_noise,
)
from qosst_bob.dsp.resample import (
    _downsample_float,
    _best_sampling_point_float,
)

from qosst_bob.dsp.dsp.base import DSPWithSpecial

logger = logging.getLogger(__name__)


class GeneralDSP(DSPWithSpecial):
    """
    General DSP.

    The steps are the following:
        - Find an approximative start of the Zadoff-Chu sequence
        - Find the pilots
        - Correct clock difference
        - Find the pilots again with the good clock
        - Estimate the beat frequency
        - Find the Zadoff-Chu sequence
        - Estimate the beat frequency (per subframe)
        - Find one pilot (per subframe)
        - Unshift the quantum signal (per subframe)
        - Apply matched RRC filter (per subframe)
        - Downsample (per subframe)
        - Correct relative phase noise (per subframe)

    The output has still a global phase difference.
    """

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def _run_dsp(
        self,
        data: np.ndarray,
        electronic_noise_data: Optional[List[np.ndarray]] = None,
        electronic_shot_noise_data: Optional[List[np.ndarray]] = None,
    ) -> Optional[List[np.ndarray]]:
        logger.info("Starting General DSP")

        if self.schema != SINGLE_POLARISATION_RF_HETERODYNE:
            logger.critical(
                "General DSP can be only be used with detection scheme %s",
                str(SINGLE_POLARISATION_RF_HETERODYNE),
            )
            return None

        # For single polarization and RF heterodyne, there is only one BHD detector, using the data from the first channel.
        data = data[0]

        # Find pilot frequency
        if self.num_pilots < 2:
            logger.error(
                "General dsp requires two pilots and only one was passed... Aborting",
            )
            return None

        # Find pilot frequency
        if self.num_pilots > 2:
            logger.warning(
                "More than 2 pilots were given but only two are necessary for recovery with unshared clock and unshared LO. Taking the two first pilots (%.2f MHz, %.2f MHz)",
                self.pilots_frequencies[0] * 1e-6,
                self.pilots_frequencies[1] * 1e-6,
            )

        f_pilot_1 = self.pilots_frequencies[0]
        f_pilot_2 = self.pilots_frequencies[1]

        # Create the synchronization sequence object
        synchro_obj = self.synchronization_cls(
            root=self.zc_root,
            length=self.zc_length,
            nbits=self.mls_nbits,
        )

        # Use the base DAC rate if the sample rate of the ZC sequence has not been
        # provided.
        if synchro_rate == 0:
            synchro_rate = self.dac_rate
        sps_approx = int(self.adc_rate / synchro_rate)

        # A first approximate search of the start of the ZC sequence, based on the
        # signal envelope.
        uniform_filter_length = int(synchro_obj.length * sps_approx)
        envelope = uniform_filter1d(np.abs(data), uniform_filter_length)
        approx_synchro = int(np.argmax(envelope) - uniform_filter_length / 2)

        # The pilot frequencies are estimated on a large sample
        # (typically 10M points) taken after the ZC sequence.
        pilot_start_point = approx_synchro + 2 * self.zc_length * sps_approx
        data_pilots = data[
            pilot_start_point : pilot_start_point + self.num_samples_pilot_search
        ]
        f_pilot_real_1, f_pilot_real_2 = find_two_pilots(
            data_pilots, self.adc_rate, excl=self.pilots_exclusion_zones
        )
        logger.info(
            "Pilots found at %f MHz and %f MHz",
            f_pilot_real_1 * 1e-6,
            f_pilot_real_2 * 1e-6,
        )

        # Measure the clock difference
        delta_f = (f_pilot_real_2 - f_pilot_real_1) / (f_pilot_2 - f_pilot_1)
        logger.info(
            "Tone difference : %.6f (expected value : %.2f)",
            (f_pilot_real_2 - f_pilot_real_1) * 1e-6,
            (f_pilot_2 - f_pilot_1) * 1e-6,
        )
        logger.info("Difference of clock is estimated at %.6f", delta_f)

        if (
            self.abort_clock_recovery != 0
            and np.abs(1 - delta_f) > self.abort_clock_recovery
        ):
            logger.warning(
                "Clock recovery algorithm aborted due to too high mismatch (%f > %f). Taking adc_rate as real adc_value.",
                np.abs(1 - delta_f),
                self.abort_clock_recovery,
            )
            equi_adc_rate = self.adc_rate
        else:
            logger.debug("Clock mismatch was accepted.")
            equi_adc_rate = self.adc_rate / delta_f

        if self.debug:
            self.debug_object["equi_adc_rate"] = equi_adc_rate
            self.debug_object["delta_frequency_pilots"] = delta_f

        logger.info("Equivalent ADC rate is %.6f MHz", equi_adc_rate * 1e-6)

        sps = equi_adc_rate / self.symbol_rate

        logger.info("Equivalent SPS is %.6f", sps)

        # Find again the real values.
        f_pilot_real_1, f_pilot_real_2 = find_two_pilots(
            data, equi_adc_rate, excl=self.pilots_exclusion_zones
        )
        logger.info(
            "Pilots found at %f MHz and %f MHz",
            f_pilot_real_1 * 1e-6,
            f_pilot_real_2 * 1e-6,
        )
        logger.info(
            "Tone difference : %.6f (expected value : %.2f)",
            (f_pilot_real_2 - f_pilot_real_1) * 1e-6,
            (f_pilot_2 - f_pilot_1) * 1e-6,
        )
        f_beat = f_pilot_real_1 - f_pilot_1

        if self.debug:
            self.debug_object["real_pilot_frequencies"] = [
                f_pilot_real_1,
                f_pilot_real_2,
            ]
            self.debug_object["beat_frequency"] = f_beat

        begin_synchro, end_synchro = synchronize(
            data
            * np.exp(-1j * 2 * np.pi * np.arange(len(data)) * f_beat / equi_adc_rate),
            synchro_obj,
            resample=equi_adc_rate / synchro_rate,
        )

        # Now that we have an estimation of the beginning of the synchronization sequence
        # let's reestimate f_beat more properly.
        len_synchro = np.ceil(
            synchro_obj.length * equi_adc_rate / self.dac_rate
        ).astype(int)
        f_pilot_real_1, f_pilot_real_2 = find_two_pilots(
            data[
                end_synchro
                + len_synchro : end_synchro
                + len_synchro
                + self.num_samples_fbeat_estimation
            ],
            equi_adc_rate,
            excl=self.pilots_exclusion_zones,
        )
        f_beat = f_pilot_real_1 - f_pilot_1

        begin_synchro, end_synchro = synchronize(
            data
            * np.exp(-1j * 2 * np.pi * np.arange(len(data)) * f_beat / equi_adc_rate),
            synchro_obj,
            resample=equi_adc_rate / synchro_rate,
        )

        begin_data = end_synchro
        end_data = int(
            begin_data + self.num_symbols * np.ceil(sps + 1)
        )  # We take a bit more of what is needed to be sure to have all symbols
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
            end_subframe = np.ceil(self.subframe_length * (sps + 1) - 0.5).astype(
                int
            )  # Take enough samples to have subframe_length symbos

        if self.debug:
            self.debug_object["tones"] = []
            self.debug_object["uncorrected_data"] = []

        result = []
        max_t0 = -1
        frequency_shift_mean = 0.0
        num_symbols_recovered = 0
        num_subframes = 0
        while num_symbols_recovered < self.num_symbols:
            subframe_data = useful_data[begin_subframe:end_subframe]

            # Find beat frequency
            f_pilot_real_1 = find_one_pilot(
                subframe_data, equi_adc_rate, excl=self.pilots_exclusion_zones
            )
            logger.info("Subframe pilot found at %f MHz", f_pilot_real_1 * 1e-6)

            f_beat = f_pilot_real_1 - f_pilot_1

            tone_data = recover_tone(
                subframe_data,
                f_pilot_real_1,
                equi_adc_rate,
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
                / equi_adc_rate
            )

            frequency_shift_mean += self.frequency_shift + f_beat

            _, filtre = root_raised_cosine_filter(
                int(10 * sps + 2),
                self.roll_off,
                1 / self.symbol_rate,
                equi_adc_rate,
            )

            subframe_data = (
                1 / np.sqrt(sps) * oaconvolve(subframe_data, filtre[1:], "same")
            )

            max_t = _best_sampling_point_float(subframe_data, sps)
            if max_t0 == -1:
                max_t0 = max_t
            subframe_data = _downsample_float(subframe_data, max_t, sps)[
                : self.subframe_length
            ]

            logger.info("Collecting %i symbols in the frame", len(subframe_data))

            last_indice = (
                begin_subframe
                + np.ceil(
                    max_t
                    + sps
                    * np.arange(
                        np.floor((len(data) - 0.5 - max_t) / sps).astype(int) + 1
                    )
                    - 0.5
                ).astype(int)[: self.subframe_length][-1]
            )

            if self.debug:
                self.debug_object["uncorrected_data"].append(subframe_data)

            # Correct phase noise
            subframe_data = correct_noise(
                subframe_data,
                max_t,
                sps,
                tone_data,
                f_pilot_real_1,
                equi_adc_rate,
                filter_size=self.pilot_phase_filtering_size,
            )
            result.append(subframe_data)
            begin_subframe = np.ceil(last_indice + sps / 2 - 0.5).astype(int)

            if process_subframes:
                end_subframe = np.ceil(
                    begin_subframe + self.subframe_length * (sps + 1) - 0.5
                ).astype(int)

            num_symbols_recovered += len(subframe_data)

            num_subframes += 1

        # Setting parameters for special DSP.
        self.special_adc_rate = equi_adc_rate
        self.special_frequency_shift = frequency_shift_mean / num_subframes
        self.special_parameters_set = True

        return result
