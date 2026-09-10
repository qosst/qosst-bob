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
General implementation of the DSP.
"""

import logging
from typing import List, Optional
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import oaconvolve, firwin

from qosst_core.comm.filters import root_raised_cosine_filter
from qosst_core.schema.detection import SINGLE_POLARISATION_RF_HETERODYNE
from qosst_bob.dsp.phase_estimation import ClassicalPhaseEstimator
from qosst_bob.dsp.synchro import synchronize
from qosst_bob.dsp.pilots import (
    find_one_pilot,
    find_two_pilots,
)
from qosst_bob.dsp.resample import upsample
from qosst_bob.dsp.dsp.base import DSPWithSpecial

logger = logging.getLogger(__name__)


class GeneralDirectTrackingDSP(DSPWithSpecial):
    """
    A less computationally-intensive variant of general DSP.

    It cancels the frequency shift induced by beating, and the phase noise,
    by directly recovering the phase of the pilot tone. This does not require
    any tracking of the pilot frequency (apart from a broad estimation of the
    frequency band in which it belongs).

    The steps are the following:
        - Find an approximation of the start of the Zadoff-Chu sequence
        - Extract the frequency of the pilot(s)
        - Find the Zadoff-Chu sequence
        - Extract the main pilot (per subframe)
        - Filter the phase of the pilot (per subframe)
        - Shift the data to base-band (per subframe)
        - Apply matched RRC filter (per subframe)
        - Find the best sampling point( per subframe)
        - Downsample (per subframe)

    The output still has a global phase difference.
    """

    # pylint: disable=too-many-statements, too-many-locals, too-many-branches
    def _run_dsp(
        self,
        data: List[np.ndarray],
        electronic_noise_data: Optional[List[np.ndarray]] = None,
        electronic_shot_noise_data: Optional[List[np.ndarray]] = None,
    ) -> Optional[List[np.ndarray]]:

        logger.info("Starting General DSP with direct pilot tracking")

        if self.schema != SINGLE_POLARISATION_RF_HETERODYNE:
            logger.critical(
                "General DSP can be only be used with detection scheme %s",
                str(SINGLE_POLARISATION_RF_HETERODYNE),
            )
            return None

        # For single polarization and RF heterodyne, there is only one BHD detector, using the data from the first channel.
        data = data[0]

        # Separate shot noise if switching time is given
        if self.switching_time:
            end_electronic_shot_noise = int(self.switching_time * self.adc_rate)
            electronic_shot_noise_data = data[:end_electronic_shot_noise]
            data = data[end_electronic_shot_noise:]

        # Find pilot frequency
        if self.num_pilots < 1:
            logger.error("At least one pilot is required... Aborting")
            return None, None, None

        if self.num_pilots > 2:
            logger.warning(
                "More than 2 pilots were given but only two are necessary for recovery with unshared clock and unshared LO. Taking the two first pilots (%.2f MHz, %.2f MHz)",
                self.pilots_frequencies[0] * 1e-6,
                self.pilots_frequencies[1] * 1e-6,
            )

        f_pilot_1 = self.pilots_frequencies[0]

        # Convert the data to float32
        data = data.astype("f")

        # Create the synchronization sequence object
        synchro_obj = self.synchronization_cls(
            root=self.zc_root,
            length=self.zc_length,
            nbits=self.mls_nbits,
        )

        # Use the base DAC rate if the sample rate of the synchronization sequence has not been
        # provided.
        if synchro_rate == 0:
            synchro_rate = self.dac_rate
        synchro_oversampling = int(self.adc_rate / synchro_rate)

        logger.info(
            "Computing envelope for approximate synchronization sequence search"
        )
        envelope = np.abs(data[::synchro_oversampling])
        envelope = uniform_filter1d(envelope, synchro_obj.length)
        preamble_synchro_start = (
            np.argmax(envelope) - synchro_obj.length // 2
        ) * synchro_oversampling

        # The pilot frequencies are estimated on a large sample
        # (typically 10M points) taken after the synchronization sequence.
        pilot_start_point = (
            preamble_synchro_start + 2 * self.zc_length * synchro_oversampling
        )
        data_pilots = data[
            pilot_start_point : pilot_start_point + self.num_samples_pilot_search
        ]

        if self.num_pilots == 2:
            f_pilot_2 = self.pilots_frequencies[1]
            logger.info("Searching for pilots")
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
                self.debug_object["real_pilot_frequencies"] = [
                    f_pilot_real_1,
                    f_pilot_real_2,
                ]
        else:
            # Only one pilot found, so we assume that the ADC rate perfectly
            # matches the specified value.
            equi_adc_rate = self.adc_rate

            logger.info("Searching for the pilot")
            f_pilot_real_1 = find_one_pilot(
                data_pilots, equi_adc_rate, excl=self.pilots_exclusion_zones
            )
            if self.debug:
                self.debug_object["real_pilot_frequencies"] = [f_pilot_real_1]

        # Correct estimates with true ADC rate (if estimated).
        f_pilot_1 *= equi_adc_rate / self.adc_rate
        sps = equi_adc_rate / self.symbol_rate
        f_beat = f_pilot_real_1 - f_pilot_1

        logger.info("Equivalent ADC rate is %.6f MHz", equi_adc_rate * 1e-6)
        logger.info("Equivalent SPS is %.6f", sps)

        if self.debug:
            self.debug_object["beat_frequency"] = f_beat

        logger.info("Searching for start of the synchronization sequence")
        synchro_search_start = max(
            preamble_synchro_start - 4 * synchro_obj.length * synchro_oversampling, 0
        )
        synchro_search_end = (
            synchro_search_start + 8 * synchro_obj.length * synchro_oversampling
        )
        data_synchro = data[synchro_search_start:synchro_search_end]
        shift = np.exp(
            -1j * 2 * np.pi * np.arange(len(data_synchro)) * f_beat / equi_adc_rate
        )
        begin_synchro, end_synchro = synchronize(
            data_synchro * shift, synchro_obj, resample=equi_adc_rate / synchro_rate
        )
        begin_synchro += synchro_search_start
        end_synchro += synchro_search_start

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
            self.debug_object["tones"] = []
            self.debug_object["uncorrected_data"] = []

        begin_subframe = 0
        end_subframe = int(np.ceil(self.subframe_length * (sps + 1) - 0.5))
        result = []
        num_symbols_recovered = 0

        # Number of samples to include from the previous subframe to account for
        # the boundary conditions of the filters.
        num_samples_previous_subframe = max(
            self.pilot_phase_filtering_size, self.fir_size
        )

        # Pre-compute the RRC filter.
        _, rrc_filter = root_raised_cosine_filter(
            int(10 * sps + 2),
            self.roll_off,
            1 / self.symbol_rate,
            equi_adc_rate,
        )
        rrc_filter = (rrc_filter[1:] / np.sqrt(sps)).astype(np.complex64)

        # Pre-compute the complex exponential for shifting.
        shift_size = self.subframe_length * (sps + 1) + num_samples_previous_subframe
        shift_up = np.exp(
            1j
            * 2
            * np.pi
            * np.arange(shift_size)
            * (f_pilot_1 - self.frequency_shift)
            / equi_adc_rate
        ).astype(np.complex64)

        # Pre-compute the filter extracting the pilot tone.
        pilot_bp_filter = (
            firwin(self.fir_size, self.tone_filtering_cutoff / equi_adc_rate)
            * np.exp(
                1j
                * 2
                * np.pi
                * np.arange(self.fir_size)
                * f_pilot_real_1
                / equi_adc_rate
            )
        ).astype(np.complex64)

        # Correct the phase noise on the whole frame before starting to extract symbols,
        # to avoid the boundary effects of the filters on the subframes.
        logger.info("Recovering first pilot tone")
        pilot_data = oaconvolve(useful_data, pilot_bp_filter, mode="same")
        shot_noise_data = oaconvolve(
            electronic_shot_noise_data, pilot_bp_filter, mode="same"
        )

        logger.info("Correcting phase noise on the whole frame")
        if self.phase_estimator_cls == ClassicalPhaseEstimator:
            phase_estimator = self.phase_estimator_cls(
                pilot_phase_filtering_size=self.pilot_phase_filtering_size,
                pilot_frequency_filtering_size=self.pilot_frequency_filtering_size,
            )
            phase_noise = phase_estimator.estimate_phase(pilot_data)
        else:
            phase_estimator = self.phase_estimator_cls(self.linewidth, equi_adc_rate)
            phase_noise = phase_estimator.estimate_phase(pilot_data, shot_noise_data)

        clean_pilot = np.exp(-1j * phase_noise).astype(np.complex64)

        logger.info("Cancelling phase noise")
        useful_data = useful_data.astype(np.complex64) * clean_pilot

        timing_estimator = self.timing_estimator_cls(
            sps, self.subframe_length, self.symbol_timing_oversampling
        )

        while num_symbols_recovered < self.num_symbols:
            # Include more samples to account for the boundary condition of filters.
            begin_extended_subframe = max(
                begin_subframe - num_samples_previous_subframe, 0
            )
            subframe_data = useful_data[begin_extended_subframe:end_subframe].astype(
                np.complex64
            )

            logger.info("Shifting quantum data to baseband")
            subframe_data *= shift_up[: len(subframe_data)]

            logger.info("Applying RRC filter")
            subframe_data = oaconvolve(subframe_data, rrc_filter, "same")

            # Ignore the extra samples at the beginning of the frame
            subframe_data = subframe_data[begin_subframe - begin_extended_subframe :]

            logger.info("Finding best decision point")
            if self.symbol_timing_oversampling != 1:
                subframe_data = upsample(
                    subframe_data, self.symbol_timing_oversampling, 2
                )

            best_grid = timing_estimator.estimate_timing(subframe_data)

            logger.info("Downsampling")
            subframe_data = subframe_data[best_grid]
            last_index = (
                begin_subframe + best_grid[-1] / self.symbol_timing_oversampling
            )

            logger.info("Collecting %i symbols in the frame", len(subframe_data))
            chunk_length = len(subframe_data) // self.subframe_subdivision
            for i in range(self.subframe_subdivision):
                start = i * chunk_length
                if i != self.subframe_subdivision - 1:
                    result.append(subframe_data[start : start + chunk_length])
                else:
                    result.append(subframe_data[start:])
            num_symbols_recovered += len(subframe_data)

            begin_subframe = int(last_index + sps / 2 - 0.5)
            end_subframe = int(begin_subframe + self.subframe_length * (sps + 1) - 0.5)

            if self.debug:
                self.debug_object["uncorrected_data"].append(np.array([]))

        # Set parameters for special DSP
        self.special_frequency_shift = self.frequency_shift + f_beat
        self.special_adc_rate = equi_adc_rate
        self.special_parameters_set = True
        return result
