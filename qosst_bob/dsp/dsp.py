# qosst-bob - Bob module of the Quantum Open Software for Secure Transmissions.
# Copyright (C) 2021-2024 Yoann Piétri
# Copyright (C) 2021-2024 Valentina Marulanda Acosta
# Copyright (C) 2021-2024 Matteo Schiavon

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
Main module for the DSP algorithm.

Warning: the DSP _dsp_bob_shared_clock_shared_lo, _dsp_bob_shared_clock_unshared_lo and _dsp_bob_unshared_clock_shared_lo
are adapated versions of old DSP and might not work. There are untested.
"""
# pylint: disable=too-many-lines
import logging
from typing import Tuple, List, Optional
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d

from qosst_core.configuration import Configuration
from qosst_core.schema.detection import (
    DetectionSchema,
    SINGLE_POLARISATION_RF_HETERODYNE,
)
from qosst_core.comm.filters import root_raised_cosine_filter

from .zc import synchronisation_zc
from .pilots import (
    recover_tone,
    find_one_pilot,
    find_two_pilots,
    correct_noise,
    equivalent_adc_rate_one_pilot,
)
from .resample import (
    _best_sampling_point_int,
    _downsample_float,
    downsample,
    _best_sampling_point_float,
    best_sampling_point,
)

logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes
@dataclass
class DSPDebug:
    """
    Dataclass for debug information for the DSP.
    """

    begin_zadoff_chu: int = 0  #: Beginning of the Zadoff-Chu sequence.
    end_zadoff_chu: int = 0  #: End of the Zadoff-Chu sequence.
    begin_data: int = 0  #: Beginning of useful data.
    end_data: int = 0  #: End of useful data.
    tones: List[np.ndarray] = field(
        default_factory=list
    )  #: List of arrays containing filtered tone used for phase recovery.
    uncorrected_data: List[np.ndarray] = field(
        default_factory=list
    )  #: List of arrays containing data before phase correction.
    real_pilot_frequencies: List[float] = field(
        default_factory=list
    )  #: List of recovered pilot frequencies.
    beat_frequency: float = (
        0  #: Beat frequency between signal and LO laser in the LLO setup.
    )
    delta_frequency_pilots: float = 0  #: Difference of frequency between the two tones.
    equi_adc_rate: float = 0  #: Equivalent ADC rate in case the clock is not shared.

    def __str__(self) -> str:
        res = "DSP Debug :\n"
        res += f"Begin Zadoff-Chu : {self.begin_zadoff_chu}\n"
        res += f"End Zadoff-Chu : {self.end_zadoff_chu}\n"
        res += f"Begin data : {self.begin_data}\n"
        res += f"End data : {self.end_data}\n"
        res += f"Tones : {len(self.tones)} arrays\n"
        res += f"Uncorrected data : {len(self.uncorrected_data)} arrays\n"
        res += f"Real pilot frequencies : {','.join([str(freq*1e-6) for freq in self.real_pilot_frequencies])} MHz\n"
        res += f"Beat frequency : {self.beat_frequency*1e-6} MHz\n"
        res += f"Delta frequency pilots : {self.delta_frequency_pilots*1e-6} MHz\n"
        res += f"Equi ADC rate : {self.equi_adc_rate*1e-9} GSamples/s\n"
        return res


@dataclass
class SpecialDSPParams:
    """
    Dataclass for the parameters to give to the special DSP function for the elec and shot noise.
    """

    symbol_rate: float  #: Symbol rate in Symbols/s,.
    adc_rate: float  #: Symbol rate in Samples/s, recovered in case clock is not shared.
    roll_off: float  #: Roll off of the RRC filter.
    frequency_shift: float  #: Frequency shift of the data, recovered in case clock is not shared and/or LLO setup.
    schema: DetectionSchema  #: Detection schema to know how to interpret the data.

    def __str__(self) -> str:
        return f"Symbol rate = {self.symbol_rate*1e-6} MBaud, ADC Rate = {self.adc_rate*1e-9} GSamples/s, Roll Off = {self.roll_off}, Frequency shift = {self.frequency_shift*1e-6} MHz, Detection schema = {str(self.schema)}"


def dsp_bob(
    data: np.ndarray, config: Configuration
) -> Tuple[Optional[List[np.ndarray]], Optional[SpecialDSPParams], Optional[DSPDebug]]:
    """
    DSP function for Bob, given the data and the configuration.

    Args:
        data (np.ndarray): the data on which to apply the DSP.
        config (Configuration): the configuration object.

    Returns:
        Tuple[Optional[List[np.ndarray]], Optional[SpecialDSPParams], Optional[DSPDebug]]: array of symbols, SpecialDSPParams containing data to apply the exact same DSP to other data and DSPDebug containing debug information,.
    """
    assert config.frame is not None
    assert config.bob is not None
    assert config.clock is not None
    assert config.local_oscillator is not None
    return dsp_bob_params(
        data,
        config.frame.quantum.symbol_rate,
        config.bob.dsp.alice_dac_rate,
        config.bob.adc.rate,
        config.frame.quantum.num_symbols,
        config.frame.quantum.roll_off,
        config.frame.quantum.frequency_shift,
        config.frame.pilots.num_pilots,
        config.frame.pilots.frequencies,
        config.frame.zadoff_chu.length,
        config.frame.zadoff_chu.root,
        config.frame.zadoff_chu.rate,
        config.clock.sharing,
        config.local_oscillator.shared,
        config.bob.dsp.process_subframes,
        config.bob.dsp.subframes_size,
        config.bob.dsp.fir_size,
        config.bob.dsp.tone_filtering_cutoff,
        config.bob.dsp.abort_clock_recovery,
        config.bob.dsp.exclusion_zone_pilots,
        config.bob.dsp.pilot_phase_filtering_size,
        config.bob.dsp.num_samples_fbeat_estimation,
        config.bob.schema,
        config.bob.dsp.debug,
    )


# pylint: disable=too-many-arguments, too-many-locals
def dsp_bob_params(
    data: np.ndarray,
    symbol_rate: float,
    dac_rate: float,
    adc_rate: float,
    num_symbols: int,
    roll_off: float,
    frequency_shift: float,
    num_pilots: int,
    pilots_frequencies: np.ndarray,
    zc_length: int,
    zc_root: int,
    zc_rate: float,
    shared_clock: bool = False,
    shared_lo: bool = False,
    process_subframes: bool = False,
    subframe_length: int = 0,
    fir_size: int = 500,
    tone_filtering_cutoff: float = 10e6,
    abort_clock_recovery: float = 0,
    excl: Optional[List[Tuple[float, float]]] = None,
    pilot_phase_filtering_size: int = 0,
    num_samples_fbeat_estimation: int = 100000,
    schema: DetectionSchema = SINGLE_POLARISATION_RF_HETERODYNE,
    debug: bool = False,
) -> Tuple[Optional[List[np.ndarray]], Optional[SpecialDSPParams], Optional[DSPDebug]]:
    """
    Apply the DSP to the data given the DSP parameters.

    Args:
        data (np.ndarray): data on which to apply the DSP.
        symbol_rate (float): symbol rate in Symbols per second.
        dac_rate (float): DAC rate in Hz.
        adc_rate (float): ADC rate in Hz.
        num_symbols (int): number of sent symbols.
        roll_off (float): roll off value for the RRC filter
        frequency_shift (float): frequency shift of the quantum data in Hz.
        num_pilots (int): number of pilots.
        pilots_frequencies (np.ndarray): list of pilot frequencies, in Hz.
        zc_length (int): length of the Zadoff-Chu sequence.
        zc_root (int): root of the Zadoff-Chu sequence.
        zc_rate (float): rate of the Zadoff-Chu sequence.
        shared_clock (bool, optional): if the clock is shared between Alice and Bob. Defaults to False.
        shared_lo (bool, optional): if the local oscillator is shared between Alice and Bob. Defaults to False.
        process_subframes (bool, optional): if the data should be processed at subframes. Defaults to False.
        subframe_length (int, optional): if the previous parameter is True, the length, in samples, of the subframe. Defaults to 0.
        fir_size (int, optional): FIR size. Defaults to 500.
        tone_filtering_cutoff (float, optional): cutoff for the FIR filter for the pilot filtering, in Hz.
        abort_clock_recovery (float, optional): Maximal mismatch allowed by the clock recovery algorithm before aborting. If 0, the algorithm never aborts. Defaults to 0.
        excl (Optional[List[Tuple[float, float]]], optional): exclusion zones for the research of pilots (i.e. frequencies where we are sure the pilots are not), given as a list of tuples of float, each elements defining excluded segment (start frequency, stop frequency).
        pilot_phase_filtering_size (int, optional): Size of the uniform1d filter to apply to the phase of the recovered pilots for correction. Defaults to 0.
        num_samples_fbeat_estimation (int, optional): number of samples to estimate the beat frequency between the two lasers. Defaults to 100000.
        schema (DetectionSchema, optional): detection schema to use for the DSP. Defaults to qosst_core.schema.emission.SINGLE_POLARISATION_RF_HETERODYNE.
        debug (bool, optional): wether to return a debug dict. Defaults to False.

    Returns:
        Tuple[Optional[np.ndarray], Optional[SpecialDSPParams], Optional[DSPDebug]]: array of symbols, SpecialDSPParams containing data to apply the exact same DPS to other data and DSPDebug containing debug information,.
    """
    # First look at which DSP apply, at they are all very different,
    # we will make several functions

    if num_pilots == 0:
        logger.critical("DSP was called with 0 pilots. Aborting...")
        return None, None, None

    if shared_clock and shared_lo:
        return _dsp_bob_shared_clock_shared_lo(
            data,
            symbol_rate,
            dac_rate,
            adc_rate,
            num_symbols,
            roll_off,
            frequency_shift,
            num_pilots,
            pilots_frequencies,
            zc_length,
            zc_root,
            zc_rate,
            process_subframes,
            subframe_length,
            fir_size=fir_size,
            tone_filtering_cutoff=tone_filtering_cutoff,
            pilot_phase_filtering_size=pilot_phase_filtering_size,
            schema=schema,
            debug=debug,
        )
    if shared_clock and not shared_lo:
        return _dsp_bob_shared_clock_unshared_lo(
            data,
            symbol_rate,
            dac_rate,
            adc_rate,
            num_symbols,
            roll_off,
            frequency_shift,
            num_pilots,
            pilots_frequencies,
            zc_length,
            zc_root,
            zc_rate,
            process_subframes,
            subframe_length,
            fir_size=fir_size,
            tone_filtering_cutoff=tone_filtering_cutoff,
            excl=excl,
            pilot_phase_filtering_size=pilot_phase_filtering_size,
            schema=schema,
            debug=debug,
        )
    if not shared_clock and shared_lo:
        return _dsp_bob_unshared_clock_shared_lo(
            data,
            symbol_rate,
            dac_rate,
            adc_rate,
            num_symbols,
            roll_off,
            frequency_shift,
            num_pilots,
            pilots_frequencies,
            zc_length,
            zc_root,
            zc_rate,
            process_subframes,
            subframe_length,
            fir_size=fir_size,
            tone_filtering_cutoff=tone_filtering_cutoff,
            pilot_phase_filtering_size=pilot_phase_filtering_size,
            schema=schema,
            debug=debug,
        )
    return _dsp_bob_general(
        data,
        symbol_rate,
        dac_rate,
        adc_rate,
        num_symbols,
        roll_off,
        frequency_shift,
        num_pilots,
        pilots_frequencies,
        zc_length,
        zc_root,
        zc_rate,
        process_subframes,
        subframe_length,
        fir_size=fir_size,
        tone_filtering_cutoff=tone_filtering_cutoff,
        abort_clock_recovery=abort_clock_recovery,
        excl=excl,
        pilot_phase_filtering_size=pilot_phase_filtering_size,
        num_samples_fbeat_estimation=num_samples_fbeat_estimation,
        schema=schema,
        debug=debug,
    )


# pylint: disable=too-many-arguments, too-many-locals, too-many-statements, too-many-branches
def _dsp_bob_shared_clock_shared_lo(
    data: np.ndarray,
    symbol_rate: float,
    dac_rate: float,
    adc_rate: float,
    num_symbols: int,
    roll_off: float,
    frequency_shift: float,
    num_pilots: int,
    pilots_frequencies: np.ndarray,
    zc_length: int,
    zc_root: int,
    zc_rate: float,
    process_subframes: bool = False,
    subframe_length: int = 0,
    fir_size: int = 500,
    tone_filtering_cutoff: float = 10e6,
    pilot_phase_filtering_size: int = 0,
    schema: DetectionSchema = SINGLE_POLARISATION_RF_HETERODYNE,
    debug: bool = False,
) -> Tuple[Optional[List[np.ndarray]], Optional[SpecialDSPParams], Optional[DSPDebug]]:
    """
    DSP in the case of a shared clock and a shared local oscillator.

    This simplifies a lot the DSP, since there is no clock difference
    or beat frequency.

    The procedure is the following:
        - Recovery of the Zadoff-Chu sequence
        - Recovery of the pilot (per subframe)
        - Unshift signal (per subframe)
        - Apply match filter (per subframe)
        - Downsample (per subframe)
        - Correct relative phase noise (per subframe)

    The output has still a global phase noise.

    Args:
        data (np.ndarray): data received by Bob.
        symbol_rate (float): symbol rate for the quantum data, in Symbols per second.
        dac_rate (float): DAC rate, in Hz.
        adc_rate (float): ADC rate in Hz.
        num_symbols (int): number of symbols.
        roll_off (float): roll-off for the RRC filter.
        frequency_shift (float): frequnecy shift in Hz for the quantum data.
        num_pilots (int): number of pilots.
        pilots_frequencies (np.ndarray): list of frequencies of the pilots.
        zc_length (int): length of the Zadoff-Chu sequence.
        zc_root (int): root of the Zadoff-Chu sequence.
        zc_rate (float): rate of the Zadoff-Chu sequence.
        process_subframes (bool, optional): if True, data is processed as subframes. Defaults to False.
        subframe_length (int, optional): number of symbols to recover in each subframe. Defaults to 0.
        fir_size (int, optional): size for the FIR filters. Defaults to 500.
        tone_filtering_cutoff (float, optional): cutoff, in Hz, for the filter of the tone. Defaults to 10e6.
        pilot_phase_filtering_size (int, optional): size of the uniform1d filter to filter the phase correction. Defaults to 0.
        schema (DetectionSchema, optional): detection schema to use for the DSP. Defaults to qosst_core.schema.emission.SINGLE_POLARISATION_RF_HETERODYNE.
        debug (bool, optional): if True, a debug dict is returned. Defaults to False.

    Returns:
        Tuple[List[np.ndarray], SpecialDSPParams, Optional[DSPDebug]]: list of np.ndarray, each one corresponding to the recovered symbols for a subframe, SpecialDSPParams object to give to the special dsp, and DSPDebug object if debug was true.
    """
    logger.warning("Using untested DSP.")
    logger.info("Starting DSP with shared clock and shared local oscillator.")
    logger.warning("This is specialized DSP that was less tested than the general DSP.")

    if schema != SINGLE_POLARISATION_RF_HETERODYNE:
        logging.critical(
            "This specialized DSP was not intended for another schema than SINGLE_POLARISATION_RF_HETERODYNE. Aborting."
        )
        return None, None, None

    if debug:
        logger.info("Debug mode is on.")
        dsp_debug = DSPDebug()
    else:
        dsp_debug = None

    # Simplest case of DSP, everything is shared

    sps = int(adc_rate / symbol_rate)

    # Recover beginning of sequence
    if zc_rate == 0:
        zc_rate = dac_rate
    begin_zc, end_zc = synchronisation_zc(
        data, zc_root, zc_length, resample=adc_rate / zc_rate
    )
    begin_data = end_zc
    end_data = int(begin_data + num_symbols * sps)
    useful_data = data[begin_data:end_data]

    if dsp_debug:
        dsp_debug.begin_zadoff_chu = begin_zc
        dsp_debug.end_zadoff_chu = end_zc
        dsp_debug.begin_data = begin_data
        dsp_debug.end_data = end_data

    # Now recover the pilot tone
    # We only need one tone
    if num_pilots > 1:
        logger.warning(
            "More than 1 pilot was given but only one is necessary for recovery with shared clock and LO. Taking the first pilot (%.2f MHz)",
            pilots_frequencies[0] * 1e-6,
        )
    f_pilot = pilots_frequencies[0]

    if dsp_debug:
        dsp_debug.real_pilot_frequencies = [f_pilot]

    begin_subframe = 0
    end_subframe = len(useful_data)

    if process_subframes:
        end_subframe = subframe_length

    if dsp_debug:
        dsp_debug.tones = []
        dsp_debug.uncorrected_data = []

    result = []
    while begin_subframe < len(useful_data):
        subframe_data = useful_data[begin_subframe:end_subframe]
        tone_data = recover_tone(
            subframe_data, f_pilot, adc_rate, fir_size, cutoff=tone_filtering_cutoff
        )

        if dsp_debug:
            dsp_debug.tones.append(tone_data)

        # Now unshift signal, apply RRC filter and downsample

        subframe_data = subframe_data * np.exp(
            -1j * 2 * np.pi * np.arange(len(subframe_data)) * frequency_shift / adc_rate
        )

        _, filtre = root_raised_cosine_filter(
            int(10 * sps + 2),
            roll_off,
            1 / symbol_rate,
            adc_rate,
        )

        subframe_data = (
            1 / np.sqrt(sps) * np.convolve(subframe_data, filtre[1:], "same")
        )

        max_t = _best_sampling_point_int(subframe_data, sps)

        subframe_data = subframe_data[max_t::sps]

        if dsp_debug:
            dsp_debug.uncorrected_data.append(subframe_data)

        # Correct phase noise
        subframe_data = correct_noise(
            subframe_data,
            max_t,
            sps,
            tone_data,
            f_pilot,
            adc_rate,
            filter_size=pilot_phase_filtering_size,
        )

        result.append(subframe_data)
        begin_subframe = end_subframe

        if process_subframes:
            end_subframe = begin_subframe + subframe_length

    special_params = SpecialDSPParams(
        symbol_rate=symbol_rate,
        adc_rate=adc_rate,
        roll_off=roll_off,
        frequency_shift=frequency_shift,
        schema=schema,
    )

    return result, special_params, dsp_debug


# pylint: disable=too-many-arguments, too-many-locals, too-many-statements
def _dsp_bob_shared_clock_unshared_lo(
    data: np.ndarray,
    symbol_rate: float,
    dac_rate: float,
    adc_rate: float,
    num_symbols: int,
    roll_off: float,
    frequency_shift: float,
    num_pilots: int,
    pilots_frequencies: np.ndarray,
    zc_length: int,
    zc_root: int,
    zc_rate: float,
    process_subframes: bool = False,
    subframe_length: int = 0,
    fir_size: int = 500,
    tone_filtering_cutoff: float = 10e6,
    excl: Optional[List[Tuple[float, float]]] = None,
    pilot_phase_filtering_size: int = 0,
    schema: DetectionSchema = SINGLE_POLARISATION_RF_HETERODYNE,
    debug: bool = False,
) -> Tuple[Optional[List[np.ndarray]], Optional[SpecialDSPParams], Optional[DSPDebug]]:
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

    Args:
        data (np.ndarray): data measured by Bob.
        symbol_rate (float): symbol rate in Symbols per second.
        dac_rate (float): DAC rate, in Hz.
        adc_rate (float): ADC rate, in Hz.
        num_symbols (int): number of symbols.
        roll_off (float): roll off factor for the RRC filter.
        frequency_shift (float): frequency shift of the quantum symbols in Hz.
        num_pilots (int): number pilots.
        pilots_frequencies (np.ndarray): list of the frequencies of the pilots.
        zc_length (int): length of the Zadoff-Chu sequence.
        zc_root (int): root of the Zadoff-Chu sequence.
        zc_rate (float): shift, in Hz, of the Zadoff-Chu sequence.
        process_subframes (bool, optional): if True, process the data with subframes. Defaults to False.
        subframe_length (int, optional): number of symbols to recover in each subframe. Defaults to 0.
        fir_size (int, optional): size of the FIR filters.. Defaults to 500.
        tone_filtering_cutoff (float, optional): cutoff, in Hz, for the filtering of the pilots.. Defaults to 10e6.
        excl (Optional[List[Tuple[float, float]]], optional): exclusion zones for the research of pilots (i.e. frequencies where we are sure the pilots are not), given as a list of tuples of float, each elements defining excluded segment (start frequency, stop frequency). Defaults to None.
        pilot_phase_filtering_size (int, optional): size of the uniform1d filter to filter the phase correction. Defaults to 0.
        schema (DetectionSchema, optional): detection schema to use for the DSP. Defaults to qosst_core.schema.emission.SINGLE_POLARISATION_RF_HETERODYNE.
        debug (bool, optional): if True, the DSPDebug object is returned. Defaults to False.

    Returns:
        Tuple[List[np.ndarray], SpecialDSPParams, Optional[DSPDebug]]: list of np.ndarray, each one corresponding to the recovered symbols for a subframe, SpecialDSPParams object to give to the special dsp, and DSPDebug object if debug was true.
    """
    logger.warning("Using untested DSP.")
    logger.info("Starting DSP with shared clock and unshared local oscillator.")
    logger.warning("This is specialized DSP that was less tested than the general DSP.")

    if schema != SINGLE_POLARISATION_RF_HETERODYNE:
        logging.critical(
            "This specialized DSP was not intended for another schema than SINGLE_POLARISATION_RF_HETERODYNE. Aborting."
        )
        return None, None, None

    if debug:
        logger.info("Debug mode is on.")
        dsp_debug = DSPDebug()
    else:
        dsp_debug = None

    # Find pilot frequency
    if num_pilots > 1:
        logger.warning(
            "More than 1 pilot was given but only one is necessary for recovery with shared clock and unshared LO. Taking the first pilot (%.2f MHz)",
            pilots_frequencies[0] * 1e-6,
        )
    f_pilot = pilots_frequencies[0]

    sps = adc_rate / symbol_rate

    # First find the pilot, shift and find the synchronization sequence
    f_pilot_real = find_one_pilot(data, adc_rate, excl=excl)
    f_beat = f_pilot_real - f_pilot

    if dsp_debug:
        dsp_debug.real_pilot_frequencies = [f_pilot_real]
        dsp_debug.beat_frequency = f_beat

    if zc_rate == 0:
        zc_rate = dac_rate
    begin_zc, end_zc = synchronisation_zc(
        data * np.exp(-1j * 2 * np.pi * np.arange(len(data)) * f_beat / adc_rate),
        zc_root,
        zc_length,
        resample=adc_rate / zc_rate,
    )

    begin_data = end_zc
    end_data = int(begin_data + num_symbols * sps)
    useful_data = data[begin_data:end_data]

    if dsp_debug:
        dsp_debug.begin_zadoff_chu = begin_zc
        dsp_debug.end_zadoff_chu = end_zc
        dsp_debug.begin_data = begin_data
        dsp_debug.end_data = end_data

    begin_subframe = 0
    end_subframe = len(useful_data)

    if process_subframes:
        end_subframe = subframe_length

    if dsp_debug:
        dsp_debug.tones = []
        dsp_debug.uncorrected_data = []

    result = []
    f_shift_mean = 0.0
    num_subframes = 0
    while begin_subframe < len(useful_data):
        subframe_data = useful_data[begin_subframe:end_subframe]

        # Find beat frequency
        f_pilot_real = find_one_pilot(subframe_data, adc_rate, excl=excl)
        f_beat = f_pilot_real - f_pilot

        tone_data = recover_tone(
            subframe_data,
            f_pilot_real,
            adc_rate,
            fir_size,
            cutoff=tone_filtering_cutoff,
        )

        if dsp_debug:
            dsp_debug.tones.append(tone_data)

        # Now unshift signal taking the beat into account, apply RRC filter and downsample

        subframe_data = subframe_data * np.exp(
            -1j
            * 2
            * np.pi
            * np.arange(len(subframe_data))
            * (frequency_shift + f_beat)
            / adc_rate
        )

        f_shift_mean += frequency_shift + f_beat

        _, filtre = root_raised_cosine_filter(
            int(10 * sps + 2),
            roll_off,
            1 / symbol_rate,
            adc_rate,
        )

        subframe_data = (
            1 / np.sqrt(sps) * np.convolve(subframe_data, filtre[1:], "same")
        )

        max_t = best_sampling_point(subframe_data, sps)

        subframe_data = downsample(subframe_data, max_t, sps)

        if dsp_debug:
            dsp_debug.uncorrected_data.append(subframe_data)

        # Correct phase noise
        subframe_data = correct_noise(
            subframe_data,
            max_t,
            sps,
            tone_data,
            f_pilot,
            adc_rate,
            filter_size=pilot_phase_filtering_size,
        )

        result.append(subframe_data)
        begin_subframe = end_subframe

        if process_subframes:
            end_subframe = begin_subframe + subframe_length

        num_subframes += 1

    special_params = SpecialDSPParams(
        symbol_rate=symbol_rate,
        adc_rate=adc_rate,
        roll_off=roll_off,
        frequency_shift=f_shift_mean / num_subframes,
        schema=schema,
    )

    return result, special_params, dsp_debug


# pylint: disable=too-many-arguments, too-many-locals, too-many-statements
def _dsp_bob_unshared_clock_shared_lo(
    data: np.ndarray,
    symbol_rate: float,
    dac_rate: float,
    adc_rate: float,
    num_symbols: int,
    roll_off: float,
    frequency_shift: float,
    num_pilots: int,
    pilots_frequencies: np.ndarray,
    zc_length: int,
    zc_root: int,
    zc_rate: float,
    process_subframes: bool = False,
    subframe_length: int = 0,
    fir_size: int = 500,
    tone_filtering_cutoff: float = 10e6,
    pilot_phase_filtering_size: int = 0,
    schema: DetectionSchema = SINGLE_POLARISATION_RF_HETERODYNE,
    debug: bool = False,
) -> Tuple[Optional[List[np.ndarray]], Optional[SpecialDSPParams], Optional[DSPDebug]]:
    """
    DSP in the case of an unshared clock and a shared local oscillator.

    This simplifies the DSP, since there is no frequency beat.

    The procedure is the following:
        - Recovery of the Zadoff-Chu sequence
        - Recovery of the pilot (per subframe)
        - Recovery of the clock (per subframe)
        - Unshift signal (per subframe)
        - Apply match filter (per subframe)
        - Downsample (per subframe)
        - Correct relative phase noise (per subframe)

    Args:
        data (np.ndarray): data measured by Bob.
        symbol_rate (float): symbol rate in Symbols per second.
        dac_rate (float): DAC rate, in Hz.
        adc_rate (float): ADC rate, in Hz.
        num_symbols (int): number of symbols.
        roll_off (float): roll-off factor for the RRC filter.
        frequency_shift (float): frequency shift of the quantum symbols, in Hz.
        num_pilots (int): number of pilots.
        pilots_frequencies (np.ndarray): list of the frequencies of the pilots.
        zc_length (int): length of the Zadoff-Chu sequence.
        zc_root (int): root of the Zadoff-Chu sequence.
        zc_rate (float): rate of the Zadoff-Chu sequence.
        process_subframes (bool, optional): if True, process the data in subframes. Defaults to False.
        subframe_length (int, optional): number of symbols to recover in each subframe. Defaults to 0.
        fir_size (int, optional): size of the FIR filters. Defaults to 500.
        tone_filtering_cutoff (float, optional): cutoff, in Hz, for the filtering of the tone. Defaults to 10e6.
        pilot_phase_filtering_size (int, optional): size of the uniform1d filter to filter the phase correction. Defaults to 0.
        schema (DetectionSchema, optional): detection schema to use for the DSP. Defaults to qosst_core.schema.emission.SINGLE_POLARISATION_RF_HETERODYNE.
        debug (bool, optional): if True, the DSPDebug object is returned. Defaults to False.

    Returns:
        Tuple[List[np.ndarray], SpecialDSPParams, Optional[DSPDebug]]: list of np.ndarray, each one corresponding to the recovered symbols for a subframe, SpecialDSPParams object to give to the special dsp, and DSPDebug object if debug was true.
    """
    logger.warning("Using untested DSP.")
    logger.info("Starting DSP with unshared clock and shared local oscillator.")
    logger.warning("This is specialized DSP that was less tested than the general DSP.")

    if schema != SINGLE_POLARISATION_RF_HETERODYNE:
        logging.critical(
            "This specialized DSP was not intended for another schema than SINGLE_POLARISATION_RF_HETERODYNE. Aborting."
        )
        return None, None, None

    # Clock is not shared. Use one pilot tone to estimate clock difference

    if debug:
        logger.info("Debug mode is on.")
        dsp_debug = DSPDebug()
    else:
        dsp_debug = None

    sps = adc_rate / symbol_rate

    # Recover beginning of sequence
    if zc_rate == 0:
        zc_rate = dac_rate
    begin_zc, end_zc = synchronisation_zc(
        data, zc_root, zc_length, resample=adc_rate / zc_rate
    )
    begin_data = end_zc
    end_data = int(
        begin_data + num_symbols * np.ceil(sps + 1)
    )  # We take a bit more of what is needed to be sure to have all symbols
    useful_data = data[begin_data:end_data]

    if dsp_debug:
        dsp_debug.begin_zadoff_chu = begin_zc
        dsp_debug.end_zadoff_chu = end_zc
        dsp_debug.begin_data = begin_data
        dsp_debug.end_data = end_data

    # To find equivalent ADC rate, we only need one tone
    if num_pilots > 1:
        logger.warning(
            "More than 1 pilot was given but only one is necessary for recovery with unshared clock and LO. Taking the first pilot (%.2f MHz)",
            pilots_frequencies[0] * 1e-6,
        )
    f_pilot = pilots_frequencies[0]

    if dsp_debug:
        dsp_debug.real_pilot_frequencies = [f_pilot]

    begin_subframe = 0
    end_subframe = len(useful_data)

    if process_subframes:
        end_subframe = subframe_length

    if dsp_debug:
        dsp_debug.tones = []
        dsp_debug.uncorrected_data = []

    result = []
    equi_adc_rate_mean = 0.0
    num_subframes = 0
    while begin_subframe < len(useful_data):
        subframe_data = useful_data[begin_subframe:end_subframe]

        equi_adc_rate = equivalent_adc_rate_one_pilot(
            subframe_data, f_pilot, adc_rate, fir_size, cutoff=tone_filtering_cutoff
        )

        equi_adc_rate += equi_adc_rate

        sps = equi_adc_rate / symbol_rate

        # Now recover the pilot tone

        tone_data = recover_tone(
            subframe_data,
            f_pilot,
            equi_adc_rate,
            fir_size,
            cutoff=tone_filtering_cutoff,
        )

        if dsp_debug:
            dsp_debug.tones.append(tone_data)

        # Now unshift signal, apply RRC filter and downsample

        useful_data = subframe_data * np.exp(
            -1j
            * 2
            * np.pi
            * np.arange(len(subframe_data))
            * frequency_shift
            / equi_adc_rate
        )

        _, filtre = root_raised_cosine_filter(
            int(10 * sps + 2),
            roll_off,
            1 / symbol_rate,
            equi_adc_rate,
        )

        subframe_data = (
            1 / np.sqrt(sps) * np.convolve(subframe_data, filtre[1:], "same")
        )

        max_t = _best_sampling_point_float(subframe_data, sps)

        subframe_data = _downsample_float(subframe_data, max_t, sps)

        if dsp_debug:
            dsp_debug.uncorrected_data.append(subframe_data)

        # Correct phase noise
        subframe_data = correct_noise(
            subframe_data,
            max_t,
            sps,
            tone_data,
            f_pilot,
            equi_adc_rate,
            filter_size=pilot_phase_filtering_size,
        )

        result.append(subframe_data)
        begin_subframe = end_subframe

        if process_subframes:
            end_subframe = begin_subframe + subframe_length

        num_subframes += 1

    special_params = SpecialDSPParams(
        symbol_rate=symbol_rate,
        adc_rate=equi_adc_rate_mean / num_subframes,
        roll_off=roll_off,
        frequency_shift=frequency_shift,
        schema=schema,
    )

    return result, special_params, dsp_debug


# pylint: disable=too-many-statements, too-many-branches
def _dsp_bob_general(
    data: np.ndarray,
    symbol_rate: float,
    dac_rate: float,
    adc_rate: float,
    num_symbols: int,
    roll_off: float,
    frequency_shift: float,
    num_pilots: int,
    pilots_frequencies: np.ndarray,
    zc_length: int,
    zc_root: int,
    zc_rate: float,
    process_subframes: bool = False,
    subframe_length: int = 0,
    fir_size: int = 500,
    tone_filtering_cutoff: float = 10e6,
    abort_clock_recovery: float = 0,
    excl: Optional[List[Tuple[float, float]]] = None,
    pilot_phase_filtering_size: int = 0,
    num_samples_fbeat_estimation: int = 100000,
    schema: DetectionSchema = SINGLE_POLARISATION_RF_HETERODYNE,
    debug: bool = False,
) -> Tuple[Optional[List[np.ndarray]], Optional[SpecialDSPParams], Optional[DSPDebug]]:
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

    The output has stil a global phase difference.

    Args:
        data (np.ndarray): data measured by Bob.
        symbol_rate (float): symbol rate in symbols per second.
        dac_rate (float): DAC rate, in Hz.
        adc_rate (float): ADC rate, in Hz.
        num_symbols (int): number of symbols.
        roll_off (float): roll-off factor for the RRC filter.
        frequency_shift (float): frequency shift of the quantum symbol, in Hz.
        num_pilots (int): number of pilots.
        pilots_frequencies (np.ndarray): list of the frequencies of the pilots.
        zc_length (int): length of the Zadoff-Chu sequence.
        zc_root (int): root of the Zadoff-Chu sequence.
        zc_rate (float): rate of the Zadoff-Chu sequence.
        process_subframes (bool, optional): if True, process the data with subframes. Defaults to False.
        subframe_length (int, optional): number of symbols to recover in each subframes. Defaults to 0.
        fir_size (int, optional): size of the FIR filters. Defaults to 500.
        tone_filtering_cutoff (float, optional): cutoff, in Hz, for the pilot filtering. Defaults to 10e6.
        abort_clock_recovery (float, optional): maximal mismatch allowed by the clock recovery algorithm before aborting. If 0, the algorithm never aborts.. Defaults to 0.
        excl (Optional[List[Tuple[float, float]]], optional): exclusion zones for the research of pilots (i.e. frequencies where we are sure the pilots are not), given as a list of tuples of float, each elements defining excluded segment (start frequency, stop frequency). Defaults to None.
        pilot_phase_filtering_size (int, optional): size of the uniform1d filter to filter the phase correction. Defaults to 0.
        num_samples_fbeat_estimation (int, optional): number of samples for the estimation of fbeat. Defaults to 100000.
        schema (DetectionSchema, optional): detection schema to use for the DSP. Defaults to qosst_core.schema.emission.SINGLE_POLARISATION_RF_HETERODYNE.
        debug (bool, optional): if True, the DSPDebug object is returned. Defaults to False.

    Returns:
        Tuple[Optional[List[np.ndarray]], Optional[SpecialDSPParams], Optional[DSPDebug]]: list of np.ndarray, each one corresponding to the recovered symbols for a subframe, SpecialDSPParams object to give to the special dsp, and DSPDebug object if debug was true.
    """
    logger.info("Starting General DSP")
    if debug:
        logger.info("Debug mode is on.")
        dsp_debug = DSPDebug()
    else:
        dsp_debug = None

    # Find pilot frequency
    if num_pilots < 2:
        logger.error(
            "General dsp requres two pilots and only one was passed... Aborting",
        )
        return None, None, None

    # Find pilot frequency
    if num_pilots > 2:
        logger.warning(
            "More than 2 pilots were given but only two are necessary for recovery with unshared clock and unshared LO. Taking the two first pilots (%.2f MHz, %.2f MHz)",
            pilots_frequencies[0] * 1e-6,
            pilots_frequencies[1] * 1e-6,
        )

    f_pilot_1 = pilots_frequencies[0]
    f_pilot_2 = pilots_frequencies[1]

    # Take a reduce data for ZC and for clock recovery

    # Find the two real frequency
    # Let's take a smaller part of the signal to compute the difference of frequency of our two pilots

    ratio_approx = 50
    num_points = 10000000
    sps_approx = int(adc_rate / zc_rate)
    approx_zc = int(
        np.argmax(uniform_filter1d(np.abs(data), int(len(data) / ratio_approx)))
        - int(len(data) / ratio_approx) / 2
    )
    data_pilots = data[
        approx_zc
        + 2 * 3989 * sps_approx : approx_zc
        + 2 * 3989 * sps_approx
        + num_points
    ]
    f_pilot_real_1, f_pilot_real_2 = find_two_pilots(data_pilots, adc_rate, excl=excl)
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

    if abort_clock_recovery != 0 and np.abs(1 - delta_f) > abort_clock_recovery:
        logger.warning(
            "Clock recovery algorithm aborted due to too high mismatch (%f > %f). Taking adc_rate as real adc_value.",
            np.abs(1 - delta_f),
            abort_clock_recovery,
        )
        equi_adc_rate = adc_rate
    else:
        logger.debug("Clock mismatch was accepted.")
        equi_adc_rate = adc_rate / delta_f

    if dsp_debug:
        dsp_debug.equi_adc_rate = equi_adc_rate
        dsp_debug.delta_frequency_pilots = delta_f

    logger.info("Equivalent ADC rate is %.6f MHz", equi_adc_rate * 1e-6)

    sps = equi_adc_rate / symbol_rate

    logger.info("Equivalent SPS is %.6f", sps)

    # Find again the real values
    f_pilot_real_1, f_pilot_real_2 = find_two_pilots(data, equi_adc_rate, excl=excl)
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

    if dsp_debug:
        dsp_debug.real_pilot_frequencies = [f_pilot_real_1, f_pilot_real_2]
        dsp_debug.beat_frequency = f_beat

    if zc_rate == 0:
        zc_rate = dac_rate
    begin_zc, end_zc = synchronisation_zc(
        data * np.exp(-1j * 2 * np.pi * np.arange(len(data)) * f_beat / equi_adc_rate),
        zc_root,
        zc_length,
        resample=equi_adc_rate / zc_rate,
    )

    # Now that we have an estimation of the beginning of the Zadoff-Chu sequence
    # let's reestimate f_beat more properly.
    len_zc = np.ceil(zc_length * equi_adc_rate / dac_rate).astype(int)
    f_pilot_real_1, f_pilot_real_2 = find_two_pilots(
        data[end_zc + len_zc : end_zc + len_zc + num_samples_fbeat_estimation],
        equi_adc_rate,
        excl=excl,
    )
    f_beat = f_pilot_real_1 - f_pilot_1

    begin_zc, end_zc = synchronisation_zc(
        data * np.exp(-1j * 2 * np.pi * np.arange(len(data)) * f_beat / equi_adc_rate),
        zc_root,
        zc_length,
        resample=equi_adc_rate / zc_rate,
    )

    begin_data = end_zc
    end_data = int(
        begin_data + num_symbols * np.ceil(sps + 1)
    )  # We take a bit more of what is needed to be sure to have all symbols
    useful_data = data[begin_data:end_data]

    if dsp_debug:
        dsp_debug.begin_zadoff_chu = begin_zc
        dsp_debug.end_zadoff_chu = end_zc
        dsp_debug.begin_data = begin_data
        dsp_debug.end_data = end_data

    begin_subframe = 0
    end_subframe = len(useful_data)

    if process_subframes:
        end_subframe = np.ceil(subframe_length * (sps + 1) - 0.5).astype(
            int
        )  # Take enough samples to have subframe_length symbos

    if dsp_debug:
        dsp_debug.tones = []
        dsp_debug.uncorrected_data = []

    result = []
    max_t0 = -1
    frequency_shift_mean = 0.0
    num_symbols_recovered = 0
    num_subframes = 0
    while num_symbols_recovered < num_symbols:
        subframe_data = useful_data[begin_subframe:end_subframe]

        # Find beat frequency
        f_pilot_real_1 = find_one_pilot(subframe_data, equi_adc_rate, excl=excl)
        logger.info("Subframe pilot found at %f", f_pilot_real_1 * 1e-6)

        f_beat = f_pilot_real_1 - f_pilot_1

        tone_data = recover_tone(
            subframe_data,
            f_pilot_real_1,
            equi_adc_rate,
            fir_size,
            cutoff=tone_filtering_cutoff,
        )

        if dsp_debug:
            dsp_debug.tones.append(tone_data)

        # Now unshift signal taking the beat into account, apply RRC filter and downsample
        subframe_data = subframe_data * np.exp(
            -1j
            * 2
            * np.pi
            * np.arange(len(subframe_data))
            * (frequency_shift + f_beat)
            / equi_adc_rate
        )

        frequency_shift_mean += frequency_shift + f_beat

        _, filtre = root_raised_cosine_filter(
            int(10 * sps + 2),
            roll_off,
            1 / symbol_rate,
            equi_adc_rate,
        )

        subframe_data = (
            1 / np.sqrt(sps) * np.convolve(subframe_data, filtre[1:], "same")
        )

        max_t = _best_sampling_point_float(subframe_data, sps)
        if max_t0 == -1:
            max_t0 = max_t
        subframe_data = _downsample_float(subframe_data, max_t, sps)[:subframe_length]

        logger.info("Collecting %i symbols in the frame", len(subframe_data))

        last_indice = (
            begin_subframe
            + np.ceil(
                max_t
                + sps
                * np.arange(np.floor((len(data) - 0.5 - max_t) / sps).astype(int) + 1)
                - 0.5
            ).astype(int)[:subframe_length][-1]
        )

        if dsp_debug:
            dsp_debug.uncorrected_data.append(subframe_data)

        # Correct phase noise
        subframe_data = correct_noise(
            subframe_data,
            max_t,
            sps,
            tone_data,
            f_pilot_real_1,
            equi_adc_rate,
            filter_size=pilot_phase_filtering_size,
        )
        result.append(subframe_data)
        begin_subframe = np.ceil(last_indice + sps / 2 - 0.5).astype(int)

        if process_subframes:
            end_subframe = np.ceil(
                begin_subframe + subframe_length * (sps + 1) - 0.5
            ).astype(int)

        num_symbols_recovered += len(subframe_data)

        num_subframes += 1

    special_params = SpecialDSPParams(
        symbol_rate=symbol_rate,
        adc_rate=equi_adc_rate,
        roll_off=roll_off,
        frequency_shift=frequency_shift_mean / num_subframes,
        schema=schema,
    )

    return result, special_params, dsp_debug


def find_global_angle(
    received_data: np.ndarray, sent_data: np.ndarray, precision: float = 0.001
) -> Tuple[float, float]:
    """
    Find global angle between received and sent data by exhaustive search.

    The best angle is found when the real part of the covariance is the highset
    between the two sets.
    A certain number of angles will be tested to statisfy the required precision.
    In fact the number of tested points will ceil(2*pi/precision) with an actual
    precision of 2*pi/(number of points) with a precision lower or equal to the
    targeted precision.

    The returned value is an angle in radian, between -pi and pi.

    Args:
        received_data (np.ndarray): the symbols received by Bob after the DSP.
        sent_data (np.ndarray): the send symbols by Alice.
        precision (float, optional): the precision wanted on the angle, in radians. Defaults to 0.001.

    Returns:
        Tuple[float,float]: the angle that maximises the covariance, in radians, and the maximal covariance.
    """
    number_of_points = int(np.ceil(2 * np.pi / precision))
    angles = np.linspace(-np.pi, np.pi, number_of_points)

    logger.debug(
        "Finding global angle with step of %f rad (targeted presicision %f rad).",
        angles[1] - angles[0],
        precision,
    )

    max_angle = 0
    max_cov = 0
    for angle in angles:
        stack = np.stack((sent_data, received_data * np.exp(1j * angle)), axis=0)
        cov = np.cov(stack)
        if cov[0][1].real > max_cov:
            max_angle = angle
            max_cov = cov[0][1].real

    logger.debug(
        "Global angle found : %.2f rad with covariance : %.2f", max_angle, max_cov
    )
    return max_angle, max_cov


def special_dsp(
    elec_noise_data: List[np.ndarray],
    elec_shot_noise_data: List[np.ndarray],
    params: SpecialDSPParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Special DSP to apply on the electronic and electronic and shot noises.

    Args:
        elec_noise_data (List[np.ndarray]): list of arrays (for each channel) of electronic noise data.
        elec_shot_noise_data (List[np.ndarray]): list of arrays (for each channel) of electronic and shot noise data.
        params (SpecialDSPParams): the dictionnary returned by the DSP containing the required parameters.

    Returns:
        Tuple[np.ndarray, np.ndarray]: the electronic symbols and electronic and shot symbols.
    """
    logger.info("Preparing special DSP with following paramaters: %s", str(params))
    return _special_dsp_params(
        elec_noise_data[0],
        elec_shot_noise_data[0],
        params.symbol_rate,
        params.adc_rate,
        params.roll_off,
        params.frequency_shift,
        params.schema,
    )


def _special_dsp_params(
    elec_noise_data: np.ndarray,
    elec_shot_noise_data: np.ndarray,
    symbol_rate: float,
    adc_rate: float,
    roll_off: float,
    frequency_shift: float,
    _schema: DetectionSchema,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Special DSP to apply the electronic and electronic and shot noises
    taking the parameters.

    Args:
        elec_noise_data (np.ndarray): array of the electronic noise.
        elec_shot_noise_data (np.ndarray): array of the electronic and shot noise.
        symbol_rate (float): symbol rate of the quantum symbols, in Symbols per second.
        adc_rate (float): ADC rate, in Hz.
        roll_off (float): roll-off factor of the RRC filter.
        frequency_shift (float): frequency shift of the quantum data, in Hz.
        schema (DetectionSchema): schema to know how to interpret the data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: the electronic symbols and electronic and shot symbols.
    """
    logger.info("Starting DSP on elec and elec+shot noise.")

    sps = adc_rate / symbol_rate
    _, filtre = root_raised_cosine_filter(
        int(10 * sps + 2),
        roll_off,
        1 / symbol_rate,
        adc_rate,
    )

    logger.info("Starting DSP on elec noise.")
    elec_noise_bb = elec_noise_data * np.exp(
        -1j
        * 2
        * np.pi
        * np.arange(len(elec_noise_data))
        * frequency_shift
        / adc_rate
    )

    # RRC filter
    elec_noise_filtered = (
        1 / np.sqrt(sps) * np.convolve(elec_noise_bb, filtre[1:], "same")
    )

    elec_symbols = downsample(elec_noise_filtered, 0, sps)

    logger.info("Starting DSP on elec+shot noise.")

    elec_shot_noise_bb = elec_shot_noise_data * np.exp(
        -1j
        * 2
        * np.pi
        * np.arange(len(elec_noise_data))
        * frequency_shift
        / adc_rate
    )

    # RRC filter
    elec_shot_noise_filtered = (
        1 / np.sqrt(sps) * np.convolve(elec_shot_noise_bb, filtre[1:], "same")
    )
    elec_shot_symbols = downsample(elec_shot_noise_filtered, 0, sps)

    logger.info("DSP on elec and elec+shot noise finished.")
    return elec_symbols, elec_shot_symbols
