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
are adapated versions of old DSP and might not work. They are untested.
"""
# pylint: disable=too-many-lines
import logging
from typing import Tuple, List, Optional, Type
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import oaconvolve, firwin

from qosst_core.configuration import Configuration
from qosst_core.schema.detection import (
    DetectionSchema,
    SINGLE_POLARISATION_RF_HETERODYNE,
)
from qosst_core.comm.filters import root_raised_cosine_filter
from qosst_core.synchronization import SynchronizationSequence
from qosst_core.dsp.phase_estimator import PhaseEstimator
from qosst_core.dsp.timing_estimator import TimingRecoveryEstimator
from qosst_bob.dsp.phase_estimation import ClassicalPhaseEstimator
from qosst_bob.dsp.timing_recovery import BestSamplingPointTimingRecovery

from .synchro import synchronize
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
    upsample
)

logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes
@dataclass
class DSPDebug:
    """
    Dataclass for debug information for the DSP.
    """

    begin_synchro: int = 0  #: Beginning of the synchronization sequence.
    end_synchro: int = 0  #: End of the synchronization sequence.
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
        res += f"Begin synchronization : {self.begin_synchronization}\n"
        res += f"End synchronization : {self.end_synchronization}\n"
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
    elec_noise_estimation_ratio: Optional[float] = 1.0  #: Ratio of electronic noise samples to analyze
    elec_shot_noise_estimation_ratio: Optional[float] = 1.0  #: Ratio of electronic and shot noise samples to analyze

    def __str__(self) -> str:
        return f"Symbol rate = {self.symbol_rate*1e-6} MBaud, ADC Rate = {self.adc_rate*1e-9} GSamples/s, Roll Off = {self.roll_off}, Frequency shift = {self.frequency_shift*1e-6} MHz, Detection schema = {str(self.schema)}, Ratio of electronic noise samples kept = {self.elec_noise_estimation_ratio}, Ratio of electronic and shot noise samples kept = {self.elec_shot_noise_estimation_ratio}"


def dsp_bob(
    data: np.ndarray, config: Configuration,
) -> Tuple[Optional[List[np.ndarray]], Optional[SpecialDSPParams], Optional[DSPDebug]]:
    """
    DSP function for Bob, given the data and the configuration.

    Args:
        data (np.ndarray): the data on which to apply the DSP.
        config (Configuration): the configuration object.
        electronic_shot_noise_data (np.ndarray, optional): the electronic shot noise data.

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
        config.frame.synchronization.synchronization_cls,
        config.frame.synchronization.zc_length,
        config.frame.synchronization.zc_root,
        config.frame.synchronization.mls_nbits,
        config.frame.synchronization.rate,
        config.bob.switch.switching_time,
        config.bob.dsp.linewidth,
        config.clock.sharing,
        config.local_oscillator.shared,
        config.bob.dsp.phase_estimator,
        config.bob.dsp.timing_recovery_estimator,
        config.bob.dsp.direct_pilot_tracking,
        config.bob.dsp.process_subframes,
        config.bob.dsp.subframes_size,
        config.bob.dsp.subframes_subdivisions,
        config.bob.dsp.fir_size,
        config.bob.dsp.tone_filtering_cutoff,
        config.bob.dsp.abort_clock_recovery,
        config.bob.dsp.exclusion_zone_pilots,
        config.bob.dsp.pilot_phase_filtering_size,
        config.bob.dsp.pilot_frequency_filtering_size,
        config.bob.dsp.num_samples_fbeat_estimation,
        config.bob.dsp.num_samples_pilot_search,
        config.bob.dsp.symbol_timing_oversampling,
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
    synchro_cls: Type[SynchronizationSequence],
    zc_length: int,
    zc_root: int,
    mls_nbits: int,
    synchro_rate: float,
    switching_time: float = 0.02,
    linewidth: float = 5e3,
    shared_clock: bool = False,
    shared_lo: bool = False,
    phase_estimator_cls: Type[PhaseEstimator] = ClassicalPhaseEstimator,
    timing_recovery_estimator_cls: Type[TimingRecoveryEstimator] = BestSamplingPointTimingRecovery,
    direct_pilot_tracking: bool = False,
    process_subframes: bool = False,
    subframe_length: int = 0,
    subframe_subdivision: int = 1,
    fir_size: int = 500,
    tone_filtering_cutoff: float = 10e6,
    abort_clock_recovery: float = 0,
    excl: Optional[List[Tuple[float, float]]] = None,
    pilot_phase_filtering_size: int = 0,
    pilot_frequency_filtering_size: int = 0,
    num_samples_fbeat_estimation: int = 100000,
    num_samples_pilot_search: int = 10_000_000,
    symbol_timing_oversampling: int = 1,
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
        synchro_cls (Type[SynchronizationSequence]): class generating the synchronization sequence.
        zc_length (int): length of the Zadoff-Chu sequence.
        zc_root (int): root of the Zadoff-Chu sequence.
        mls_nbits (int): number of bits of the Maximum Length Sequence.
        synchro_rate (float): rate of the synchronization sequence.
        shared_clock (bool, optional): if the clock is shared between Alice and Bob. Defaults to False.
        shared_lo (bool, optional): if the local oscillator is shared between Alice and Bob. Defaults to False.
        direct_pilot_tracking (bool, optional): whether the first pilot can directly be used to estimate beat frequency and phase noise. Defaults to False.
        process_subframes (bool, optional): if the data should be processed at subframes. Defaults to False.
        subframe_length (int, optional): if the previous parameter is True, the length, in samples, of the subframe. Defaults to 0.
        subframe_subdivision (int, optional): number of subdivisions of each frame for a finer-grained global phase recovery. Defaults to 1.
        fir_size (int, optional): FIR size. Defaults to 500.
        tone_filtering_cutoff (float, optional): cutoff for the FIR filter for the pilot filtering, in Hz.
        abort_clock_recovery (float, optional): Maximal mismatch allowed by the clock recovery algorithm before aborting. If 0, the algorithm never aborts. Defaults to 0.
        excl (Optional[List[Tuple[float, float]]], optional): exclusion zones for the research of pilots (i.e. frequencies where we are sure the pilots are not), given as a list of tuples of float, each elements defining excluded segment (start frequency, stop frequency).
        pilot_phase_filtering_size (int, optional): Size of the uniform1d filter to apply to the phase of the recovered pilots for correction. Defaults to 0.
        pilot_frequency_filtering_size (int, optional): Size of the uniform1d filter to apply to the frequency of the recovered pilots for correction. Defaults to 0.
        num_samples_fbeat_estimation (int, optional): number of samples to estimate the beat frequency between the two lasers. Defaults to 100000.
        num_samples_pilot_search (int, optional): number of samples to estimate the frequency of pilots. Defaults to 10000000.
        symbol_timing_oversampling (int, optional): by which factor the signal is oversampled when searching for the optimal symbol sampling time. Defaults to 1.
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
            synchro_cls,
            zc_length,
            zc_root,
            mls_nbits,
            synchro_rate,
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
            synchro_cls,
            zc_length,
            zc_root,
            mls_nbits,
            synchro_rate,
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
            synchro_cls,
            zc_length,
            zc_root,
            mls_nbits,
            synchro_rate,
            process_subframes,
            subframe_length,
            fir_size=fir_size,
            tone_filtering_cutoff=tone_filtering_cutoff,
            pilot_phase_filtering_size=pilot_phase_filtering_size,
            schema=schema,
            debug=debug,
        )
    if direct_pilot_tracking:
        return _dsp_bob_direct_pilot_tracking(
            data,
            symbol_rate,
            dac_rate,
            adc_rate,
            num_symbols,
            roll_off,
            frequency_shift,
            num_pilots,
            pilots_frequencies,
            synchro_cls,
            zc_length,
            zc_root,
            mls_nbits,
            synchro_rate,
            phase_estimator_cls=phase_estimator_cls,
            timing_recovery_estimator_cls=timing_recovery_estimator_cls,
            switching_time=switching_time,
            linewidth=linewidth,
            subframe_length=subframe_length,
            subframe_subdivision=subframe_subdivision,
            fir_size=fir_size,
            tone_filtering_cutoff=tone_filtering_cutoff,
            abort_clock_recovery=abort_clock_recovery,
            excl=excl,
            pilot_phase_filtering_size=pilot_phase_filtering_size,
            pilot_frequency_filtering_size=pilot_frequency_filtering_size,
            symbol_timing_oversampling=symbol_timing_oversampling,
            num_samples_pilot_search=num_samples_pilot_search,
            schema=schema,
            debug=debug)
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
        synchro_cls,
        zc_length,
        zc_root,
        mls_nbits,
        synchro_rate,
        process_subframes,
        subframe_length,
        fir_size=fir_size,
        tone_filtering_cutoff=tone_filtering_cutoff,
        abort_clock_recovery=abort_clock_recovery,
        excl=excl,
        pilot_phase_filtering_size=pilot_phase_filtering_size,
        num_samples_fbeat_estimation=num_samples_fbeat_estimation,
        num_samples_pilot_search=num_samples_pilot_search,
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
    synchro_cls: Type[SynchronizationSequence],
    zc_length: int,
    zc_root: int,
    mls_nbits: int,
    synchro_rate: float,
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
        synchro_cls (Type[SynchronizationSequence]): class generating the synchronization sequence.
        zc_length (int): length of the Zadoff-Chu sequence.
        zc_root (int): root of the Zadoff-Chu sequence.
        mls_nbits (int): number of bits of the Maximum Length Sequence.
        synchro_rate (float): rate of the synchronization sequence.
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

    # Create the synchronization sequence object
    synchro_obj = synchro_cls(
        root=zc_root,
        length=zc_length,
        nbits=mls_nbits,
    )

    # Recover beginning of sequence
    if syncho_rate == 0:
        syncho_rate = dac_rate
    begin_synchro, end_synchro = synchronize(
        data, synchro_obj, resample=adc_rate / synchro_rate
    )
    begin_data = end_synchro
    end_data = int(begin_data + num_symbols * sps)
    useful_data = data[begin_data:end_data]

    if dsp_debug:
        dsp_debug.begin_synchro = begin_synchro
        dsp_debug.end_synchro = end_synchro
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
            1 / np.sqrt(sps) * oaconvolve(subframe_data, filtre[1:], "same")
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
    synchro_cls: Type[SynchronizationSequence],
    zc_length: int,
    zc_root: int,
    mls_nbits: int,
    synchro_rate: float,
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
        synchro_cls (Type[SynchronizationSequence]): class generating the synchronization sequence.
        zc_length (int): length of the Zadoff-Chu sequence.
        zc_root (int): root of the Zadoff-Chu sequence.
        mls_nbits (int): number of bits of the Maximum Length Sequence.
        synchro_rate (float): rate of the synchronization sequence.
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

    # Create the synchronization sequence object
    synchro_obj = synchro_cls(
        root=zc_root,
        length=zc_length,
        nbits=mls_nbits,
    )

    if synchro_rate == 0:
        synchro_rate = dac_rate
    begin_synchro, end_synchro = synchronize(
        data * np.exp(-1j * 2 * np.pi * np.arange(len(data)) * f_beat / adc_rate),
        synchro_obj,
        resample=adc_rate / synchro_rate,
    )

    begin_data = end_synchro
    end_data = int(begin_data + num_symbols * sps)
    useful_data = data[begin_data:end_data]

    if dsp_debug:
        dsp_debug.begin_synchro = begin_synchro
        dsp_debug.end_synchro = end_synchro
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
            1 / np.sqrt(sps) * oaconvolve(subframe_data, filtre[1:], "same")
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
    synchro_cls: Type[SynchronizationSequence],
    zc_length: int,
    zc_root: int,
    mls_nbits: int,
    synchro_rate: float,
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
        synchro_cls (Type[SynchronizationSequence]): class generating the synchronization sequence.
        zc_length (int): length of the Zadoff-Chu sequence.
        zc_root (int): root of the Zadoff-Chu sequence.
        mls_nbits (int): number of bits of the Maximum Length Sequence.
        synchro_rate (float): rate of the synchronization sequence.
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

    # Create the synchronization sequence object
    synchro_obj = synchro_cls(
        root=zc_root,
        length=zc_length,
        nbits=mls_nbits,
    )

    # Recover beginning of sequence
    if synchro_rate == 0:
        synchro_rate = dac_rate
    begin_synchro, end_synchro = synchronize(
        data, synchro_obj, resample=adc_rate / synchro_rate
    )
    begin_data = end_synchro
    end_data = int(
        begin_data + num_symbols * np.ceil(sps + 1)
    )  # We take a bit more of what is needed to be sure to have all symbols
    useful_data = data[begin_data:end_data]

    if dsp_debug:
        dsp_debug.begin_synchro = begin_synchro
        dsp_debug.end_synchro = end_synchro
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
            1 / np.sqrt(sps) * oaconvolve(subframe_data, filtre[1:], "same")
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
    synchro_cls: Type[SynchronizationSequence],
    zc_length: int,
    zc_root: int,
    mls_nbits: int,
    synchro_rate: float,
    process_subframes: bool = False,
    subframe_length: int = 0,
    fir_size: int = 500,
    tone_filtering_cutoff: float = 10e6,
    abort_clock_recovery: float = 0,
    excl: Optional[List[Tuple[float, float]]] = None,
    pilot_phase_filtering_size: int = 0,
    num_samples_fbeat_estimation: int = 100000,
    num_samples_pilot_search=10_000_000,
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

    The output has still a global phase difference.

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
        synchro_cls (Type[SynchronizationSequence]): class generating the synchronization sequence.
        zc_length (int): length of the Zadoff-Chu sequence.
        zc_root (int): root of the Zadoff-Chu sequence.
        mls_nbits (int): number of bits of the Maximum Length Sequence.
        synchro_rate (float): rate of the synchronization sequence.
        process_subframes (bool, optional): if True, process the data with subframes. Defaults to False.
        subframe_length (int, optional): number of symbols to recover in each subframes. Defaults to 0.
        fir_size (int, optional): size of the FIR filters. Defaults to 500.
        tone_filtering_cutoff (float, optional): cutoff, in Hz, for the pilot filtering. Defaults to 10e6.
        abort_clock_recovery (float, optional): maximal mismatch allowed by the clock recovery algorithm before aborting. If 0, the algorithm never aborts.. Defaults to 0.
        excl (Optional[List[Tuple[float, float]]], optional): exclusion zones for the research of pilots (i.e. frequencies where we are sure the pilots are not), given as a list of tuples of float, each elements defining excluded segment (start frequency, stop frequency). Defaults to None.
        pilot_phase_filtering_size (int, optional): size of the uniform1d filter to filter the phase correction. Defaults to 0.
        num_samples_fbeat_estimation (int, optional): number of samples for the estimation of fbeat. Defaults to 100000.
        num_samples_pilot_search (int, optional): number of samples to estimate the frequency of pilots. Defaults to 10000000.
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
            "General dsp requires two pilots and only one was passed... Aborting",
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

    # Create the synchronization sequence object
    synchro_obj = synchro_cls(
        root=zc_root,
        length=zc_length,
        nbits=mls_nbits,
    )

    # Use the base DAC rate if the sample rate of the ZC sequence has not been
    # provided.
    if synchro_rate == 0:
        synchro_rate = dac_rate
    sps_approx = int(adc_rate / synchro_rate)

    # A first approximate search of the start of the ZC sequence, based on the
    # signal envelope.
    uniform_filter_length = int(synchro_obj.length * sps_approx)
    envelope = uniform_filter1d(np.abs(data), uniform_filter_length)
    approx_synchro = int(np.argmax(envelope) - uniform_filter_length / 2)

    # The pilot frequencies are estimated on a large sample
    # (typically 10M points) taken after the ZC sequence.
    pilot_start_point = approx_synchro + 2 * zc_length * sps_approx
    data_pilots = data[pilot_start_point:pilot_start_point+num_samples_pilot_search]
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

    # Find again the real values.
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

    begin_synchro, end_synchro = synchronize(
        data * np.exp(-1j * 2 * np.pi * np.arange(len(data)) * f_beat / equi_adc_rate),
        synchro_obj,
        resample=equi_adc_rate / synchro_rate,
    )

    # Now that we have an estimation of the beginning of the synchronization sequence
    # let's reestimate f_beat more properly.
    len_synchro = np.ceil(synchro_obj.length * equi_adc_rate / dac_rate).astype(int)
    f_pilot_real_1, f_pilot_real_2 = find_two_pilots(
        data[end_synchro + len_synchro : end_synchro + len_synchro + num_samples_fbeat_estimation],
        equi_adc_rate,
        excl=excl,
    )
    f_beat = f_pilot_real_1 - f_pilot_1

    begin_synchro, end_synchro = synchronize(
        data * np.exp(-1j * 2 * np.pi * np.arange(len(data)) * f_beat / equi_adc_rate),
        synchro_obj,
        resample=equi_adc_rate / synchro_rate,
    )

    begin_data = end_synchro
    end_data = int(
        begin_data + num_symbols * np.ceil(sps + 1)
    )  # We take a bit more of what is needed to be sure to have all symbols
    useful_data = data[begin_data:end_data]

    if dsp_debug:
        dsp_debug.begin_synchro = begin_synchro
        dsp_debug.end_synchro = end_synchro
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
        logger.info("Subframe pilot found at %f MHz", f_pilot_real_1 * 1e-6)

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
            1 / np.sqrt(sps) * oaconvolve(subframe_data, filtre[1:], "same")
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


# pylint: disable=too-many-statements, too-many-branches
def _dsp_bob_direct_pilot_tracking(
    data: np.ndarray,
    symbol_rate: float,
    dac_rate: float,
    adc_rate: float,
    num_symbols: int,
    roll_off: float,
    frequency_shift: float,
    num_pilots: int,
    pilots_frequencies: np.ndarray,
    synchro_cls: Type[SynchronizationSequence],
    zc_length: int,
    zc_root: int,
    mls_nbits: int,
    synchro_rate: float,
    phase_estimator_cls: Type[PhaseEstimator],
    timing_estimator_cls: Type[TimingRecoveryEstimator],
    switching_time: float = 0.02,
    linewidth: float = 5e3,
    subframe_length: int = 50_000,
    subframe_subdivision: int = 1,
    fir_size: int = 500,
    tone_filtering_cutoff: float = 10e6,
    abort_clock_recovery: float = 0,
    excl: Optional[List[Tuple[float, float]]] = None,
    pilot_phase_filtering_size: int = 0,
    pilot_frequency_filtering_size: int = 0,
    symbol_timing_oversampling: int = 1,
    num_samples_pilot_search: int = 1_000_000,
    schema: DetectionSchema = SINGLE_POLARISATION_RF_HETERODYNE,
    debug: bool = False,
) -> Tuple[Optional[List[np.ndarray]], Optional[SpecialDSPParams], Optional[DSPDebug]]:
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
        synchro_cls (Type[SynchronizationSequence]): class generating the synchronization sequence.
        zc_length (int): length of the Zadoff-Chu sequence.
        zc_root (int): root of the Zadoff-Chu sequence.
        mls_nbits (int): number of bits of the Maximum Length Sequence.
        synchro_rate (float): rate of the synchronization sequence.
        phase_estimator_cls (Type[PhaseEstimator]): class to use for the estimation of the phase of the pilot.
        subframe_length (int, optional): number of symbols to recover in each subframes. Defaults to 50000.
        subframe_subdivision (int, optional): number of subdivisions of each frame for a finer-grained global phase recovery. Defaults to 1.
        fir_size (int, optional): size of the FIR filters. Defaults to 500.
        tone_filtering_cutoff (float, optional): cutoff, in Hz, for the pilot filtering. Defaults to 10e6.
        abort_clock_recovery (float, optional): maximal mismatch allowed by the clock recovery algorithm before aborting. If 0, the algorithm never aborts.. Defaults to 0.
        excl (Optional[List[Tuple[float, float]]], optional): exclusion zones for the research of pilots (i.e. frequencies where we are sure the pilots are not), given as a list of tuples of float, each elements defining excluded segment (start frequency, stop frequency). Defaults to None.
        pilot_phase_filtering_size (int, optional): size of the uniform1d filter to filter the phase correction. Defaults to 0.
        pilot_frequency_filtering_size (int, optional): size of the uniform1d filter to filter the phase correction. Defaults to 0.
        symbol_timing_oversampling (int, optional): by which factor the signal is oversampled when searching for the optimal symbol sampling time. Defaults to 1.
        num_samples_pilot_search (int, optional): number of samples to estimate the frequency of pilots. Defaults to 10000000.
        schema (DetectionSchema, optional): detection schema to use for the DSP. Defaults to qosst_core.schema.emission.SINGLE_POLARISATION_RF_HETERODYNE.
        debug (bool, optional): if True, the DSPDebug object is returned. Defaults to False.

    Returns:
        Tuple[Optional[List[np.ndarray]], Optional[SpecialDSPParams], Optional[DSPDebug]]: list of np.ndarray, each one corresponding to the recovered symbols for a subframe, SpecialDSPParams object to give to the special dsp, and DSPDebug object if debug was true.
    """
    logger.info("Starting General DSP with direct pilot tracking")
    if debug:
        logger.info("Debug mode is on.")
        dsp_debug = DSPDebug()
    else:
        dsp_debug = None

    # Separate shot noise if switching time is given
    if switching_time:
        end_electronic_shot_noise = int(
            switching_time * adc_rate
        )
        electronic_shot_noise_data = data[:end_electronic_shot_noise]
        data = data[end_electronic_shot_noise:]

    # Find pilot frequency
    if num_pilots < 1:
        logger.error("At least one pilot is required... Aborting")
        return None, None, None

    if num_pilots > 2:
        logger.warning(
            "More than 2 pilots were given but only two are necessary for recovery with unshared clock and unshared LO. Taking the two first pilots (%.2f MHz, %.2f MHz)",
            pilots_frequencies[0] * 1e-6,
            pilots_frequencies[1] * 1e-6,
        )

    f_pilot_1 = pilots_frequencies[0]

    # Convert the data to float32
    data = data.astype('f')

    # Create the synchronization sequence object
    synchro_obj = synchro_cls(
        root=zc_root,
        length=zc_length,
        nbits=mls_nbits,
    )

    # Use the base DAC rate if the sample rate of the synchronization sequence has not been
    # provided.
    if synchro_rate == 0:
        synchro_rate = dac_rate
    synchro_oversampling = int(adc_rate / synchro_rate)

    logger.info("Computing envelope for approximate synchronization sequence search")
    envelope = np.abs(data[::synchro_oversampling])
    envelope = uniform_filter1d(envelope, synchro_obj.length)
    preamble_synchro_start = (np.argmax(envelope) - synchro_obj.length // 2) * synchro_oversampling

    # The pilot frequencies are estimated on a large sample
    # (typically 10M points) taken after the synchronization sequence.
    pilot_start_point = preamble_synchro_start + 2 * zc_length * synchro_oversampling
    data_pilots = data[pilot_start_point:pilot_start_point+num_samples_pilot_search]

    if num_pilots == 2:
        f_pilot_2 = pilots_frequencies[1]
        logger.info("Searching for pilots")
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
            dsp_debug.real_pilot_frequencies = [f_pilot_real_1, f_pilot_real_2]
    else:
        # Only one pilot found, so we assume that the ADC rate perfectly
        # matches the specified value.
        equi_adc_rate = adc_rate

        logger.info("Searching for the pilot")
        f_pilot_real_1 = find_one_pilot(data_pilots, equi_adc_rate, excl=excl)
        dsp_debug.real_pilot_frequencies = [f_pilot_real_1]

    # Correct estimates with true ADC rate (if estimated).
    f_pilot_1 *= equi_adc_rate / adc_rate
    sps = equi_adc_rate / symbol_rate
    f_beat = f_pilot_real_1 - f_pilot_1

    logger.info("Equivalent ADC rate is %.6f MHz", equi_adc_rate * 1e-6)
    logger.info("Equivalent SPS is %.6f", sps)

    if dsp_debug:
        dsp_debug.beat_frequency = f_beat

    logger.info('Searching for start of the synchronization sequence')
    synchro_search_start = max(preamble_synchro_start - 4 * synchro_obj.length * synchro_oversampling, 0)
    synchro_search_end = synchro_search_start + 8 * synchro_obj.length * synchro_oversampling
    data_synchro = data[synchro_search_start:synchro_search_end]
    shift = np.exp(-1j * 2 * np.pi * np.arange(len(data_synchro)) * f_beat / equi_adc_rate)
    begin_synchro, end_synchro = synchronize(
        data_synchro * shift, synchro_obj,
        resample=equi_adc_rate / synchro_rate)
    begin_synchro += synchro_search_start
    end_synchro += synchro_search_start

    begin_data = end_synchro
    end_data = int(
        begin_data + num_symbols * np.ceil(sps + 1)
    )  # We take a bit more of what is needed to be sure to have all symbols
    useful_data = data[begin_data:end_data]

    if dsp_debug:
        dsp_debug.begin_synchro = begin_synchro
        dsp_debug.end_synchro = end_synchro
        dsp_debug.begin_data = begin_data
        dsp_debug.end_data = end_data
        dsp_debug.tones = []
        dsp_debug.uncorrected_data = []

    begin_subframe = 0
    end_subframe = int(np.ceil(subframe_length * (sps + 1) - 0.5))
    result = []
    num_symbols_recovered = 0

    # Number of samples to include from the previous subframe to account for
    # the boundary conditions of the filters.
    num_samples_previous_subframe = max(pilot_phase_filtering_size, fir_size)

    # Pre-compute the RRC filter.
    _, rrc_filter = root_raised_cosine_filter(
        int(10 * sps + 2),
        roll_off,
        1 / symbol_rate,
        equi_adc_rate,
    )
    rrc_filter = (rrc_filter[1:] / np.sqrt(sps)).astype(np.complex64)

    # Pre-compute the complex exponential for shifting.
    shift_size = subframe_length * (sps + 1) + num_samples_previous_subframe
    shift_up = np.exp(
        1j
        * 2
        * np.pi
        * np.arange(shift_size)
        * (f_pilot_1 - frequency_shift)
        / equi_adc_rate
    ).astype(np.complex64)

    # Pre-compute the filter extracting the pilot tone.
    pilot_bp_filter = (firwin(fir_size, tone_filtering_cutoff / equi_adc_rate) * np.exp(
        1j * 2 * np.pi * np.arange(fir_size) * f_pilot_real_1 / equi_adc_rate
    )).astype(np.complex64)

    # Correct the phase noise on the whole frame before starting to extract symbols, 
    # to avoid the boundary effects of the filters on the subframes.
    logger.info("Recovering first pilot tone")
    pilot_data = oaconvolve(useful_data, pilot_bp_filter, mode="same")
    shot_noise_data = oaconvolve(electronic_shot_noise_data, pilot_bp_filter, mode="same")
    
    logger.info("Correcting phase noise on the whole frame")
    if phase_estimator_cls == ClassicalPhaseEstimator:
        phase_estimator = phase_estimator_cls(pilot_phase_filtering_size=pilot_phase_filtering_size, pilot_frequency_filtering_size=pilot_frequency_filtering_size)
        phase_noise = phase_estimator.estimate_phase(pilot_data)
    else:
        phase_estimator = phase_estimator_cls(linewidth, equi_adc_rate)
        phase_noise = phase_estimator.estimate_phase(pilot_data, shot_noise_data)

    clean_pilot = np.exp(-1j * phase_noise).astype(np.complex64)

    logger.info("Cancelling phase noise")
    useful_data = useful_data.astype(np.complex64) * clean_pilot

    timing_estimator = timing_estimator_cls(sps, subframe_length, symbol_timing_oversampling)

    while num_symbols_recovered < num_symbols:
        # Include more samples to account for the boundary condition of filters.
        begin_extended_subframe = max(
            begin_subframe - num_samples_previous_subframe, 0
        )
        subframe_data = useful_data[begin_extended_subframe:end_subframe].astype(np.complex64)

        logger.info("Shifting quantum data to baseband")
        subframe_data *= shift_up[:len(subframe_data)]

        logger.info("Applying RRC filter")
        subframe_data = oaconvolve(subframe_data, rrc_filter, "same")

        # Ignore the extra samples at the beginning of the frame
        subframe_data = subframe_data[begin_subframe - begin_extended_subframe:]

        logger.info("Finding best decision point")
        if symbol_timing_oversampling != 1:
            subframe_data = upsample(subframe_data, symbol_timing_oversampling, 2)

        best_grid = timing_estimator.estimate_timing(subframe_data)

        logger.info("Downsampling")
        subframe_data = subframe_data[best_grid]
        last_index = begin_subframe + best_grid[-1] / symbol_timing_oversampling

        logger.info("Collecting %i symbols in the frame", len(subframe_data))
        chunk_length = len(subframe_data) // subframe_subdivision
        for i in range(subframe_subdivision):
            start = i * chunk_length
            if i != subframe_subdivision - 1:
                result.append(subframe_data[start:start + chunk_length])
            else:
                result.append(subframe_data[start:])
        num_symbols_recovered += len(subframe_data)

        begin_subframe = int(last_index + sps / 2 - 0.5)
        end_subframe = int(begin_subframe + subframe_length * (sps + 1) - 0.5)

        if dsp_debug:
            dsp_debug.uncorrected_data.append(np.array([]))

    special_params = SpecialDSPParams(
        symbol_rate=symbol_rate,
        adc_rate=equi_adc_rate,
        roll_off=roll_off,
        frequency_shift=frequency_shift + f_beat,
        schema=schema,
    )
    return result, special_params, dsp_debug


def find_global_angle(
    received_data: np.ndarray,
    sent_data: np.ndarray) -> Tuple[float, float]:
    """
    Find the global angle between received and sent data.

    The best angle is found when the real part of the covariance is the highest
    between the two sets.

    Args:
        received_data (np.ndarray): the symbols received by Bob after the DSP.
        sent_data (np.ndarray): the send symbols by Alice.

    Returns:
        Tuple[float,float]: the angle that maximises the covariance, in radians, and the maximal covariance.
    """
    stack = np.stack((sent_data, received_data), axis=0)
    cov = np.cov(stack)[0][1]
    max_angle = np.angle(cov)
    max_cov = (cov * np.exp(-1j * max_angle)).real
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
    logger.info("Preparing special DSP with following parameters: %s", str(params))
    return _special_dsp_params(
        elec_noise_data[0],
        elec_shot_noise_data[0],
        params.symbol_rate,
        params.adc_rate,
        params.roll_off,
        params.frequency_shift,
        params.schema,
        params.elec_noise_estimation_ratio,
        params.elec_shot_noise_estimation_ratio
    )


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

    if position in ['m', 'middle']:
        index_from = int(n / 2 * (1 - ratio))
        index_to = int(n / 2 * (1 + ratio))
    elif position in ['h', 'head']:
        index_from = 0
        index_to = int(n * ratio)
    elif position in ['t', 'tail']:
        index_from = int(n * (1 - ratio))
        index_to = n
    else:
        raise ValueError(f"position must be one of 'head', 'middle' or 'tail' (got '{ position }')")
    return data[index_from:index_to]


def _special_dsp_params(
    elec_noise_data: np.ndarray,
    elec_shot_noise_data: np.ndarray,
    symbol_rate: float,
    adc_rate: float,
    roll_off: float,
    frequency_shift: float,
    _schema: DetectionSchema,
    elec_noise_estimation_ratio: float,
    elec_shot_noise_estimation_ratio: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Special DSP to apply on the electronic and electronic and shot noise samples

    Args:
        elec_noise_data (np.ndarray): array of the electronic noise.
        elec_shot_noise_data (np.ndarray): array of the electronic and shot noise.
        symbol_rate (float): symbol rate of the quantum symbols, in Symbols per second.
        adc_rate (float): ADC rate, in Hz.
        roll_off (float): roll-off factor of the RRC filter.
        frequency_shift (float): frequency shift of the quantum data, in Hz.
        schema (DetectionSchema): schema to know how to interpret the data.
        elec_noise_estimation_ratio (float): proportion of electronic noise samples to keep for estimation.
        elec_shot_noise_estimation_ratio (float): proportion of electronic and shot noise samples to keep for estimation.

    Returns:
        Tuple[np.ndarray, np.ndarray]: the electronic symbols and electronic and shot symbols.
    """
    logger.info("Starting DSP on elec and elec+shot noise.")

    sps = adc_rate / symbol_rate

    # For efficiency reasons, the DSP on this section is performed
    # on float32s/complex64s.
    elec_noise_data = _subsample(elec_noise_data, elec_noise_estimation_ratio, 'tail')
    elec_noise_data = elec_noise_data.astype(np.complex64)
    n_elec_noise_data = len(elec_noise_data)

    elec_shot_noise_data = _subsample(elec_shot_noise_data, elec_shot_noise_estimation_ratio, 'tail')
    elec_shot_noise_data = elec_shot_noise_data.astype(np.complex64)
    n_elec_shot_noise_data = len(elec_shot_noise_data)

    # Precompute the filter and complex exponential for shifting.
    _, rrc_filter = root_raised_cosine_filter(
        int(10 * sps + 2),
        roll_off,
        1 / symbol_rate,
        adc_rate,
    )
    rrc_filter = rrc_filter[1:].astype('f')

    n_shift = max(n_elec_noise_data, n_elec_shot_noise_data)
    shift = np.exp(
        -1j
        * 2
        * np.pi
        * np.arange(n_shift)
        * frequency_shift
        / adc_rate).astype(np.complex64)

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