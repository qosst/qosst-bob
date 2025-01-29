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
Pilots processing and related functions.
"""
import logging
from typing import Tuple, List, Optional

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.ndimage import uniform_filter1d

from .resample import downsample

logger = logging.getLogger(__name__)


def recover_tones(
    data: np.ndarray,
    frequencies: List[float],
    rate: float,
    fir_size: int,
    cutoff: float = 10e6,
) -> List[np.ndarray]:
    """
    Recover all the tones within data given the list of frequencies of the tones.

    This will get the tones by applying a FIR of size fir_size and cut-off cutoff on the data.

    Args:
        data (np.ndarray): data on which the tones should be recovered.
        frequencies (List[float]): list of the frequencies of the tones.
        rate (float): rate of the data, in Samples per second.
        fir_size (int): the size of the FIR.
        cutoff (float, optional): the cut-off frequency of the filter, in Hz. Defaults to 10e6.

    Returns:
        List[np.ndarray]: the list of received tones.
    """
    logger.debug(
        "Recovering the tones with frequencies %s MHz (frequency width : %f MHz)",
        str([freq * 1e-6 for freq in frequencies]),
        cutoff * 1e-6,
    )
    return [
        recover_tone(data, frequency, rate, fir_size, cutoff)
        for frequency in frequencies
    ]


def recover_tone(
    data: np.ndarray,
    frequency: float,
    rate: float,
    fir_size: int,
    cutoff: float = 10e6,
) -> np.ndarray:
    """
    Recover the tone within data given the frequency of the tone.

    This will get the tones by applying a FIR of size fir_size and cut-off cutoff on the data.

    Args:
        data (np.ndarray): the data on which the tones should be recovered.
        frequencies (float): the frequency of the tone.
        rate (float): the rate of the data.
        fir_size (int): the size of the FIR.
        cutoff (float, optional): the cut-off frequency of the filter. Defaults to 10e6.

    Returns:
        np.ndarray: the received tone.
    """
    logger.debug(
        "Recovering tone with frequency %f MHz (frequency width : %f MHz)",
        frequency * 1e-6,
        cutoff * 1e-6,
    )
    fir = signal.firwin(fir_size, cutoff / rate) * np.exp(
        1j * 2 * np.pi * np.arange(fir_size) * frequency / rate
    )
    return signal.oaconvolve(data, fir, mode="same")


def find_one_pilot(
    data: np.ndarray, rate: float, excl: Optional[List[Tuple[float, float]]] = None
) -> float:
    """
    Find the frequency of one pilot by looking at maximums in the FFT.

    Args:
        data (np.ndarray): the data to analyze.
        rate (float): the sampling rate, in Samples per second.
        excl (Optional[List[Tuple[float, float]]], optional): List of exclusion zones. Each tuple will be considered as (beginning of exclusion zone in Hz, end of exclusion zone in Hz). Defaults to None.

    Returns:
        float: frequency of the pilot, in Hz.
    """
    logger.debug("Finding one tone by maximal peak in fft")

    if excl is None:
        excl = []

    data_fft = fft(data)
    data_fftfreq = fftfreq(len(data), 1 / rate)

    mask_exclusion_zone = data_fftfreq > 0

    for excl_zone in excl:
        mask_exclusion_zone = mask_exclusion_zone & (
            (data_fftfreq < excl_zone[0]) | (data_fftfreq > excl_zone[1])
        )

    mask_exclusion_zone = np.where(mask_exclusion_zone)[0]

    # Find maximum in the remaining data
    freq = data_fftfreq[mask_exclusion_zone][
        np.argmax(np.abs(data_fft[mask_exclusion_zone]))
    ]
    logger.debug("Tone found at %.6f MHz", freq * 1e-6)
    return freq


def find_two_pilots(
    data: np.ndarray,
    rate: float,
    tone_excl: float = 5e6,
    excl: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[float, float]:
    """
    Find the frequency of two pilots by looking at maximums in the FFT.

    Args:
        data (np.ndarray): the data to analyze.
        rate (float): the sampling rate, in Samples per second.
        tone_excl (float, optional): exclusion zone after first pilot is found (f_pilot1 - tone_excl, f_pilot1 + tone_excl). Defaults to 5e6.
        excl (List[Tuple[float, float]], optional): List of exclusion zones. Each tuple will be considered as (beginning of exclusion zone in Hz, end of exclusion zone in Hz). Defaults to None.

    Returns:
        Tuple[float, float]: frequency of the first pilot and frequency of the second pilot, in Hz. freq1 < freq2.
    """
    logger.debug("Finding two tones by maximal peak in fft")
    logger.debug("Finding first tone by maximal peak in fft")

    if excl is None:
        excl = []
    data_fft = fft(data)
    data_fftfreq = fftfreq(len(data), 1 / rate)

    mask_exclusion_zone = data_fftfreq > 0

    for excl_zone in excl:
        mask_exclusion_zone = mask_exclusion_zone & (
            (data_fftfreq < excl_zone[0]) | (data_fftfreq > excl_zone[1])
        )

    mask_exclusion_zone = np.where(mask_exclusion_zone)[0]

    # Find maximum in the remaining data
    freq_1 = data_fftfreq[mask_exclusion_zone][
        np.argmax(np.abs(data_fft[mask_exclusion_zone]))
    ]
    logger.debug("Tone found at %.6f MHz", freq_1 * 1e-6)

    # Exclud area around first tone
    logger.debug("Finding second tone by maximal peak in fft")
    logger.debug(
        "Area at +/- %.2f MHz of first tone at %.2f MHz is excluded",
        tone_excl * 1e-6,
        freq_1 * 1e-6,
    )

    mask_exclusion_zone = (data_fftfreq > 0) & (
        (data_fftfreq < (freq_1 - tone_excl)) | (data_fftfreq > (freq_1 + tone_excl))
    )

    for excl_zone in excl:
        mask_exclusion_zone = mask_exclusion_zone & (
            (data_fftfreq < excl_zone[0]) | (data_fftfreq > excl_zone[1])
        )

    mask_exclusion_zone = np.where(mask_exclusion_zone)[0]

    # Find maximum in the remaining data
    freq_2 = data_fftfreq[mask_exclusion_zone][
        np.argmax(np.abs(data_fft[mask_exclusion_zone]))
    ]
    logger.debug("Tone found at %.6f MHz", freq_2 * 1e-6)

    if freq_1 > freq_2:
        freq_1, freq_2 = freq_2, freq_1
    return (freq_1, freq_2)


def equivalent_adc_rate_one_pilot(
    data: np.ndarray,
    frequency: float,
    rate: float,
    fir_size: int,
    cutoff: float = 10e6,
) -> float:
    """
    Find the equivalent ADC rate with a linear fit on the angle difference
    of the received tone.

    Args:
        data (np.ndarray): the received data.
        frequencies (float): the frequency of the pilot tone.
        rate (float): the rate of the ADC in Hz.
        fir_size (int): the fir size of the filters for the tone recovery.
        cutoff (float, optional): the cut-off frequency of the filter. Defaults to 10e6.

    Returns:
        float: the equivalent ADC rate in Hz.
    """
    tones = recover_tone(data, frequency, rate, fir_size, cutoff)

    # For now we only take the first tone

    expected_tone = np.exp(1j * 2 * np.pi * np.arange(tones[0].size) * frequency / rate)

    angle_diff = np.angle(tones[0]) - np.angle(expected_tone)

    linear_fit, _ = np.polyfit(
        np.arange(angle_diff.size) / rate,
        np.unwrap((angle_diff + np.pi) % (2 * np.pi) - np.pi),
        1,
    )

    delta_f = linear_fit / (2 * np.pi)
    conversion_factor = (frequency + delta_f) / frequency

    return 1 / conversion_factor * rate


def phase_noise_correction(
    received_tone: np.ndarray, frequency: float, rate: float
) -> np.ndarray:
    """Return the phase difference to apply to correct the relative phase noise.

    The phase noise is computed as the difference between the received tone and the expected tone.

    Args:
        received_tone (np.ndarray): the received tone.
        frequency (float): the frequency of the received tone, in Hz.
        rate (float): the rate of the data, in Samples per second.

    Returns:
        np.ndarray: the array of phase difference.
    """
    expected_phase = np.fmod(
        0.5 + np.arange(received_tone.size) * frequency / rate,
        1.0) * 2 * np.pi - np.pi
    return np.angle(received_tone) - expected_phase


# pylint: disable=too-many-arguments
def correct_noise(
    data: np.ndarray,
    sampling_point: int,
    sps: float,
    received_tone: np.ndarray,
    frequency: float,
    rate: float,
    filter_size: int = 0,
) -> np.ndarray:
    """
    Correct phase noise using phase reference.

    Args:
        data (np.ndarray): data to correct.
        sampling_point (int): the best sampling point found.
        sps (float): the value of samples per symbol.
        received_tone (np.ndarray): the data for the received tone.
        frequency (float): the frequency of this received tone, in Hz.
        rate (float): the sampling rate, in Samples per second.
        filter_size (int, optional): size of the uniform1d filter to apply to the phase. Defaults to 0.

    Returns:
        np.ndarray: the corrected data.
    """
    angle_diff = phase_noise_correction(received_tone, frequency, rate)
    if filter_size:
        # Filter the angle diff
        logger.debug(
            "Filtering angle diff with uniform filter 1D of size %i", filter_size
        )
        angle_diff = uniform_filter1d(np.unwrap(angle_diff), filter_size)
    return data * np.exp(-1j * downsample(angle_diff, sampling_point, sps)[: len(data)])
