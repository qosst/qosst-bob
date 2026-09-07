# qosst-bob - Bob module of the Quantum Open Software for Secure Transmissions.
# Copyright (C) 2021-2024 Yoann Piétri
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
DSP functions to deal with synchronization.
"""

from typing import Tuple
import logging

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from qosst_core.synchronization import SynchronizationSequence

from .resample import upsample

logger = logging.getLogger(__name__)


# pylint: disable=too-many-arguments
def synchronize(
    data: np.ndarray,
    synchro_obj: SynchronizationSequence,
    use_abs: bool = False,
    resample: float = 1,
) -> Tuple[int, int]:
    """
    Find the beginning of a synchronization sequence in data.

    This function finds the beginning and the end of a synchronization sequence
    by computing the cross-correlation of the synchronization sequence
    and the data.

    From version 0.4.27, the behavior of this function is a little different:

    First we find a first approximate of the synchronization location by computing
    the correlation of the envelope of the signal and the envelope of the synchronization
    (which is a rectangle function). A more computationally expensive, but
    accurate, search is then performed by cross-correlating the signal with
    the synchronization sequence.

    Args:
        data (np.ndarray): the data from where the synchronization should be found.
        synchro_obj (SynchronizationSequence): instance of the class that generates the synchronization sequence.
        use_abs (bool, optional): use the absolute value of the synchronization in the second search. Defaults to True.
        resample (float, optional): the optional resample to apply to the synchronization sequence. Defaults to 1.

    Returns:
        Tuple[int, int]: tuple including the beginning and the end of the synchronization sequence.
    """
    logger.debug("Trying to synchronise the %s with data.", str(synchro_obj))
    logger.debug(
        "Computing rolling average to get approximation of the synchronization location."
    )

    uniform_filter_length = int(synchro_obj.length * resample)
    envelope = uniform_filter1d(np.abs(data), uniform_filter_length)
    approx_synchro = int(np.argmax(envelope) - uniform_filter_length / 2)
    logger.debug("Approximative position found at %i.", approx_synchro)
    synchro = synchro_obj.sequence()
    # Resample to the correct size using a zero order hold.
    synchro = upsample(synchro, resample, 0)

    n = len(synchro)
    logger.debug(
        "Upsampling sequence with resample value %f. New length is %i",
        resample,
        n,
    )

    xcorr_start_point = max(approx_synchro - 2 * n, 0)
    xcorr_end_point = min(approx_synchro + 2 * n, len(data))
    data_synchro = data[xcorr_start_point:xcorr_end_point]
    lags = signal.correlation_lags(len(data_synchro), len(synchro), mode="same")
    if use_abs:
        xcorr = np.abs(
            signal.correlate(np.abs(data_synchro), np.abs(synchro), mode="same")
        )
    else:
        xcorr = np.abs(signal.correlate(data_synchro, synchro, mode="same"))

    beginning_synchro = lags[np.argmax(xcorr)] + xcorr_start_point
    end_synchro = len(synchro) + beginning_synchro

    logger.debug(
        "Beginning was found at %i and end at %i", beginning_synchro, end_synchro
    )
    return beginning_synchro, end_synchro
