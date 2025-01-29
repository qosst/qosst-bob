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
DSP functions to deal with Zadoff-Chu sequences and synchronisation.
"""
from typing import Tuple
import logging

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from qosst_core.comm.zc import zcsequence

from .resample import upsample

logger = logging.getLogger(__name__)


# pylint: disable=too-many-arguments
def synchronisation_zc(
    data: np.ndarray,
    zc_root: int,
    zc_length: int,
    resample: float = 1,
    use_abs=True,
    ratio_approx=50,
) -> Tuple[int, int]:
    """
    Find the beginning of a Zadoff-Chu sequence in data.

    This function finds the beginning and the end of a Zadoff-Chu sequence
    by computing the cross-correlation of the Zadoff-Chu sequence
    and the data.

    From version 0.4.27, the behavior of this function is a little different:

    first we find a first approximate of the Zadoff-Chu location by making
    a rolling average of the data, and then, we make the cross-correlation
    around this point.

    Args:
        data (np.ndarray): the data from where the Zadoff-Chu should be found.
        zc_root (int): the root of the Zadoff-Chu sequence.
        zc_length (int): the length of the Zadoff-Chu sequence.
        resample (float, optional): the optional resample to apply to the Zadoff-Chu sequence. Defaults to 1.
        ratio_approx (int, optional): the length of the data will be divided by this value to get the window size of the rolling average for the approximation. Defaults to 50.

    Returns:
        Tuple[int, int]: tuple including the beginning and the end of the Zadoff-Chu sequence.
    """
    logger.debug(
        "Trying to synchronise the Zadoff-Chu sequence of root %i and length %i with data.",
        zc_root,
        zc_length,
    )
    logger.debug(
        "Computing rolling average to get approximation of Zadoff-Chu location."
    )

    uniform_filter_length = int(zc_length * resample)
    envelope = uniform_filter1d(np.abs(data), uniform_filter_length)
    approx_zc = int(np.argmax(envelope) - uniform_filter_length / 2)
    logger.debug("Approximative position found at %i.", approx_zc)
    zadoff_chu = zcsequence(zc_root, zc_length)
    # Resample to the correct size using a zero order hold.
    zadoff_chu = upsample(zadoff_chu, resample, 0)

    n = len(zadoff_chu)
    logger.debug(
        "Upsampling sequence with resample value %f. New length is %i",
        resample,
        n,
    )

    xcorr_start_point = max(approx_zc - 2 * n, 0)
    xcorr_end_point = min(approx_zc + 2 * n, len(data))
    data_zc = data[xcorr_start_point:xcorr_end_point]
    lags = signal.correlation_lags(len(data_zc), len(zadoff_chu), mode="same")
    if use_abs:
        xcorr = signal.correlate(np.abs(data_zc), np.abs(zadoff_chu), mode="same")
    else:
        xcorr = signal.correlate(data_zc, zadoff_chu, mode="same")

    beginning_zc = lags[np.argmax(xcorr)] + xcorr_start_point
    end_zc = len(zadoff_chu) + beginning_zc

    logger.debug("Beginning was found at %i and end at %i", beginning_zc, end_zc)
    return beginning_zc, end_zc
