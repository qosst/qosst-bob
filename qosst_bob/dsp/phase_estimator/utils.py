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
Utility functions for the phase estimation.
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def find_global_angle(
    received_data: np.ndarray, sent_data: np.ndarray
) -> Tuple[float, float]:
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
