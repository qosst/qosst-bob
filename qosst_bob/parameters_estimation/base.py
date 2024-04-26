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
Define abstract class for estimators.
"""
import abc
import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def complex_to_real(input_data: np.ndarray) -> np.ndarray:
    """
    Transform the input data of a complex np array of size n
    to a real np array of size 2n such that if the input data is
    [a_1+i*b_1, a_2+i*b_2, ..., a_n+i*b_n] then the output array
    is [a_1, b_1, a_2, b_2, ..., a_n, b_n].

    Args:
        input_data (np.ndarray): the input complex array of size n.

    Returns:
        np.ndarray: the output real array of size 2n.
    """
    out = np.zeros(2 * len(input_data))
    out[::2] = input_data.real
    out[1::2] = input_data.imag
    return out


# pylint: disable=too-few-public-methods
class BaseEstimator(abc.ABC):
    """
    Base abstract estimator.
    """

    @staticmethod
    @abc.abstractmethod
    def estimate(
        alice_symbols: np.ndarray,
        bob_symbols: np.ndarray,
        alice_photon_number: float,
        electronic_symbols: np.ndarray,
        electronic_shot_symbols: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Estimate the transmittance and excess noise given
        the symbols of Alice and Bob, symbols for the shot noise and
        electronic noise and the avarage photon number at Alice's output.

        Transmittance should be here understood as total transmittance hence
        eta * T.

        Args:
            alice_symbols (np.ndarray): symbols sent by Alice.
            bob_symbols (np.ndarray): symbols received by Bob, after DSP.
            alice_photon_number (float): average number of photon at Alice's output.
            electronic_symbols (np.ndarray): electronic noise data after equivalent DSP.
            electronic_shot_symbols (np.ndarray): electronic and shot noise data, after equivalent DSP.

        Returns:
            Tuple[float, float, float]: tuple containing the transmittance, the excess noise at Bob side and the electronic noise.
        """


class DefaultEstimator(BaseEstimator):
    """
    Default estimator.
    """

    @staticmethod
    def estimate(
        alice_symbols: np.ndarray,
        bob_symbols: np.ndarray,
        alice_photon_number: float,
        electronic_symbols: np.ndarray,
        electronic_shot_symbols: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Estimate the transmittance, excess noise and electronic noise by
        using the covariance method.

        Args:
            alice_symbols (np.ndarray): symbols sent by Alice.
            bob_symbols (np.ndarray): symbols received by Bob, after DSP.
            alice_photon_number (float): average number of photon at Alice's output.
            electronic_symbols (np.ndarray): electronic noise data after equivalent DSP.
            electronic_shot_symbols (np.ndarray): electronic and shot noise data, after equivalent DSP.

        Returns:
            Tuple[float, float, float]: tuple containing the transmittance, the excess noise at Bob side and the electronic noise.
        """
        electronic_symbols = complex_to_real(electronic_symbols)
        electronic_shot_symbols = complex_to_real(electronic_shot_symbols)
        alice_symbols = complex_to_real(alice_symbols)
        bob_symbols = complex_to_real(bob_symbols)

        conversion_factor = np.sqrt(
            alice_photon_number / np.mean(np.abs(alice_symbols) ** 2)
        )

        shot = np.var(electronic_shot_symbols) - np.var(electronic_symbols)
        vel = np.var(electronic_symbols) / shot
        bob_symbols = bob_symbols / np.sqrt(shot)
        factor = np.cov([alice_symbols, bob_symbols])[0][1].real / np.var(alice_symbols)

        excess_noise_bob = np.var(factor * alice_symbols - bob_symbols) - 1 - vel

        transmittance = factor**2 / conversion_factor**2

        return transmittance, excess_noise_bob, vel
