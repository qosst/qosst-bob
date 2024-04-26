# qosst-bob - Bob module of the Quantum Open Software for Secure Transmissions.
# Copyright (C) 2021-2024 Yoann Piétri
# Copyright (C) 2021-2024 Ilektra Karakosta-Amarantidou

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
Module for equalization.
"""
import logging
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CMAEqualizer:
    """
    Channel equalizer based on the Constant Modulus Algorithm (CMA).
    """

    length: int  #: Length of the equalizer.
    step: float  #: Step of the equalizer.
    p_param: int  #: P parameter of the equalizer.
    q_param: int  #: Q parameter of the equalizer.
    target_radius: float  #: Target radius of the equalizer.
    error_threshold: (
        float  #: Error threshold (training stop when this value is reached).
    )
    weights: Optional[np.ndarray]  #: Weights of the equalizer.

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        length: int,
        step: float,
        p_param: int = 2,
        q_param: int = 2,
        target_radius: float = 1,
        error_threshold: float = 0.02,
    ) -> None:
        """Initialize the CMA equalizer.

        Args:
            length (int): length of the equalizer.
            step (float): step of the equalizer.
            p_param (int, optional): p parameter of the equalizer. Defaults to 2.
            q_param (int, optional): q parametwer of the equalizer. Defaults to 2.
            target_radius (float, optional): target radius of the equalizer. Defaults to 1.
            error_threshold (float, optional): error threshold. The training will stop wen the desired error is reached. Setting 0 will make the algorithm run on all the data. Defaults to 0.02.
        """
        logger.info(
            "Initializing CMA equalizer with parameter length = %i, step = %f, p = %i, q = %i, target radius = %f, error threshold = %f.",
            length,
            step,
            p_param,
            q_param,
            target_radius,
            error_threshold,
        )
        self.length = length
        self.step = step
        self.p_param = p_param
        self.q_param = q_param
        self.target_radius = target_radius
        self.error_threshold = error_threshold
        self.weights = None

    def train(
        self, train_data: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Train the CMA on the train data and updates the weights.

        Args:
            train_data (np.ndarray): the data that should have a constant modulus.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]: a tuple containing the corrected tain data (only before the error threshold was reached), the errors vector and the weights vector.
        """
        logger.info("Start training of equalizer.")
        n_symbols = np.size(train_data)
        error = np.zeros(n_symbols, dtype=complex)
        weights = np.zeros(self.length, dtype=complex)
        weights[-1] = 1
        equalized = np.zeros(n_symbols, dtype=complex)
        index = 0
        current_error = self.error_threshold + 1

        while (
            np.abs(current_error) > self.error_threshold
            and index <= n_symbols - self.length
        ):
            current_data = train_data[index : self.length + index]
            try:
                current_data_corrected = weights.conj().T @ current_data
                equalized[index] = current_data_corrected
                current_error = (
                    (
                        (
                            np.abs(current_data_corrected) ** self.p_param
                            - self.target_radius
                        )
                        ** (self.q_param - 1)
                    )
                    * (np.abs(current_data_corrected) ** (self.p_param - 2))
                    * np.conj(current_data_corrected)
                )
                logger.debug("Current error : %f", current_error)
                error[index] = current_error
                weights = weights - self.step * error[index] * current_data
                index += 1
            except RuntimeWarning:
                logger.error(
                    "There was an overflow error in the equalizer. Choose different combination of step size and number of symbols for equalization. Aborting equalization."
                )
                return train_data, None, None
        self.weights = weights
        logger.info(
            "Equalizer trained. Final error : %f after %i rounds", current_error, index
        )
        return equalized, error, weights

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the equalizer to the data.

        Args:
            data (np.ndarray): the data to apply the equalizer on.

        Returns:
            np.ndarray: the equalized data.
        """
        if self.weights is None:
            logger.warning(
                "CMA equalizer was not trained. Not applying it and returning the initial data."
            )
            return data
        logger.info("Applying the equalizer.")
        n_symbols = np.size(data)
        return np.array(
            [
                self.weights.conj().T @ data[index : self.length + index]
                for index in range(n_symbols - self.length + 1)
            ]
        )
