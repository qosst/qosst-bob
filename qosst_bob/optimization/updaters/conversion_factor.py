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
Updater to optimize the excess noise while varying the roll-off.
"""

import logging
from typing import Dict

import numpy as np

from qosst_bob import __version__
from qosst_bob.optimization import Updater

logger = logging.getLogger(__name__)


class ConversionFactorUpdater(Updater):
    """
    Experiments to measure the excess noise variations as a function of the variance of Alice's modulation.
    """

    conversion_factors: np.ndarray

    def _init_parameters(self):
        """
        Generate the array of conversion factors.
        """
        logger.info(
            "Initializing ConversionFactor updater with initial value=%f begin error=%f end error=%f and step error=%f",
            self.args.initial_value,
            self.args.begin_error,
            self.args.end_error,
            self.args.step_error,
        )
        self.conversion_factors = self.args.initial_value * (
            1
            + np.arange(
                self.args.begin_error,
                self.args.end_error,
                self.args.step_error,
            )
            / 100
        )

    def number_of_rounds(self) -> int:
        """
        Return the number of rounds, which is the length of the array of conversion factors.

        Returns:
            int: number of rounds.
        """
        return len(self.conversion_factors)

    def update(self) -> Dict:
        """
        Update the parameter by changing the conversion factor at Alice side only.

        Returns:
            Dict: dict with the new value of the conversion factor.
        """
        new_value = self.conversion_factors[self.round]

        logger.info("Requesting change of conversion factor of Alice to %f", new_value)
        self.bob.request_parameter_change(
            "alice.photodiode_to_output_conversion", new_value
        )
        self.round += 1
        return {"alice.photodiode_to_output_conversion": new_value}

    def name(self) -> str:
        """
        Name of the updater. To be used in the name of the saved file.

        Returns:
            str: name of the updater.
        """
        return "conversion-factor"
