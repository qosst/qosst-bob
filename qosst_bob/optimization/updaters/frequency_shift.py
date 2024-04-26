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
Updater to optimize the excess noise while varying the frequency shift.
"""

import logging
from typing import Dict

import numpy as np

from qosst_bob import __version__
from qosst_bob.optimization import Updater

logger = logging.getLogger(__name__)


class FrequencyShiftUpdater(Updater):
    """
    Experiments to measure the excess noise variations as a function of the frequency shift of the symbols.
    """

    frequency_shifts: np.ndarray

    def _init_parameters(self):
        """
        Generate the array of frequency shifts.
        """
        logger.info(
            "Initializing FrequencyShift updater with begin frequency shifts : %s",
            str(self.args.frequency_shifts),
        )
        self.frequency_shifts = np.array(self.args.frequency_shifts)

    def number_of_rounds(self) -> int:
        """
        Return the number of rounds, which is the length of the array of frequency shifts.

        Returns:
            int: number of rounds.
        """
        return len(self.frequency_shifts)

    def update(self) -> Dict:
        """
        Update the parameter by changing the frequency shift at Alice and Bob sides.

        Returns:
            Dict: dict with the new value of the frequench shift.
        """
        assert self.config.frame is not None
        new_value = self.frequency_shifts[self.round]
        logger.info("Changing Bob frequency shift parameter to %f", new_value)
        self.config.frame.quantum.frequency_shift = new_value

        logger.info("Requesting change of frequency shift of Alice to %f", new_value)
        self.bob.request_parameter_change("frame.quantum.frequency_shift", new_value)
        self.round += 1
        return {"frame.quantum.frequency_shift": new_value}

    def name(self) -> str:
        """
        Name of the updater. To be used in the name of the saved file.

        Returns:
            str: name of the updater.
        """
        return "frequency-shift"
