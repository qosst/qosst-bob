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


class RollOffUpdater(Updater):
    """
    Experiments to measure the excess noise variations as a function of the variance of Alice's modulation.
    """

    roll_offs: np.ndarray

    def _init_parameters(self):
        """
        Generate the array of roll-offs.
        """
        logger.info(
            "Initializing RollOff updater with begin roll-off=%f end roll-off=%f and step roll-off=%f",
            self.args.begin_roll_off,
            self.args.end_roll_off,
            self.args.step_roll_off,
        )
        self.roll_offs = np.arange(
            self.args.begin_roll_off, self.args.end_roll_off, self.args.step_roll_off
        )

    def number_of_rounds(self) -> int:
        """
        Return the number of rounds, which is the length of the array of roll-offs.

        Returns:
            int: number of rounds.
        """
        return len(self.roll_offs)

    def update(self) -> Dict:
        """
        Update the parameter by changing the roll-off factor at Alice and Bob sides.

        Returns:
            Dict: dict with the new value of the roll-off.
        """
        assert self.config.frame is not None
        new_value = self.roll_offs[self.round]
        logger.info("Changing Bob roll-off parameter to %f", new_value)
        self.config.frame.quantum.roll_off = new_value

        logger.info("Requesting change of roll-off of Alice to %f", new_value)
        self.bob.request_parameter_change("frame.quantum.roll_off", new_value)
        self.round += 1
        return {"frame.quantum.roll_off": new_value}

    def name(self) -> str:
        """
        Name of the updater. To be used in the name of the saved file.

        Returns:
            str: name of the updater.
        """
        return "roll-off"
