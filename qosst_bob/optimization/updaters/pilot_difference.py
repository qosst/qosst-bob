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
Updater to optimize the excess noise while varying the difference of pilot frequency.
"""

import logging
from typing import Dict

import numpy as np

from qosst_bob import __version__
from qosst_bob.optimization import Updater

logger = logging.getLogger(__name__)


class PilotDifferenceUpdater(Updater):
    """
    Experiments to measure the excess noise variations as a function of the difference of frequency between the two pilots. The first pilot will be left untouched.
    """

    differences: np.ndarray

    def _init_parameters(self):
        """
        Generate the array of differences.
        """
        logger.info(
            "Initializing PilotDifference updater with begin difference=%f end difference=%f and step difference=%f",
            self.args.begin_difference,
            self.args.end_difference,
            self.args.step_difference,
        )
        self.differences = np.arange(
            self.args.begin_difference,
            self.args.end_difference,
            self.args.step_difference,
        )

    def number_of_rounds(self) -> int:
        """
        Return the number of rounds, which is the length of the array of differences.

        Returns:
            int: number of rounds.
        """
        return len(self.differences)

    def update(self) -> Dict:
        """
        Update the parameter by changing the frequencies of the pilot. The first frequency stays the same but the second one is updated with the new difference.
        It is done at Alice and Bob sides.

        Returns:
            Dict: dict with the new values of the frequencies of the pilots.
        """
        assert self.config.frame is not None
        new_diff = self.differences[self.round]
        new_value = [
            self.config.frame.pilots.frequencies[0],
            self.config.frame.pilots.frequencies[0] + new_diff,
        ]
        logger.info("Changing Bob difference parameter to %f", new_diff)
        self.config.frame.pilots.frequencies = np.array(new_value)

        logger.info("Requesting change of difference of Alice to %f", new_diff)
        self.bob.request_parameter_change("frame.pilots.frequencies", new_value)
        self.round += 1
        return {"frame.pilots.frequencies": new_value}

    def name(self) -> str:
        """
        Name of the updater. To be used in the name of the saved file.

        Returns:
            str: name of the updater.
        """
        return "pilot-difference"
