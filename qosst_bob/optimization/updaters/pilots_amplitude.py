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
Updater to optimize the excess noise while varying the amplitude of the pilots.
"""

import logging
from typing import Dict

import numpy as np

from qosst_bob import __version__
from qosst_bob.optimization import Updater

logger = logging.getLogger(__name__)


class PilotsAmplitudeUpdater(Updater):
    """
    Experiments to measure the excess noise variations as a function of the amplitude of pilots.
    """

    amplitudes: np.ndarray
    num_pilots: int

    def _init_parameters(self):
        """
        Generate the array of amplitudes.
        """
        self.num_pilots = self.config.frame.pilots.num_pilots
        logger.info(
            "Initializing PilotsAmplitude updater with begin amplitude=%f end amplitude=%f step amplitude=%f and number of pilots=%d",
            self.args.begin_amplitude,
            self.args.end_amplitude,
            self.args.step_amplitude,
            self.num_pilots,
        )
        self.amplitudes = np.arange(
            self.args.begin_amplitude, self.args.end_amplitude, self.args.step_amplitude
        )

    def number_of_rounds(self) -> int:
        """
        Return the number of rounds, which is the length of the array of amplitudes.

        Returns:
            int: number of rounds.
        """
        return len(self.amplitudes)

    def update(self) -> Dict:
        """
        Update the parameter by changing the amplitudes of pilots at Alice and Bob sides.

        Returns:
            Dict: dict with the new value of the amplitudes.
        """
        assert self.config.frame is not None
        new_value = [self.amplitudes[self.round] for _ in range(self.num_pilots)]
        logger.info("Changing Bob pilots amplitude parameter to %s", str(new_value))
        self.config.frame.pilots.amplitudes = np.array(new_value)

        logger.info(
            "Requesting change of pilots amplitude of Alice to %s", str(new_value)
        )
        self.bob.request_parameter_change("frame.pilots.amplitudes", new_value)
        self.round += 1
        return {"frame.pilots.amplitudes": new_value}

    def name(self) -> str:
        """
        Name of the updater. To be used in the name of the saved file.

        Returns:
            str: name of the updater.
        """
        return "pilots-amplitude"
