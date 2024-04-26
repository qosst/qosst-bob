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
Updater to optimize the excess noise while varying the cutoff for the filter of the tone.
"""

import logging
from typing import Dict

import numpy as np

from qosst_bob import __version__
from qosst_bob.optimization import Updater

logger = logging.getLogger(__name__)


class FrequencyCutoffToneUpdater(Updater):
    """
    Experiments to measure the excess noise variations as a function of the cutoff for the filtering of the tone at Bob side.
    """

    cutoffs: np.ndarray

    def _init_parameters(self):
        """
        Generate the array of cut-offs.
        """
        logger.info(
            "Initializing FrequencyCutoffTone updater with begin cutoff=%f end cutoff=%f and step cutoff=%f",
            self.args.begin_cutoff,
            self.args.end_cutoff,
            self.args.step_cutoff,
        )
        self.cutoffs = np.arange(
            self.args.begin_cutoff, self.args.end_cutoff, self.args.step_cutoff
        )

    def number_of_rounds(self) -> int:
        """
        Return the number of rounds, which is the length of the array of cutoffs.

        Returns:
            int: number of rounds.
        """
        return len(self.cutoffs)

    def update(self) -> Dict:
        """
        Update the parameter by changing the cutoff for the tone filtering at Bob side only.

        Returns:
            Dict: dict with the new value of the cutoff.
        """
        assert self.config.bob is not None
        new_value = self.cutoffs[self.round]
        logger.info("Changing Bob cutoff parameter to %f", new_value)
        self.config.bob.dsp.tone_filtering_cutoff = new_value

        self.round += 1
        return {"bob.dsp.tone_filtering_cutoff": new_value}

    def name(self) -> str:
        """
        Name of the updater. To be used in the name of the saved file.

        Returns:
            str: name of the updater.
        """
        return "frequency-cutoff-tone"
