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
Updater to optimize the excess noise while varying the size of subframes.
"""

import logging
from typing import Dict

import numpy as np

from qosst_bob import __version__
from qosst_bob.optimization import Updater

logger = logging.getLogger(__name__)


class SubframeSizeUpdater(Updater):
    """
    Experiments to measure the excess noise variations as a function of the subframe size at Bob side.
    """

    sizes: np.ndarray

    def _init_parameters(self):
        """
        Generate the array of subframe sizes.
        """
        logger.info(
            "Initializing SubframeSize updater with begin sizes : %s",
            str(self.args.sizes),
        )
        self.sizes = np.array(self.args.sizes)

    def number_of_rounds(self) -> int:
        """
        Return the number of rounds, which is the length of the array of subframe sizes.

        Returns:
            int: number of rounds.
        """
        return len(self.sizes)

    def update(self) -> Dict:
        """
        Update the parameter by changing the subframe size at Bob side only.

        Returns:
            Dict: dict with the new value of the subframe size.
        """
        assert self.config.bob is not None
        new_value = self.sizes[self.round]
        logger.info("Changing Bob subframe size parameter to %f", new_value)
        self.config.bob.dsp.subframes_size = new_value

        self.round += 1
        return {"bob.dsp.subframes_size": new_value}

    def name(self) -> str:
        """
        Name of the updater. To be used in the name of the saved file.

        Returns:
            str: name of the updater.
        """
        return "subframe-size"
