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
Classical phase estimator: angle of the pilot, optionally filtered.
"""

from typing import List, Optional

import numpy as np
from scipy.ndimage import uniform_filter1d

from qosst_core.dsp.phase_estimator import BasePhaseEstimator


class ClassicalPhaseEstimator(BasePhaseEstimator):
    def estimate_phase(
        self,
        pilot_data: List[np.ndarray],
        shot_noise_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Estimate the phase of the signal using the pilot tone.
        """
        # Only the first pilot is used.
        pilot_data = pilot_data[0]

        pilot_angle = np.angle(pilot_data)

        if (
            self.pilot_phase_filtering_size > 1
            or self.pilot_frequency_filtering_size > 1
        ):
            print("Filtering pilot")
            # The unwrapped angle can grow into a large number, but the full
            # precision is needed. Convert to double.
            pilot_angle = np.unwrap(pilot_angle.astype("d"))
            if self.pilot_phase_filtering_size > 1:
                pilot_angle = uniform_filter1d(
                    pilot_angle, self.pilot_phase_filtering_size
                )

            if self.pilot_frequency_filtering_size > 1:
                pilot_angle = np.cumsum(
                    uniform_filter1d(
                        np.diff(pilot_angle, append=pilot_angle[-1]),
                        self.pilot_frequency_filtering_size,
                    )
                )
        return pilot_angle
