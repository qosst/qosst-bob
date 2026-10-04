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
Phase estimator based on Savitzky-Golay filters.
"""

from functools import lru_cache
from typing import List, Optional

import numpy as np
from scipy.signal import savgol_coeffs, savgol_filter, oaconvolve

from qosst_core.dsp.phase_estimator import BasePhaseEstimator


class SavGolPhaseEstimator(BasePhaseEstimator):
    """
    Phase estimator similar to ClassicalPhaseEstimator, but using Savitzky-Golay filters.

    Logic:
    - unwrap the pilot phase
    - if pilot_phase_filtering_size > 1: smooth the unwrapped phase
    - if pilot_frequency_filtering_size > 1: smooth the discrete phase derivative
      and reconstruct the phase by cumulative summation

    Notes:
    - pilot_data is assumed to already be in baseband
    - phase and frequency smoothing can use different polynomial orders
    """

    savgol_phase_polyorder: int = 2
    savgol_frequency_polyorder: int = 2
    savgol_mode: str = "mirror"
    savgol_use_fft: bool = True

    def _apply_savgol(
        self,
        x: np.ndarray,
        window_length: int,
        polyorder: int,
    ) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing to a real 1D array using either:
        - scipy.signal.savgol_filter
        - FFT-based overlap-add convolution
        """
        x = np.asarray(x, dtype=np.float64)

        w = _prepare_savgol_window(
            window_length=window_length,
            polyorder=polyorder,
            n=x.size,
        )

        if w == 0:
            return x

        if self.savgol_mode == "interp" or not self.savgol_use_fft:
            return np.asarray(
                savgol_filter(
                    x,
                    window_length=w,
                    polyorder=polyorder,
                    mode=self.savgol_mode,
                ),
                dtype=np.float64,
            )

        return _fast_savgol_filter_fft(
            x,
            window_length=w,
            polyorder=polyorder,
            mode=self.savgol_mode,
        )

    def estimate_phase(
        self,
        pilot_data: List[np.ndarray],
        shot_noise_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Estimate the phase of a pilot tone already shifted to baseband.
        """
        # Only the first pilot is used.
        pilot_data = np.asarray(pilot_data[0])

        if pilot_data.size == 0:
            return np.array([], dtype=np.float64)

        pilot_angle = np.unwrap(np.angle(pilot_data).astype(np.float64))

        # Smooth the unwrapped phase directly
        if self.pilot_phase_filtering_size > 1:
            pilot_angle = self._apply_savgol(
                pilot_angle,
                int(self.pilot_phase_filtering_size),
                self.savgol_phase_polyorder,
            )

        # Smooth the discrete phase derivative and reconstruct the phase
        if self.pilot_frequency_filtering_size > 1:
            dphi = np.diff(pilot_angle, append=pilot_angle[-1])
            dphi = self._apply_savgol(
                dphi,
                int(self.pilot_frequency_filtering_size),
                self.savgol_frequency_polyorder,
            )
            pilot_angle = np.cumsum(dphi)

        return np.asarray(pilot_angle, dtype=np.float64)


def _prepare_savgol_window(window_length: int, polyorder: int, n: int) -> int:
    """
    Make a Savitzky-Golay window valid:
    - integer
    - odd
    - <= n
    - > polyorder
    """
    if n <= 0:
        return 0

    w = int(window_length)

    if w < 1:
        return 0

    w = min(w, n)

    if w % 2 == 0:
        w -= 1

    min_valid = polyorder + 2
    if min_valid % 2 == 0:
        min_valid += 1

    if w < min_valid:
        w = min_valid

    if w > n:
        w = n if n % 2 == 1 else n - 1

    if w <= polyorder or w < 3:
        return 0

    return w


@lru_cache(maxsize=128)
def _cached_savgol_kernel(window_length: int, polyorder: int) -> np.ndarray:
    """
    Cached Savitzky-Golay FIR coefficients.
    """
    return np.asarray(
        savgol_coeffs(
            window_length=window_length,
            polyorder=polyorder,
            deriv=0,
            delta=1.0,
            use="conv",
        ),
        dtype=np.float64,
    )


def _map_savgol_mode_to_pad(mode: str) -> str:
    """
    Map savgol_filter modes to equivalent np.pad modes.
    """
    mode = mode.lower()

    mapping = {
        "mirror": "reflect",
        "nearest": "edge",
        "wrap": "wrap",
        "constant": "constant",
    }

    if mode not in mapping:
        raise ValueError(
            f"Mode '{mode}' is not supported in the FFT-based implementation. "
            "Use 'mirror', 'nearest', 'wrap', 'constant', or 'interp'."
        )

    return mapping[mode]


def _fast_savgol_filter_fft(
    x: np.ndarray,
    window_length: int,
    polyorder: int,
    mode: str = "mirror",
) -> np.ndarray:
    """
    Fast Savitzky-Golay smoothing using:
    - precomputed SavGol FIR coefficients
    - padding
    - overlap-add convolution
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size

    if n == 0:
        return x.copy()

    pad_mode = _map_savgol_mode_to_pad(mode)
    kernel = _cached_savgol_kernel(window_length, polyorder)
    half = window_length // 2

    x_pad = np.pad(x, (half, half), mode=pad_mode)
    y_pad = oaconvolve(x_pad, kernel, mode="same")
    y = y_pad[half : half + n]

    return y
