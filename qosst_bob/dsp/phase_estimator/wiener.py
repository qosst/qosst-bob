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
Phase estimator based on a Wiener filter.
"""

from typing import List, Optional

import numpy as np
from scipy.signal import welch

from qosst_core.dsp.phase_estimator import BasePhaseEstimator


class WienerPhaseEstimator(BasePhaseEstimator):
    nperseg: int = 32768
    bpf_cutoff_hz: Optional[float] = None

    def _compute_noise_floor(
        self,
        pilot_data: np.ndarray,
        shot_noise_data,
        freqs: np.ndarray,
        S_total,
    ):
        """Return S_eta on the grid `freqs` (fftshift order)."""
        if shot_noise_data is not None:
            A_pilot = np.abs(pilot_data).mean()
            freqs_sn, S_Q = welch(
                shot_noise_data.imag,
                fs=self.adc_rate,
                nperseg=self.nperseg,
                return_onesided=False,
            )
            freqs_sn = np.fft.fftshift(freqs_sn)
            S_Q = np.fft.fftshift(S_Q)
            return np.interp(freqs, freqs_sn, S_Q) / (A_pilot**2)
        hf_mask = np.abs(freqs) > (self.adc_rate / 4)
        if hf_mask.sum() == 0:
            hf_mask = np.abs(freqs) > (self.adc_rate / 8)
        return np.median(S_total[hf_mask]) * np.ones_like(freqs)

    def estimate_phase(
        self,
        pilot_data: List[np.ndarray],
        shot_noise_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Estimates the pilot phase via a Wiener filter.

        Model-based : the signal PSD is the theoretical
        Lorentzian S_laser(f) = linewidth/(2π f²) of a laser phase random walk.
        H(f) = S_laser / (S_laser + S_eta).

        S_eta is taken from the shot noise quadrature PSD scaled
        by the pilot amplitude (S_Q/A²), or falls back to the HF tail of S_total.
        H is forced to 0 above bpf_cutoff_hz (adc_rate / 8 if None).
        """
        # Only the first pilot is used.
        pilot_data = pilot_data[0]

        bpf_cutoff_hz = (
            self.bpf_cutoff_hz if self.bpf_cutoff_hz is not None else self.adc_rate / 8
        )
        N = len(pilot_data)
        pilot_phase = np.unwrap(np.angle(pilot_data))

        freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / self.adc_rate))

        with np.errstate(divide="ignore", invalid="ignore"):
            S_laser = np.where(
                freqs == 0,
                np.inf,
                self.linewidth / (2.0 * np.pi * freqs**2),
            )

        # Noise floor (shot-noise-based or HF fallback).
        if shot_noise_data is None:
            freqs_w, S_total = welch(
                pilot_phase,
                fs=self.adc_rate,
                nperseg=self.nperseg,
                return_onesided=False,
            )
            freqs_w = np.fft.fftshift(freqs_w)
            S_total = np.fft.fftshift(S_total)
        else:
            freqs_w = S_total = None

        S_eta = self._compute_noise_floor(pilot_data, shot_noise_data, freqs, S_total)

        H = S_laser / (S_laser + np.maximum(S_eta, 0))
        H = np.clip(np.where(np.isfinite(H), H, 1.0), 0.0, 1.0)
        H[np.abs(freqs) > bpf_cutoff_hz] = 0.0

        H_full = np.fft.ifftshift(H)

        PILOT_F = np.fft.fft(pilot_phase)
        phi_est = np.fft.ifft(H_full * PILOT_F).real
        return phi_est
