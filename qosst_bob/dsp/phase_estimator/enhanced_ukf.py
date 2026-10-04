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
Phase estimator based on an unscented Kalman filter and RTS smoother with physical noise parameters.
"""

from typing import List, Optional

import numpy as np
from numba import njit
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch

from qosst_core.dsp.phase_estimator import BasePhaseEstimator


# Combined Alice + Bob LO linewidth (Hz), best value found on the test captures
LINEWIDTH = 300.0


class EnhancedUKFPhaseEstimator(BasePhaseEstimator):
    decimation: int = 80
    smooth: bool = True
    amplitude_window: int = 1000

    def estimate_phase(
        self,
        pilot_data: List[np.ndarray],
        shot_noise_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Estimate the phase of the pilot with a UKF using physical noise parameters.

        Steps:
            1. carrier removal (phase increment between samples), decimation, FFT refinement of the frequency
            2. shot-noise PSD at the pilot frequency: R = N0 * f / 2 per quadrature, Q = 2 pi linewidth / f
            3. pilot normalised by its local power (fading), R scaled accordingly
            4. other pilots aligned on the first one and combined (maximum ratio)
            5. UKF forward pass and RTS backward pass, interpolated back to the ADC rate

        Parameters:
            - pilot_data: band-pass filtered pilots at the ADC rate, the first one is the reference
            - shot_noise_data: shot noise filtered by the band-pass filter of the first pilot
        Returns:
            - estimated_phase: phase of the pilot at the ADC rate, carrier included
        """
        m = self.decimation
        f = self.adc_rate / m
        z, carrier, w = _baseband(pilot_data[0], m)
        n0, noise_power = _noise_level(shot_noise_data[:len(pilot_data[0])], w, m, f)
        power = _local_power(z, noise_power, self.amplitude_window)
        z = z / np.sqrt(power)
        weight = power

        # same noise level assumed in all the pilot bands
        for pilot in pilot_data[1:]:
            z2, _, _ = _baseband(pilot, m)
            power_2 = _local_power(z2, noise_power, self.amplitude_window)
            z2 = z2 / np.sqrt(power_2)
            z2 = z2 * np.exp(1j * _relative_phase(z, z2))
            z = (weight * z + power_2 * z2) / (weight + power_2)
            weight = weight + power_2

        phase = run_phase_ukf_rts(
            z.real.copy(),
            z.imag.copy(),
            2 * np.pi * self.linewidth / f,
            n0 * f / 2 / weight,
            self.smooth,
        )
        phase = np.interp(np.arange(len(pilot_data[0])), (np.arange(len(phase)) + 0.5) * m - 0.5, phase)
        return phase + carrier


def _decimate(x: np.ndarray, m: int) -> np.ndarray:
    return x[: len(x) // m * m].reshape(-1, m).mean(1)


def _fft_frequency(x: np.ndarray, pad: int = 4) -> float:
    """
    Frequency (rad/sample) of the strongest line: zero-padded FFT peak with parabolic interpolation.
    """
    nfft = 1 << int(np.ceil(np.log2(len(x) * pad)))
    S = np.abs(np.fft.fft(x, nfft)) ** 2
    k = int(np.argmax(S))
    a, b, c = S[k - 1], S[k], S[(k + 1) % nfft]
    w = 2 * np.pi * (k + 0.5 * (a - c) / (a - 2 * b + c)) / nfft
    return (w + np.pi) % (2 * np.pi) - np.pi


def _baseband(pilot: np.ndarray, m: int):
    """
    Decimated baseband pilot, carrier phase at the ADC rate and carrier frequency (rad/sample at the ADC rate).
    """
    k = np.arange(len(pilot))
    w = np.angle(np.sum(pilot[1:] * np.conj(pilot[:-1])))
    z = _decimate(pilot * np.exp(-1j * w * k), m)
    w_fine = _fft_frequency(z)
    z = z * np.exp(-1j * w_fine * np.arange(len(z)))
    w = w + w_fine / m
    return z, w * k, w


def _noise_level(noise: np.ndarray, w: float, m: int, f: float):
    """
    Two-sided PSD (per Hz) of the noise at the pilot frequency and its power per decimated sample.
    """
    nb = _decimate(noise * np.exp(-1j * w * np.arange(len(noise))), m)
    freqs, S = welch(nb, fs=f, nperseg=4096, return_onesided=False)
    return np.median(S[np.abs(freqs) < 0.5e6]), np.mean(np.abs(nb) ** 2)


def _local_power(z: np.ndarray, noise_power: float, window: int) -> np.ndarray:
    """
    Pilot power per decimated sample: local average minus noise (fading), or constant if window is 0.
    Floored at 1 % of the noise power, for a pilot lost in a deep fade.
    """
    mean_power = max(np.mean(np.abs(z) ** 2) - noise_power, 0.01 * noise_power)
    if not window:
        return np.full(len(z), mean_power)
    local = uniform_filter1d(np.abs(z) ** 2, window, mode="nearest") - noise_power
    return np.maximum(local, 0.05 * mean_power)


def _relative_phase(z: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """
    Phase of z relative to z2 (laser phase cancels): FFT frequency of the product and mean offset.
    """
    p = z * np.conj(z2)
    k = np.arange(len(p))
    w = _fft_frequency(p)
    return w * k + np.angle(np.sum(p * np.exp(-1j * w * k)))


@njit
def run_phase_ukf_rts(
        z_real: np.ndarray,
        z_imag: np.ndarray,
        Q: float,
        R: np.ndarray,
        smooth: bool,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0
    ) -> np.ndarray:
    """
    UKF of the phase of a unit-amplitude pilot, with an optional RTS smoother.

    Parameters:
        - z_real, z_imag: measurements
        - Q: process noise variance per sample (2 pi linewidth / sample rate)
        - R: measurement noise variance per quadrature, per sample
        - smooth: if True, run the RTS backward pass
        - alpha, beta, kappa: UKF scaling parameters
    Returns:
        - phase_est: estimated phase
    """
    N = len(z_real)
    lambda_ = alpha**2 * (1 + kappa) - 1.0
    gamma = np.sqrt(1.0 + lambda_)
    Wm0 = lambda_ / (1.0 + lambda_)
    Wc0 = Wm0 + (1.0 - alpha**2 + beta)
    Wi = 1.0 / (2.0 * (1.0 + lambda_))

    x_filt = np.empty(N)
    P_filt = np.empty(N)
    P_pred = np.empty(N)
    x = np.arctan2(z_imag[0], z_real[0])
    P = 0.1

    for k in range(N):
        P = P + Q
        P_pred[k] = P
        sqrtP = np.sqrt(P)
        x1 = x + gamma * sqrtP
        x2 = x - gamma * sqrtP

        c0, s0 = np.cos(x), np.sin(x)
        c1, s1 = np.cos(x1), np.sin(x1)
        c2, s2 = np.cos(x2), np.sin(x2)
        z_pred0 = Wm0 * c0 + Wi * (c1 + c2)
        z_pred1 = Wm0 * s0 + Wi * (s1 + s2)

        dz0_0, dz0_1 = c0 - z_pred0, s0 - z_pred1
        dz1_0, dz1_1 = c1 - z_pred0, s1 - z_pred1
        dz2_0, dz2_1 = c2 - z_pred0, s2 - z_pred1
        S00 = R[k] + Wc0 * dz0_0 * dz0_0 + Wi * (dz1_0 * dz1_0 + dz2_0 * dz2_0)
        S01 = Wc0 * dz0_0 * dz0_1 + Wi * (dz1_0 * dz1_1 + dz2_0 * dz2_1)
        S11 = R[k] + Wc0 * dz0_1 * dz0_1 + Wi * (dz1_1 * dz1_1 + dz2_1 * dz2_1)

        dx1 = x1 - x
        dx2 = x2 - x
        Pxz0 = Wi * (dx1 * dz1_0 + dx2 * dz2_0)
        Pxz1 = Wi * (dx1 * dz1_1 + dx2 * dz2_1)

        detS = S00 * S11 - S01 * S01
        K0 = (Pxz0 * S11 - Pxz1 * S01) / detS
        K1 = (Pxz1 * S00 - Pxz0 * S01) / detS

        x = x + K0 * (z_real[k] - z_pred0) + K1 * (z_imag[k] - z_pred1)
        P = P - (K0 * (S00 * K0 + S01 * K1) + K1 * (S01 * K0 + S11 * K1))
        x_filt[k] = x
        P_filt[k] = P

    if smooth:
        for k in range(N - 2, -1, -1):
            x_filt[k] = x_filt[k] + P_filt[k] / P_pred[k + 1] * (x_filt[k + 1] - x_filt[k])

    return x_filt
