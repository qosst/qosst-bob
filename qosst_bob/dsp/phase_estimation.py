import logging
from typing import Tuple, Optional

import numpy as np
from numba import njit
from scipy.ndimage import uniform_filter1d

from qosst_core.dsp.phase_estimator import BasePhaseEstimator

logger = logging.getLogger(__name__)


class ClassicalPhaseEstimator(BasePhaseEstimator):
    def estimate_phase(
        self,
        pilot_data: np.ndarray,
        shot_noise_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Estimate the phase of the signal using the pilot tone.
        """
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


class UKFPhaseEstimator(BasePhaseEstimator):
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0

    def estimate_phase(
        self,
        pilot_data: np.ndarray,
        shot_noise_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Estimate the phase of the signal using the pilot tone and the UKF.

        Parameters:
            - pilot_data: array of complex pilot tone measurements
            - shot_noise_data: array of complex shot noise samples
            - linewidth: laser linewidth (Hz)
            - adc_rate: sampling rate of the ADC (Hz)
        Returns:
            - estimated_phase: array of estimated phase values
        """
        process_noise_covariance = 2 * np.pi * self.linewidth * (1 / self.adc_rate)
        measurement_noise_covariance = np.array(
            [[np.var(shot_noise_data.real), 0], [0, np.var(shot_noise_data.imag)]]
        )
        phase_diff = np.angle(pilot_data[1:] * np.conj(pilot_data[:-1]))
        delta_omega_est = np.mean(phase_diff)

        k = np.arange(len(pilot_data))
        pilot_baseband = pilot_data * np.exp(-1j * delta_omega_est * k)

        estimated_phase = run_phase_ukf(
            pilot_baseband.real,
            pilot_baseband.imag,
            process_noise_covariance,
            measurement_noise_covariance,
            amplitude=np.mean(np.abs(pilot_data)),
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
        )
        estimated_phase = estimated_phase + delta_omega_est * k

        return estimated_phase


# pylint:disable=too-many-arguments, too-many-positional-arguments, too-many-locals, too-many-statements
@njit
def run_phase_ukf(
    z_real: np.ndarray,
    z_imag: np.ndarray,
    process_noise_covariance: float,
    measurement_noise_covariance: np.ndarray,
    amplitude: float = 1.0,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> np.ndarray:
    """
    Run a UKF to estimate the phase of a signal given noisy measurements of its cosine and sine projections.

    Parameters:
        - z_real: array of real parts of the measurements
        - z_imag: array of imaginary parts of the measurements
        - Q: process noise covariance
        - R: measurement noise covariance matrix (2x2)
        - A: amplitude of the signal
        - alpha, beta, kappa: UKF scaling parameters
    Returns:
        - phase_est: array of estimated phase values
    """
    assert (
        z_real.shape == z_imag.shape
    ), "Real and imaginary measurement arrays must have the same shape"
    assert measurement_noise_covariance.shape == (
        2,
        2,
    ), "Measurement noise covariance R must be a 2x2 matrix"
    assert process_noise_covariance > 0, "Process noise covariance Q must be positive"

    r00 = measurement_noise_covariance[0, 0]
    r01 = measurement_noise_covariance[0, 1]
    r11 = measurement_noise_covariance[1, 1]
    n = len(z_real)
    phase_est = np.zeros(n)

    # UKF parameters for 1D state (phase)
    # lambda_, gamma: scaling parameters for sigma points
    lambda_ = alpha**2 * (1 + kappa) - 1.0
    gamma = np.sqrt(1.0 + lambda_)

    # Weights for mean and covariance
    wm0 = lambda_ / (1.0 + lambda_)
    wc0 = wm0 + (1.0 - alpha**2 + beta)
    wi = 1.0 / (2.0 * (1.0 + lambda_))

    # Initial state estimate (phase) and covariance
    x = np.angle(z_real[0] + 1j * z_imag[0])  # initial phase estimate
    phase_variance = 0.1  # initial phase variance

    for k in range(n):
        # Increase covariance by process noise Q
        phase_variance = phase_variance + process_noise_covariance

        sqrt_p = np.sqrt(phase_variance)

        # Generate sigma points for the phase
        x0 = x
        x1 = x + gamma * sqrt_p
        x2 = x - gamma * sqrt_p

        # Predict measurement for each sigma point (cosine and sine projections)
        c0 = amplitude * np.cos(x0)
        s0 = amplitude * np.sin(x0)

        c1 = amplitude * np.cos(x1)
        s1 = amplitude * np.sin(x1)

        c2 = amplitude * np.cos(x2)
        s2 = amplitude * np.sin(x2)

        # Weighted mean of predicted measurements
        z_pred0 = wm0 * c0 + wi * c1 + wi * c2  # mean of cosines
        z_pred1 = wm0 * s0 + wi * s1 + wi * s2  # mean of sines

        # Start with measurement noise covariance
        s00 = r00
        s01 = r01
        s11 = r11

        # Add contributions from each sigma point
        # sigma 0
        dz0_0 = c0 - z_pred0
        dz0_1 = s0 - z_pred1
        s00 += wc0 * dz0_0 * dz0_0
        s01 += wc0 * dz0_0 * dz0_1
        s11 += wc0 * dz0_1 * dz0_1

        # sigma 1
        dz1_0 = c1 - z_pred0
        dz1_1 = s1 - z_pred1
        s00 += wi * dz1_0 * dz1_0
        s01 += wi * dz1_0 * dz1_1
        s11 += wi * dz1_1 * dz1_1

        # sigma 2
        dz2_0 = c2 - z_pred0
        dz2_1 = s2 - z_pred1
        s00 += wi * dz2_0 * dz2_0
        s01 += wi * dz2_0 * dz2_1
        s11 += wi * dz2_1 * dz2_1

        # Cross covariance between phase and measurement
        pxz0 = 0.0
        pxz1 = 0.0

        # sigma 0
        dx0 = x0 - x
        pxz0 += wc0 * dx0 * dz0_0
        pxz1 += wc0 * dx0 * dz0_1

        # sigma 1
        dx1 = x1 - x
        pxz0 += wi * dx1 * dz1_0
        pxz1 += wi * dx1 * dz1_1

        # sigma 2
        dx2 = x2 - x
        pxz0 += wi * dx2 * dz2_0
        pxz1 += wi * dx2 * dz2_1

        # Kalman gain calculation (for 2D measurement)
        # Invert 2x2 innovation covariance matrix
        det_s = s00 * s11 - s01 * s01

        inv_s00 = s11 / det_s
        inv_s01 = -s01 / det_s
        inv_s11 = s00 / det_s

        # Kalman gain for each measurement dimension
        k0 = pxz0 * inv_s00 + pxz1 * inv_s01
        k1 = pxz0 * inv_s01 + pxz1 * inv_s11

        # Update step
        # Innovation (difference between actual and predicted measurement)
        innov0 = z_real[k] - z_pred0
        innov1 = z_imag[k] - z_pred1

        # Update phase estimate and covariance
        x = x + k0 * innov0 + k1 * innov1
        phase_variance = phase_variance - (
            k0 * (s00 * k0 + s01 * k1) + k1 * (s01 * k0 + s11 * k1)
        )

        # Store phase estimate
        phase_est[k] = x

    return phase_est


def find_global_angle(
    received_data: np.ndarray, sent_data: np.ndarray
) -> Tuple[float, float]:
    """
    Find the global angle between received and sent data.

    The best angle is found when the real part of the covariance is the highest
    between the two sets.

    Args:
        received_data (np.ndarray): the symbols received by Bob after the DSP.
        sent_data (np.ndarray): the send symbols by Alice.

    Returns:
        Tuple[float,float]: the angle that maximises the covariance, in radians, and the maximal covariance.
    """
    stack = np.stack((sent_data, received_data), axis=0)
    cov = np.cov(stack)[0][1]
    max_angle = np.angle(cov)
    max_cov = (cov * np.exp(-1j * max_angle)).real
    logger.debug(
        "Global angle found : %.2f rad with covariance : %.2f", max_angle, max_cov
    )
    return max_angle, max_cov
