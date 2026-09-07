import numpy as np
import numpy as np
from numba import njit
from scipy.ndimage import uniform_filter1d


class ClassicalPhaseEstimator:
    def __init__(self, pilot_phase_filtering_size=1, pilot_frequency_filtering_size=1):
        self.pilot_phase_filtering_size = pilot_phase_filtering_size
        self.pilot_frequency_filtering_size = pilot_frequency_filtering_size

    def estimate_phase(
        self,
        pilot_data: np.ndarray,
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


class UKFPhaseEstimator:
    def __init__(
        self,
        adc_rate: float,
        linewidth: float,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.linewidth = linewidth
        self.adc_rate = adc_rate
        self.Q = 2 * np.pi * self.linewidth * (1 / self.adc_rate)
        self.R = None
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def estimate_phase(
        self,
        pilot_data: np.ndarray,
        shot_noise_data: np.ndarray,
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
        self.R = np.array(
            [[np.var(shot_noise_data.real), 0], [0, np.var(shot_noise_data.imag)]]
        )
        phase_diff = np.angle(pilot_data[1:] * np.conj(pilot_data[:-1]))
        delta_omega_est = np.mean(phase_diff)

        k = np.arange(len(pilot_data))
        pilot_baseband = pilot_data * np.exp(-1j * delta_omega_est * k)

        estimated_phase = run_phase_ukf(
            pilot_baseband.real,
            pilot_baseband.imag,
            self.Q,
            self.R,
            A=np.mean(np.abs(pilot_data)),
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
        )
        estimated_phase = estimated_phase + delta_omega_est * k

        return estimated_phase


@njit
def run_phase_ukf(
    z_real: np.ndarray,
    z_imag: np.ndarray,
    Q: float,
    R: np.ndarray,
    A: float = 1.0,
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
    assert R.shape == (2, 2), "Measurement noise covariance R must be a 2x2 matrix"
    assert Q > 0, "Process noise covariance Q must be positive"

    R00 = R[0, 0]
    R01 = R[0, 1]
    R11 = R[1, 1]
    N = len(z_real)
    phase_est = np.zeros(N)

    # UKF parameters for 1D state (phase)
    # lambda_, gamma: scaling parameters for sigma points
    lambda_ = alpha**2 * (1 + kappa) - 1.0
    gamma = np.sqrt(1.0 + lambda_)

    # Weights for mean and covariance
    Wm0 = lambda_ / (1.0 + lambda_)
    Wc0 = Wm0 + (1.0 - alpha**2 + beta)
    Wi = 1.0 / (2.0 * (1.0 + lambda_))

    # Initial state estimate (phase) and covariance
    x = np.angle(z_real[0] + 1j * z_imag[0])  # initial phase estimate
    P = 0.1  # initial phase variance

    for k in range(N):
        # Increase covariance by process noise Q
        P = P + Q

        sqrtP = np.sqrt(P)

        # Generate sigma points for the phase
        x0 = x
        x1 = x + gamma * sqrtP
        x2 = x - gamma * sqrtP

        # Predict measurement for each sigma point (cosine and sine projections)
        c0 = A * np.cos(x0)
        s0 = A * np.sin(x0)

        c1 = A * np.cos(x1)
        s1 = A * np.sin(x1)

        c2 = A * np.cos(x2)
        s2 = A * np.sin(x2)

        # Weighted mean of predicted measurements
        z_pred0 = Wm0 * c0 + Wi * c1 + Wi * c2  # mean of cosines
        z_pred1 = Wm0 * s0 + Wi * s1 + Wi * s2  # mean of sines

        # Start with measurement noise covariance
        S00 = R00
        S01 = R01
        S11 = R11

        # Add contributions from each sigma point
        # sigma 0
        dz0_0 = c0 - z_pred0
        dz0_1 = s0 - z_pred1
        S00 += Wc0 * dz0_0 * dz0_0
        S01 += Wc0 * dz0_0 * dz0_1
        S11 += Wc0 * dz0_1 * dz0_1

        # sigma 1
        dz1_0 = c1 - z_pred0
        dz1_1 = s1 - z_pred1
        S00 += Wi * dz1_0 * dz1_0
        S01 += Wi * dz1_0 * dz1_1
        S11 += Wi * dz1_1 * dz1_1

        # sigma 2
        dz2_0 = c2 - z_pred0
        dz2_1 = s2 - z_pred1
        S00 += Wi * dz2_0 * dz2_0
        S01 += Wi * dz2_0 * dz2_1
        S11 += Wi * dz2_1 * dz2_1

        # Cross covariance between phase and measurement
        Pxz0 = 0.0
        Pxz1 = 0.0

        # sigma 0
        dx0 = x0 - x
        Pxz0 += Wc0 * dx0 * dz0_0
        Pxz1 += Wc0 * dx0 * dz0_1

        # sigma 1
        dx1 = x1 - x
        Pxz0 += Wi * dx1 * dz1_0
        Pxz1 += Wi * dx1 * dz1_1

        # sigma 2
        dx2 = x2 - x
        Pxz0 += Wi * dx2 * dz2_0
        Pxz1 += Wi * dx2 * dz2_1

        # Kalman gain calculation (for 2D measurement)
        # Invert 2x2 innovation covariance matrix
        detS = S00 * S11 - S01 * S01

        invS00 = S11 / detS
        invS01 = -S01 / detS
        invS11 = S00 / detS

        # Kalman gain for each measurement dimension
        K0 = Pxz0 * invS00 + Pxz1 * invS01
        K1 = Pxz0 * invS01 + Pxz1 * invS11

        # Update step
        # Innovation (difference between actual and predicted measurement)
        innov0 = z_real[k] - z_pred0
        innov1 = z_imag[k] - z_pred1

        # Update phase estimate and covariance
        x = x + K0 * innov0 + K1 * innov1
        P = P - (K0 * (S00 * K0 + S01 * K1) + K1 * (S01 * K0 + S11 * K1))

        # Store phase estimate
        phase_est[k] = x

    return phase_est
