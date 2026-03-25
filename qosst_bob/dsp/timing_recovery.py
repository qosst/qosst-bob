from qosst_bob.dsp.resample import _best_sampling_point_float
import numpy as np
from math import gcd
from scipy.signal import resample_poly
from numba import njit

class BestSamplingPointTimingRecovery:
    """
    Sampling class based on finding the best sampling point using the method from "A New Timing Recovery Method for Digital Communication Systems" by J. C. Candy and G. C. Temes (1986).
    """
    def __init__(self, sps, data_length, symbol_timing_oversampling):
        self.sps = sps
        self.data_length = data_length
        self.symbol_timing_oversampling = symbol_timing_oversampling

    def sample(self, data):
        best_t = _best_sampling_point_float(
            data,
            self.sps * self.symbol_timing_oversampling
        )
        best_grid = np.round(
            best_t + self.sps * self.symbol_timing_oversampling * np.arange(
                self.data_length)
        ).astype(int)
        
        return best_grid

class StaticTimingRecovery:
    def __init__(self, sps, data_length, symbol_timing_oversampling, offset=10, initial_sampling_point=13):
        self.data_length = data_length
        self.sps = sps
        self.offset = offset
        self.symbol_timing_oversampling = symbol_timing_oversampling
        self.initial_sampling_point = initial_sampling_point

    def resample(self, data):
        epsilon = self.offset / len(self.data_length)  # ~3.8e-7, refine this from your 13→23 measurement
        denom = 10_000_000
        numer = int(round(denom / (1 + epsilon)))
        g = gcd(numer, denom)
        data = resample_poly(
            data, numer // g, denom // g
        ).astype(np.complex64)
    
    def sample(self, data):
        best_grid = np.round(
            self.initial_sampling_point + self.sps * self.symbol_timing_oversampling * np.arange(
                self.data_length)
        ).astype(int)
        
        return best_grid


    
class KalmanTimingRecovery:
    """
    Tilming recovery class based on a Kalman filter. The parameters alpha and beta can be optimized to find the best sampling point.
    """
    def __init__(self, sps, adc_rate, data_length, symbol_timing_oversampling):
        self.sps = sps
        self.adc_rate = adc_rate
        self.data_length = data_length
        self.symbol_timing_oversampling = symbol_timing_oversampling

    def sample(
        self,
        data: np.ndarray,
        pilot_data: np.ndarray,
    ):
        # Find a coarse estimate of the residual frequency offset by looking at the phase of the pilot tone. We average the phase increment over blocks of samples to reduce noise.
        block = 10000
        pilot_data = pilot_data[:int(2.5e7)]
        dphi = np.angle(pilot_data[1:] * np.conj(pilot_data[:-1]))
        dphi_blocked = dphi[:len(dphi)//block*block].reshape(-1, block).mean(axis=1)

        # From block-averaged dphi, compute the measurement noise variance R and the residual frequency offset f_residual, which will be used in the Kalman filter.
        mean_dphi_per_sample = np.mean(dphi_blocked) / block 
        f_residual_corrected = mean_dphi_per_sample * self.adc_rate / (2*np.pi)
        dphi_zero_mean = dphi - np.mean(dphi)
        R = np.var(dphi_zero_mean)

        # Get the initial sampling point
        tau = _best_sampling_point_float(data[:1250000], self.sps)

        # Get process noise variance of sps. 
        Q_sps = 1e-10

        grid = _timing_kalman(
                pilot_data.real,
                pilot_data.imag,
                self.data_length,
                tau,
                self.sps,
                f_residual_corrected,
                self.adc_rate,
                Q_sps,
                R,
            )
        return grid
    
@njit
def _timing_kalman(
        pilot_data_real: np.ndarray,
        pilot_data_imag: np.ndarray,
        num_symbols: int,
        tau: float,
        sps: float,
        f_residual: float,
        adc_rate: float,
        Q_sps: float,
        R: float
        ) -> np.ndarray:
    """
    Downsample the signal using a Kalman filter to track the timing offset and symbol rate.
    The timing error is estimated using the phase of the pilot tone.
    Parameters:
        - data_real: real part of the input signal
        - data_imag: imaginary part of the input signal
        - num_symbols: number of symbols to sample
        - tau: initial timing offset (in samples)
        - sps: initial symbol rate (in samples per symbol)
        - f_residual: residual frequency offset (Hz)
        - adc_rate: sampling rate of the ADC (Hz)
        - Q_sps: process noise variance for symbol rate
        - R: measurement noise variance
    Returns:
        - symbols: array of sampled symbols
        - grid: array of sampling points (in samples)
    """
    grid = np.zeros(num_symbols, dtype=np.float64)

    # Initial guess variance
    P = 0.1

    # expected phase increment per symbol
    phase_to_samples = adc_rate / (2*np.pi*f_residual)

    y_pilot_prev_real = pilot_data_real[0]
    y_pilot_prev_imag = pilot_data_imag[0]
    tau_next = tau
    sps_next = sps

    errors = np.zeros(num_symbols)

    for k in range(num_symbols):
        tau = int(np.floor(tau_next))
        sps = sps_next

        # Sample the data at the predicted timing by interpolating using a cubic Farrow filter.
        mu = tau_next - tau
        y_pilot_real, y_pilot_imag = _farrow_cubic(pilot_data_real, pilot_data_imag, tau, mu) # We compute the error on the phase increment between two consecutive symbols, which should be equal to omega0 if the timing is correct.
        grid[k] = tau_next

        # Timing errror using the pilot tone
        if k > 0:
            y_pilot = y_pilot_real + 1j*y_pilot_imag
            y_pilot_prev = y_pilot_prev_real + 1j*y_pilot_prev_imag
            dphi = np.angle(y_pilot * np.conj(y_pilot_prev)) # Phase error between two samples
            e = dphi - sps / phase_to_samples 
        else:
            e = 0.0
        errors[k] = e
    
        # Kalman update
        # predict covariance P_prediction = P + Q
        P_pred = P + Q_sps
        # compute gain from predicted covariance
        H = 1.0 / phase_to_samples
        S = H*H*P_pred + R
        K = P_pred * H / S
        # update state and covariance
        sps_next = sps + K * e
        P = (1 - K*H) * P_pred

        # integrate tau deterministically
        alpha = 0.00 # Small drift for tau also
        tau_next += sps_next + alpha * e

        y_pilot_prev_real, y_pilot_prev_imag = y_pilot_real, y_pilot_imag

    return grid


@njit
def _farrow_cubic(
    data_real: np.ndarray, 
    data_imag: np.ndarray, 
    n: float, 
    mu: float
    ):
    """
    Cubic Farrow interpolator (JIT-compiled for speed).

    Parameters:
        data_real : real part of the input signal
        data_imag : imaginary part of the input signal
        n  : integer sample index
        mu : fractional delay in [0,1)

    Returns:
        y       : interpolated sample
    """

    x_1_real = data_real[n - 1]
    x_1_imag = data_imag[n - 1]
    x0_real = data_real[n]
    x0_imag = data_imag[n]
    x1_real = data_real[n + 1]
    x1_imag = data_imag[n + 1]
    x2_real = data_real[n + 2]
    x2_imag = data_imag[n + 2]

    a0_real = x0_real
    a0_imag = x0_imag
    a1_real = 0.5 * (x1_real - x_1_real)
    a1_imag = 0.5 * (x1_imag - x_1_imag)
    a2_real = x_1_real - 2.5 * x0_real + 2 * x1_real - 0.5 * x2_real
    a2_imag = x_1_imag - 2.5 * x0_imag + 2 * x1_imag - 0.5 * x2_imag
    a3_real = 0.5 * (x2_real - x_1_real) + 1.5 * (x0_real - x1_real)
    a3_imag = 0.5 * (x2_imag - x_1_imag) + 1.5 * (x0_imag - x1_imag)

    y_real = a0_real + mu * (a1_real + mu * (a2_real + mu * a3_real))
    y_imag = a0_imag + mu * (a1_imag + mu * (a2_imag + mu * a3_imag))

    return y_real, y_imag
