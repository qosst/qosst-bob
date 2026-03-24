from qosst_bob.dsp.resample import _best_sampling_point_float
import numpy as np

class BestSamplingPointTimingRecovery:
    """
    Sampling class based on finding the best sampling point using the method from "A New Timing Recovery Method for Digital Communication Systems" by J. C. Candy and G. C. Temes (1986).
    """
    def __init__(self, sps, subframe_length, symbol_timing_oversampling):
        self.sps = sps
        self.subframe_length = subframe_length
        self.symbol_timing_oversampling = symbol_timing_oversampling

    def sample(self, data):
        best_t = _best_sampling_point_float(
            data,
            self.sps * self.symbol_timing_oversampling
        )
        best_grid = np.round(
            best_t + self.sps * self.symbol_timing_oversampling * np.arange(
                self.subframe_length)
        ).astype(int)
        
        return data[best_grid], best_grid
    
class KalmanTimingRecovery:
    """
    Tilming recovery class based on a Kalman filter. The parameters alpha and beta can be optimized to find the best sampling point.
    """
    def __init__(self, sps, adc_rate, subframe_length, symbol_timing_oversampling):
        self.sps = sps
        self.adc_rate = adc_rate
        self.subframe_length = subframe_length
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
                self.subframe_length,
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

    P = 0.001

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
        y_pilot_real, y_pilot_imag = farrow_cubic(pilot_data_real, pilot_data_imag, tau, mu) # We compute the error on the phase increment between two consecutive symbols, which should be equal to omega0 if the timing is correct.
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

