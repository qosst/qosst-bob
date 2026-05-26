from qosst_bob.dsp.resample import _best_sampling_point_float
import numpy as np
from math import gcd
from scipy.signal import resample_poly
from numba import njit
from .resample import (
    _best_sampling_point_float,
    upsample
)
import logging
from scipy.signal import lfilter, butter, sosfiltfilt
from qosst_core.comm.filters import root_raised_cosine_filter
from scipy.signal import oaconvolve

from qosst_core.dsp.timing_estimator import TimingRecoveryEstimator

logger = logging.getLogger(__name__)

class BestSamplingPointTimingRecovery(TimingRecoveryEstimator):
    """
    Sampling class based on finding the best sampling point using the method from "A New Timing Recovery Method for Digital Communication Systems" by J. C. Candy and G. C. Temes (1986).
    """
    def __init__(
            self, 
            sps: float,
            adc_rate: float,
            num_symbols: int,
            subframe_length: int,
            symbol_timing_oversampling: int,
            roll_off: float,
            symbol_rate: float,
            f_pilot_1: float,
            frequency_shift: float,
            num_samples_previous_subframe: int,
            pulse_sampling: bool = False,
            **kwargs,
            ):
        self.sps = sps
        self.adc_rate = adc_rate
        self.symbol_timing_oversampling = symbol_timing_oversampling
        self.num_symbols = num_symbols
        self.subframe_length = subframe_length
        self.roll_off = roll_off
        self.symbol_rate = symbol_rate
        self.f_pilot_1 = f_pilot_1
        self.frequency_shift = frequency_shift
        self.num_samples_previous_subframe = num_samples_previous_subframe
        self.pulse_sampling = pulse_sampling

    def sample(self, data, **kwargs):
        rrc_filter, shift_up = pre_compute_filters(
            shift_size = self.subframe_length * (self.sps + 1) + self.num_samples_previous_subframe,
            sps = self.sps,
            roll_off = self.roll_off,
            symbol_rate = self.symbol_rate,
            adc_rate = self.adc_rate,
            f_pilot_1 = self.f_pilot_1,
            frequency_shift = self.frequency_shift,
        )

        begin_subframe = 0
        end_subframe = int(np.ceil(self.subframe_length * (self.sps + 1) - 0.5))
        result = []
        num_symbols_recovered = 0
        while num_symbols_recovered < self.num_symbols:
            # Include more samples to account for the boundary condition of filters.
            begin_extended_subframe = max(
                begin_subframe - self.num_samples_previous_subframe, 0
            )
            subframe_data = data[begin_extended_subframe:end_subframe].astype(np.complex64)

            subframe_data *= shift_up[:len(subframe_data)]

            subframe_data = oaconvolve(subframe_data, rrc_filter, "same")

            # Ignore the extra samples at the beginning of the frame
            subframe_data = subframe_data[begin_subframe - begin_extended_subframe:]

            if self.symbol_timing_oversampling != 1:
                subframe_data = upsample(subframe_data, self.symbol_timing_oversampling, 2)

            best_t = _best_sampling_point_float(
                subframe_data,
                self.sps * self.symbol_timing_oversampling
            )
            best_grid = np.round(
                best_t + self.sps * self.symbol_timing_oversampling * np.arange(
                    self.subframe_length)
            ).astype(int)

            if self.pulse_sampling:
                subframe_data = pulse_sampling(
                    subframe_data,
                    self.sps * self.symbol_timing_oversampling,
                    self.subframe_length
                )
            else:
                subframe_data = subframe_data[best_grid]

            last_index = begin_subframe + best_grid[-1] / self.symbol_timing_oversampling

            result.append(subframe_data)
            num_symbols_recovered += len(subframe_data)

            begin_subframe = int(last_index + self.sps / 2 - 0.5)
            end_subframe = int(begin_subframe + self.subframe_length * (self.sps + 1) - 0.5)

        return result, best_grid

class StaticTimingRecovery(TimingRecoveryEstimator):

    def __init__(
            self, 
            sps: float,
            adc_rate: float,
            num_symbols: int,
            subframe_length: int,
            symbol_timing_oversampling: int,
            roll_off: float,
            symbol_rate: float,
            f_pilot_1: float,
            frequency_shift: float,
            offset: float = 10, 
            initial_sampling_point: float = 13,
            pulse_sampling: bool = False,
            **kwargs,
        ):
        self.sps = sps
        self.adc_rate = adc_rate
        self.num_symbols = num_symbols
        self.subframe_length = subframe_length
        self.symbol_timing_oversampling = symbol_timing_oversampling
        self.roll_off = roll_off
        self.symbol_rate = symbol_rate
        self.f_pilot_1 = f_pilot_1
        self.frequency_shift = frequency_shift
        self.offset = offset
        self.initial_sampling_point = initial_sampling_point
        self.pulse_sampling = pulse_sampling
    
    def sample(self, data, **kwargs):
        result = []
        rrc_filter, shift_up = pre_compute_filters(
            shift_size = len(data),
            sps = self.sps,
            roll_off = self.roll_off,
            symbol_rate = self.symbol_rate,
            adc_rate = self.adc_rate,
            f_pilot_1 = self.f_pilot_1,
            frequency_shift = self.frequency_shift
        )

        epsilon = self.offset / len(data)
        denom = 10_000_000
        numer = int(round(denom / (1 + epsilon)))
        g = gcd(numer, denom)
        data = resample_poly(
            data, numer // g, denom // g
        ).astype(np.complex64)
        data *= shift_up[:len(data)]
        data = oaconvolve(data, rrc_filter, "same")
        if self.symbol_timing_oversampling != 1:          # <-- add this
            data = upsample(data, self.symbol_timing_oversampling, 2)
        best_grid = np.round(
            self.initial_sampling_point + self.sps * self.symbol_timing_oversampling * np.arange(self.num_symbols)
        ).astype(int)


        if self.pulse_sampling:
            data = pulse_sampling(
                data,
                self.sps * self.symbol_timing_oversampling,
                self.num_symbols
            )
        else:
            data = data[best_grid]
        
        for i in range(0, self.num_symbols // self.subframe_length):
            subframe = data[i*self.subframe_length:(i+1)*self.subframe_length]
            if len(subframe) > 0:
                result.append(subframe)
        return result, best_grid

class KalmanTimingRecovery(TimingRecoveryEstimator):
    """
    Tilming recovery class based on a Kalman filter. The parameters alpha and beta can be optimized to find the best sampling point.
    """
    def __init__(
            self, 
            sps: float, 
            adc_rate: float, 
            num_symbols: int,
            subframe_length: int,
            symbol_timing_oversampling: int,
            roll_off: float,
            symbol_rate: float,
            f_pilot_1: float,
            frequency_shift: float,
            block_size: int = 10000,
            processed_variance: float = 1e-10,
            initial_variance: float = 0.1,
            pulse_sampling: bool = False,
            **kwargs,
        ):
        self.sps = sps
        self.adc_rate = adc_rate
        self.num_symbols = num_symbols
        self.subframe_length = subframe_length
        self.symbol_timing_oversampling = symbol_timing_oversampling
        self.roll_off = roll_off
        self.symbol_rate = symbol_rate
        self.f_pilot_1 = f_pilot_1
        self.frequency_shift = frequency_shift
        self.block_size = block_size
        self.processed_variance = processed_variance
        self.initial_variance = initial_variance
        self.pulse_sampling = pulse_sampling

    def sample(
        self,
        data: np.ndarray,
        pilot_data: np.ndarray,
        **kwargs,
    ):
        rrc_filter, shift_up = pre_compute_filters(
            shift_size = len(data),
            sps = self.sps,
            roll_off = self.roll_off,
            symbol_rate = self.symbol_rate,
            adc_rate = self.adc_rate,
            f_pilot_1 = self.f_pilot_1,
            frequency_shift = self.frequency_shift
        )

        # Find a coarse estimate of the residual frequency offset by looking at the phase of the pilot tone. We average the phase increment over blocks of samples to reduce noise.
        pilot_data = pilot_data[:int(self.num_symbols * self.sps * self.symbol_timing_oversampling)]
        dphi = np.angle(pilot_data[1:] * np.conj(pilot_data[:-1]))
        dphi_blocked = dphi[:len(dphi)//self.block_size*self.block_size].reshape(-1, self.block_size).mean(axis=1)

        # From block-averaged dphi, compute the measurement noise variance R and the residual frequency offset f_residual, which will be used in the Kalman filter.
        mean_dphi_per_sample = np.mean(dphi_blocked) / self.block_size
        f_residual_corrected = mean_dphi_per_sample * self.adc_rate / (2*np.pi)
        dphi_zero_mean = dphi - np.mean(dphi)
        R = np.var(dphi_zero_mean)

        # Get the initial sampling point
        tau = _best_sampling_point_float(data[:int(self.subframe_length * self.sps)], self.sps)

        best_grid = _timing_kalman(
                pilot_data.real,
                pilot_data.imag,
                self.num_symbols,
                tau,
                self.sps,
                f_residual_corrected,
                self.adc_rate,
                self.processed_variance,
                self.initial_variance,
                R,
            )
        
        data *= shift_up[:len(data)]
        data = oaconvolve(data, rrc_filter, "same")
        
        result = []
        for i in range(0, self.num_symbols // self.subframe_length):
            subgrid = best_grid[i*self.subframe_length:(i+1)*self.subframe_length]
            if self.pulse_sampling:
                subframe = data[subgrid[0] - self.sps * self.symbol_timing_oversampling / 2 - 0.5 : subgrid[-1] + self.sps * self.symbol_timing_oversampling / 2 + 0.5]
                subframe = pulse_sampling(
                    subframe,
                    self.sps * self.symbol_timing_oversampling,
                    self.subframe_length
                )
            else:
                subframe = data[subgrid.astype(int)]

            if len(subframe) > 0:
                result.append(subframe)

        return result, best_grid


class LeastSquaresTimingRecovery(TimingRecoveryEstimator):
    """
    Offline timing recovery based on:
      1) matched filtering
      2) local timing-anchor estimation on a few bootstrap blocks
      3) global least-squares timing drift fit
      4) final sampling on the fitted grid with cubic Farrow interpolation

    This is designed for slowly varying timing drift.
    """

    def __init__(
        self,
        sps: float,
        adc_rate: float,
        num_symbols: int,
        subframe_length: int,
        symbol_timing_oversampling: int,
        roll_off: float,
        symbol_rate: float,
        f_pilot_1: float,
        frequency_shift: float,
        num_samples_previous_subframe: int,
        bootstrap_subframes: int = 9,
        bootstrap_block_symbols: int = 5000,
        fit_method: str = "polyfit",
        max_relative_sps_deviation: float = 2e-3,
        pulse_sampling: bool = False,
        **kwargs,
    ):
        self.sps = sps
        self.adc_rate = adc_rate
        self.num_symbols = num_symbols
        self.subframe_length = subframe_length
        self.symbol_timing_oversampling = symbol_timing_oversampling
        self.roll_off = roll_off
        self.symbol_rate = symbol_rate
        self.f_pilot_1 = f_pilot_1
        self.frequency_shift = frequency_shift
        self.num_samples_previous_subframe = num_samples_previous_subframe
        self.bootstrap_subframes = bootstrap_subframes
        self.bootstrap_block_symbols = bootstrap_block_symbols
        self.fit_method = fit_method
        self.max_relative_sps_deviation = max_relative_sps_deviation
        self.pulse_sampling = pulse_sampling

    def sample(self, data: np.ndarray, **kwargs):
        if self.pulse_sampling:
            logger.warning(
                "LeastSquaresTimingRecovery ignores pulse_sampling and uses "
                "fractional point sampling on the fitted grid."
            )

        data = np.asarray(data, dtype=np.complex64)

        # 1) matched filtering once, on the whole useful_data
        rrc_filter, shift_up = pre_compute_filters(
            shift_size=len(data),
            sps=self.sps,
            roll_off=self.roll_off,
            symbol_rate=self.symbol_rate,
            adc_rate=self.adc_rate,
            f_pilot_1=self.f_pilot_1,
            frequency_shift=self.frequency_shift,
        )

        data_filt = data * shift_up[:len(data)]
        data_filt = oaconvolve(data_filt, rrc_filter, "same").astype(np.complex64)

        # 2) estimate local timing anchors
        L = max(1, int(self.symbol_timing_oversampling))
        block_symbols = int(max(200, self.bootstrap_block_symbols))
        num_anchors = int(max(3, self.bootstrap_subframes))

        max_start_sym = max(0, self.num_symbols - block_symbols)
        start_symbols = np.linspace(0, max_start_sym, num_anchors).astype(int)

        anchor_symbol_idx = []
        anchor_sample_idx = []

        for start_sym in start_symbols:
            # local region around that block
            block_start = int(np.floor(start_sym * self.sps))
            block_end = int(np.ceil((start_sym + block_symbols + 2) * self.sps))

            if block_end > len(data_filt):
                block_end = len(data_filt)

            block = data_filt[block_start:block_end]

            if len(block) < max(32, int(np.ceil(4 * self.sps))):
                continue

            if L > 1:
                block_os = upsample(block, L, 2)
                best_t_os = _best_sampling_point_float(block_os, self.sps * L)
                best_t = best_t_os / L
            else:
                best_t = _best_sampling_point_float(block, self.sps)

            # use the center of the bootstrap block as the anchor point
            center_sym = start_sym + 0.5 * block_symbols
            center_sample = block_start + best_t + 0.5 * block_symbols * self.sps

            anchor_symbol_idx.append(center_sym)
            anchor_sample_idx.append(center_sample)

        anchor_symbol_idx = np.asarray(anchor_symbol_idx, dtype=np.float64)
        anchor_sample_idx = np.asarray(anchor_sample_idx, dtype=np.float64)

        if len(anchor_symbol_idx) < 2:
            # fallback: constant timing from the first part
            logger.warning(
                "Too few timing anchors. Falling back to constant-sps timing."
            )
            first_block_end = int(np.ceil((block_symbols + 2) * self.sps))
            first_block = data_filt[:first_block_end]

            if L > 1:
                first_block_os = upsample(first_block, L, 2)
                best_t_os = _best_sampling_point_float(first_block_os, self.sps * L)
                tau0 = best_t_os / L
            else:
                tau0 = _best_sampling_point_float(first_block, self.sps)

            a = tau0
            b = self.sps
        else:
            a, b = _fit_timing_anchors(
                anchor_symbol_idx,
                anchor_sample_idx,
                fit_method=self.fit_method,
            )

        # constrain the fitted slope around nominal sps
        b_min = self.sps * (1.0 - self.max_relative_sps_deviation)
        b_max = self.sps * (1.0 + self.max_relative_sps_deviation)
        b = float(np.clip(b, b_min, b_max))

        # diagnostics
        fitted_anchor = a + b * anchor_symbol_idx
        if len(anchor_symbol_idx) >= 2:
            residual_std = np.std(anchor_sample_idx - fitted_anchor)
            logger.info(
                "LS timing fit: a=%.6f, b=%.9f (nominal sps=%.9f), anchor residual std=%.6f samples",
                a, b, self.sps, residual_std
            )
            # print("LS timing fit:")
            # print("  intercept a =", a)
            # print("  slope b =", b)
            # print("  nominal sps =", self.sps)
            # print("  anchor residual std [samples] =", residual_std)

        # 3) build the full fitted grid
        symbol_idx = np.arange(self.num_symbols, dtype=np.float64)
        sampled_grid = a + b * symbol_idx

        # keep grid valid for Farrow
        sampled_grid = np.clip(sampled_grid, 1.0, len(data_filt) - 3.000001)

        # 4) final sampling with cubic Farrow interpolation
        sampled_symbols = _sample_grid_farrow_array(
            data_filt.real.astype(np.float32),
            data_filt.imag.astype(np.float32),
            sampled_grid.astype(np.float64),
        )

        # split into subframes, like the other estimators
        result = []
        for i in range(0, self.num_symbols, self.subframe_length):
            subframe = sampled_symbols[i:i + self.subframe_length]
            if len(subframe) > 0:
                result.append(subframe)

        return result, sampled_grid


@njit
def _sample_grid_farrow_array(
    data_real: np.ndarray,
    data_imag: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Sample a complex signal on a non-uniform fractional grid using the
    existing cubic Farrow interpolator.
    """
    out = np.zeros(grid.size, dtype=np.complex64)
    nmax = len(data_real) - 3  # because _farrow_cubic uses n-1, n, n+1, n+2

    for k in range(grid.size):
        g = grid[k]

        if g < 1.0:
            g = 1.0
        elif g > nmax - 1e-9:
            g = nmax - 1e-9

        n = int(np.floor(g))
        mu = g - n

        y_real, y_imag = _farrow_cubic(data_real, data_imag, n, mu)
        out[k] = y_real + 1j * y_imag

    return out


def _fit_timing_anchors(
    anchor_symbol_idx: np.ndarray,
    anchor_sample_idx: np.ndarray,
    fit_method: str = "polyfit",
):
    """
    Fit sample_index = a + b * symbol_index from local timing anchors.

    fit_method:
      - "polyfit": ordinary least squares line fit
      - "median": robust median slope + median intercept
    """
    x = np.asarray(anchor_symbol_idx, dtype=np.float64)
    y = np.asarray(anchor_sample_idx, dtype=np.float64)

    if len(x) < 2:
        raise ValueError("Need at least two timing anchors.")

    fit_method = fit_method.lower()

    if fit_method == "polyfit":
        b, a = np.polyfit(x, y, deg=1)
        return a, b

    if fit_method == "median":
        dx = np.diff(x)
        dy = np.diff(y)

        valid = dx != 0
        if not np.any(valid):
            raise ValueError("Invalid timing anchors for median fit.")

        slopes = dy[valid] / dx[valid]
        b = np.median(slopes)
        a = np.median(y - b * x)
        return a, b

    raise ValueError(f"Unknown fit_method='{fit_method}'") 
    
def pre_compute_filters(
        shift_size: int,
        sps: float,
        roll_off: float,
        symbol_rate: float,
        adc_rate: float,
        f_pilot_1: float,
        frequency_shift: float):
    """
    Pre-compute the filters used in the timing recovery to reduce the computational load during the actual timing recovery.
    """
    _, rrc_filter = root_raised_cosine_filter(
        int(10 * sps + 2),
        roll_off,
        1 / symbol_rate,
        adc_rate,
    )
    rrc_filter = (rrc_filter[1:] / np.sqrt(sps)).astype(np.complex64)
    
    shift_up = np.exp(
        1j
        * 2
        * np.pi
        * np.arange(shift_size)
        * (f_pilot_1 - frequency_shift)
        / adc_rate
    ).astype(np.complex64)

    return rrc_filter, shift_up


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
        P_initial: float,
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
    P = P_initial

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

        tau_next += sps_next

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


def pulse_sampling(
        data: np.ndarray,
        sps: float,
        num_symbols: int,
    ):
    """
    Sample the signal using pulse sampling, which consists in convolving the signal with a rectangular pulse of width equal to the symbol period, and then sampling at the symbol rate. 
    This is equivalent to integrating the signal over each symbol period, which can be more robust to noise than sampling at a single point.
    """
    sps = int(sps)
    hn = (
        (1/sps) * np.ones(sps)
    )
    pulse_symbols = lfilter(hn, [1], data)[sps-1::sps][:num_symbols]
    assert len(pulse_symbols) == num_symbols, f"Expected {num_symbols} symbols, got {len(pulse_symbols)}"
    return pulse_symbols