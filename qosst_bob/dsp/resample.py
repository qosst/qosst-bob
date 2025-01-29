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
Modules to resample data (mostly downsample) and associated functions.
"""
import numpy as np
from scipy import signal


def downsample(
    data: np.ndarray, start_point: int, downsampling_factor: float
) -> np.ndarray:
    """
    Downsample data starting with start_point and with a downsampling factor
    of downsampling_factor.

    If the downsampling_factor is an integer, it uses the standard slice method.

    Args:
        data (np.ndarray): the data to downsample.
        start_point (int): the start point: i.e. the first point in the downsample array is data[start_point].
        downsampling_factor (float): the downsampling factor: i.e. points in the downsampled arry are in the form data[start_point + k*downsampling_factor].

    Returns:
        np.ndarray: downsampled data.
    """
    if int(downsampling_factor) == downsampling_factor:
        return _downsample_int(data, start_point, int(downsampling_factor))
    return _downsample_float(data, start_point, downsampling_factor)


def _downsample_int(
    data: np.ndarray, start_point: int, downsampling_factor: int
) -> np.ndarray:
    """
    Downsample data starting with start_point and with a downsampling factor
    of downsampling_factor, in the case where downsampling_factor is an integer.

    It uses the standard slice method.

    Args:
        data (np.ndarray): the data to downsample.
        start_point (int): the start point: i.e. the first point in the downsample array is data[start_point].
        downsampling_factor (float): the downsampling factor: i.e. points in the downsampled arry are in the form data[start_point + k*downsampling_factor].

    Returns:
        np.ndarray: downsampled data.
    """
    return data[start_point::downsampling_factor]


def _downsample_float(
    data: np.ndarray, start_point: int, downsampling_factor: float
) -> np.ndarray:
    """
    Downsample by a floating factor.

    Usually the operation is to take the point start_point + k * downsampling_factor.

    However, here downsampling_factor is a float so we want to compensate by approximating
    to the nearest integer with rint.

    Hence if L denotes the length of the list, the requirement on k is that

    rint(start_point + k * downsampling_factor) < L
    start_point+k*downsampling_factor <= L-0.5
    k <= (L-0.5-start_point)/downsampling_factor

    As k is an integer

    k <= np.floor((L-0.5-start_point)/downsampling_factor)

    As k must take this last value to recover everything the arange is
    np.arange(np.floor((L-0.5-start_point)/downsampling_factor) + 1)

    Note that the rint function was not used as its behavior is weird when being exactly between
    two integers (i.e. it rounds to the nearest integer value) which is hard to take into account for every possibility.

    Instead, we use np.ceil(x-0.5) that has the same behavior as np.rint except that is always round down when
    between exactly two values.

    Args:
        data (np.ndarray): the data to downsample.
        start_point (int): the start point: i.e. the first point in the downsample array is data[start_point].
        downsampling_factor (float): the downsampling factor: i.e. points in the downsampled arry are in the form data[start_point + k*downsampling_factor].

    Returns:
        np.ndarray: downsampled data.
    """
    return data[
        np.ceil(
            start_point
            + downsampling_factor
            * np.arange(
                np.floor((len(data) - 0.5 - start_point) / downsampling_factor).astype(
                    int
                )
                + 1
            )
            - 0.5
        ).astype(int)
    ]


def best_sampling_point(data: np.ndarray, sps: float) -> int:
    """
    Find the best sampling point with maximal variance, for the
    downsampling after the matched filter.

    The best sampling point is found by testing all
    the possible sampling point and by taking the one
    with maximal variance.

    Args:
        data (np.ndarray): the data from which to find the best sampling point.
        sps (float): the samples per symbol value.

    Returns:
        int: the best sampling point.
    """

    # If sps is an integer, we use the dedicated function
    if int(sps) == sps:
        return _best_sampling_point_int(data, int(sps))
    # Otherwise we use the function with a float sps
    return _best_sampling_point_float(data, sps)


def _best_sampling_point_int(data: np.ndarray, sps: int) -> int:
    """
    Find the best sampling point with maximal variance,
    with the assumption that the sps is an integer value.

    The best sampling point is found by testing all
    the possible sampling point and by taking the one
    with maximal variance.

    Args:
        data (np.ndarray): the data from which to find the best sampling point.
        sps (int): the samples per symbol value.

    Returns:
        int: the best sampling point.
    """
    return np.argmax(np.array([np.var(data[i::sps]) for i in range(sps)])).astype(int)


def _best_sampling_point_float(data: np.ndarray, sps: float) -> int:
    """
    Find the best sampling point with maximal variance,
    with the assumption that the sps is not an integer value.

    The best sampling point is found by testing all
    the possible sampling point and by taking the one
    with maximal variance.

    Args:
        data (np.ndarray): the data from which to find the best sampling point.
        sps (float): the samples per symbol value.

    Returns:
        int: the best sampling point.
    """
    return np.argmax(
        np.array(
            [
                np.var([_downsample_float(data, i, sps)])
                for i in range(np.ceil(sps).astype(int))
            ]
        )
    ).astype(int)


def upsample(data: np.ndarray, upsampling_factor: float, order: int) -> np.ndarray:
    """
    Upsample data with an upsampling factor of upsampling_factor, and an
    interpolation of order specified by order.

    Order:
    - 0: Zero order hold.
    - 1: Linear interpolation.
    - >1: Shannon interpolation (invokes scipy.signal.resample).

    Args:
        data (np.ndarray): the data to upsample.
        upsampling_factor (float): the upsampling factor.
        order (int): the order of the interpolation.

    Returns:
        np.ndarray: upsampled data.
    """
    if order == 0:
        return _upsample_zoh(data, upsampling_factor)
    elif order == 1:
        return _upsample_linear_interpolation(data, upsampling_factor)
    else:
        return signal.resample(data, int(len(data) * upsampling_factor))


def _upsample_zoh(data: np.ndarray, upsampling_factor) -> np.ndarray:
    """
    Upsample data without interpolation (zero order hold), with an upsampling
    factor of upsampling_factor.

    Args:
        data (np.ndarray): the data to upsample.
        upsampling_factor (float): the upsampling factor.

    Returns:
        np.ndarray: upsampled data.
    """
    if int(upsampling_factor) == upsampling_factor:
        return np.repeat(data, int(upsampling_factor))
    else:
        n = len(data)
        source_index = np.arange(int(n * upsampling_factor)) / upsampling_factor
        _, index_integral = np.modf(source_index)
        index_integral = index_integral.astype(int)
        return data[index_integral]


def _upsample_linear_interpolation(data: np.ndarray, upsampling_factor) -> np.ndarray:
    """
    Upsample data with linear interpolation, with an upsampling factor of
    upsampling_factor.

    Args:
        data (np.ndarray): the data to upsample.
        upsampling_factor (float): the upsampling factor.

    Returns:
        np.ndarray: upsampled data.
    """
    n = len(data)
    source_index = np.arange(int(n * upsampling_factor)) / upsampling_factor
    index_fractional, index_integral = np.modf(source_index)
    index_integral = index_integral.astype(int)
    x0 = data[index_integral]
    x1 = data[np.minimum(index_integral + 1, n - 1)]
    return x0 + (x1 - x0) * index_fractional
