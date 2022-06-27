from typing import Tuple

import numpy as np
from numpy import ndarray, polyfit
from scipy.fft import rfft, rfftfreq
from scipy.signal import (correlate, find_peaks, hilbert, peak_prominences,
                          peak_widths)
from scipy.stats import linregress, stats, variation


def identity(data: ndarray) -> ndarray:
    return data


def normalizeArray(data: ndarray) -> ndarray:
    """
    Normalize data in array, i.e. scale data such that it fits between -1 and 1, inclusively.

    :param data: array of data to normalize
    """
    data_abs = np.abs(data)
    data_maxabs = data_abs.max()
    data_normalized = data / data_maxabs

    return data_normalized


def normalizeOverAxes(data: ndarray, axes: Tuple[int]) -> ndarray:
    """
    Normalize data over given axes.

    :param data: array of data to normalize
    :param axes: indicies of axes (in data) to normalize over
    """
    if len(axes) >= 1:
        axes = tuple(sorted(axes))

        data_abs = np.abs(data)
        data_max = np.max(data_abs, axis=axes)

        if isinstance(data_max, ndarray):
            data_max[data_max == 0] = 1

        """data_max_inds = []
        for dimension_index, dimension_size in enumerate(data.shape):
            if dimension_index in axes:
                new_inds = np.newaxis
            else:
                new_inds = np.arange(data.shape[dimension_index])
            data_max_inds.append(new_inds)
        data_max_inds = tuple(data_max_inds)"""

        data_shape_array = np.array(data.shape)
        data_ndim = data.ndim
        normalize_ndim = len(axes)
        nonnormalize_ndim = data_ndim - normalize_ndim

        other_axes = tuple([
            axis
            for axis in range(data_ndim)
            if axis not in axes
        ])

        shape_missing = tuple(data_shape_array[list(axes)])
        tile_reps = tuple([
            *shape_missing,
            *np.ones(nonnormalize_ndim, dtype=int)
        ])
        data_max_shaped = np.tile(data_max, tile_reps)

        dimension_map = np.array([*axes, *other_axes])
        dimension_map_reverse = np.zeros(data_ndim, dtype=int)
        dimension_map_reverse[dimension_map] = np.arange(data_ndim)
        transpose_axes = tuple(dimension_map_reverse)
        data_max_shaped = np.transpose(data_max_shaped, axes=transpose_axes)

        data_normalized = np.divide(data, data_max_shaped)
    else:
        data_normalized = data

    return data_normalized


def oscillationFrequency(
    data: ndarray,
    times: ndarray,
    calculation_method: str = "maxima_separation",
    condensing_method: str = "average"
) -> float:
    """
    Calculate oscillation frequency for array.

    :param data: array to calculate frequency for
    :param times: array of times corresponding to :paramref:`~CustomMath.oscillationFrequency.data`
    :param calculation_method: method used to calculate frequencies.
        "maxima_separation" uses peak-to-peak separation of waveform.
        "minima_separation" uses trough-to-trough separation of waveform.
        "extrema_separation" uses peaks and troughs separation of waveform.
    :param condensing_method: method used to "average" frequencies.
        "average" uses arithmetic mean of frequencies.
        "maximum" uses maximum of frequencies.
        "minimum" uses minium of frequencies.
        "initial" uses first frequency in frequencies.
        "final" uses last frequency in frequencies.
    """
    calculation_method = calculation_method.lower()
    condensing_method = condensing_method.lower()

    if "separation" in calculation_method:
        if "max" in calculation_method or "min" in calculation_method:
            def time_to_frequency(initial_time: float, final_time: float) -> float:
                return 1 / (final_time - initial_time)
        elif "extrema" in calculation_method:
            def time_to_frequency(initial_time: float, final_time: float) -> float:
                return 0.5 / (final_time - initial_time)
        else:
            raise ValueError("separation method must include maxima, minima, xor extrema")

        extrema_indicies = np.array([], dtype=np.int32)
        if "max" in calculation_method or "extrema" in calculation_method:
            maxima_indicies = find_peaks(data)[0]
            extrema_indicies = np.append(extrema_indicies, maxima_indicies)
        if "min" in calculation_method or "extrema" in calculation_method:
            minima_indicies = find_peaks(-data)[0]
            extrema_indicies = np.append(extrema_indicies, minima_indicies)
        extrema_indicies.sort()

        extrema_times = times[extrema_indicies]
        frequencies = time_to_frequency(extrema_times[0:-1], extrema_times[1:])
    else:
        raise ValueError("invalid calculation method")

    if frequencies.size >= 1:
        if condensing_method == "average":
            frequency = np.mean(frequencies)
        elif condensing_method == "maximum":
            frequency = np.amax(frequencies)
        elif condensing_method == "minimum":
            frequency = np.amin(frequencies)
        elif condensing_method == "initial":
            frequency = frequencies[0]
        elif condensing_method == "final":
            frequency = frequencies[-1]
        else:
            raise ValueError("invalid condensing method")
    else:
        frequency = 0

    return frequency


def holderMean(
    data: ndarray,
    order: int = 1
) -> float:
    """
    Calculate Holder mean for array.

    :param data: array to calculate Holder mean for
    :param order: order of Holder mean
    """
    if order == 1:
        mean = np.mean(data)
    elif order == 2:
        mean = np.sqrt(np.mean(data**2))
    elif order == 0:
        mean = stats.gmean(data)
    # elif order == -1: mean = stats.hmean(results)
    elif np.isinf(order):
        if np.sign(order) == 1:
            mean = np.amax(data)
        elif np.sign(order) == -1:
            mean = np.amin(data)
    else:
        mean = np.mean(data**order)**(1 / order)

    return mean


def arithmeticMean(data: ndarray) -> float:
    """
    Calculate arithmetic mean for array.

    :param data: array to calculate mean for
    """
    mean = np.mean(data)
    return mean


def geometricMean(data: ndarray):
    """
    Calculate geometric mean for array.

    :param data: array to calculate mean for
    """
    mean = stats.gmean(data)
    return mean


def absoluteMinimum(data: ndarray):
    """
    Calculate absolute minimum for array.

    :param data: array to calculate max for
    """
    absolute_minimum = np.min(np.abs(data))
    return absolute_minimum


def absoluteMaximum(data: ndarray):
    """
    Calculate absolute maximum for array.

    :param data: array to calculate max for
    """
    absolute_maximum = np.max(np.abs(data))
    return absolute_maximum


def rootMeanSquare(data: ndarray) -> float:
    """
    Calculate root-mean-square for array.

    :param data: array to calculate RMS for
    """
    rms = np.sqrt(np.mean(data**2))
    return rms


def sumSquaresError(data: ndarray, data_compare: ndarray) -> ndarray:
    """
    Get residual sum of squares (RSS) between data and comparison.

    :param data: vector of points to compare to comparison data (e.g. fit data)
    :param data_compare: vector of points to compare to data
    """
    rss = np.sum((data - data_compare) ** 2)
    return rss


def standardDeviation(data: ndarray) -> float:
    """
    Calculate standard deviation for array.

    :param data: array to calculate deviation for
    """
    standard_deviation = np.std(data)
    return standard_deviation


def relativeStandardDeviation(data: ndarray) -> float:
    """
    Calculate relative standard deviation for array.

    :param data: array to calculate deviation for
    """
    rsd = variation(data)
    return rsd


def fourierTransform(data: ndarray) -> ndarray:
    """
    Calculate Fourier tranform for array.

    :param data: array to calculate Fourier transform for
    """
    fourier_data = rfft(data)
    abs_fourier_data = abs(fourier_data)
    return abs_fourier_data


def fourierFrequencies(times: ndarray) -> ndarray:
    """
    Calculate Fourier frequencies from times.

    :param times: array of times to calculate frequencies from
    """
    times_size = times.size

    initial_time = times[0]
    final_time = times[-1]
    time_resolution = (final_time - initial_time) / (times_size - 1)
    frequencies = rfftfreq(times_size, time_resolution)

    return frequencies


def correlationTransform(data: ndarray) -> ndarray:
    """
    Get autocorrelation function of given data.

    :param data: array for function to retrieve autocorrelation from
    """
    data_size = data.size
    correlation = correlate(
        data,
        data,
        mode="same"
    )[data_size // 2:]

    return correlation


def correlationLags(times: ndarray) -> ndarray:
    """
    Get lag times association with autocorrelation function.

    :param times: array of times to retrieve lag times from
    """
    times_size = times.size
    """lags = signal.correlation_lags(
        times_size,
        times_size,
        mode="same"
    )[times_size // 2:]"""

    lag_times = times[:(times_size + 1) // 2] - times[0]
    return lag_times


def analyticSignal(data: ndarray) -> ndarray:
    mean_data = np.mean(data)
    centered_data = data - mean_data
    analytic_signal = hilbert(centered_data)
    return analytic_signal


def instantaneousAmplitude(data: ndarray) -> ndarray:
    analytic_signal = analyticSignal(data)
    instantaneous_amplitude = np.abs(analytic_signal)
    return instantaneous_amplitude


def instantaneousPhase(data: ndarray) -> ndarray:
    analytic_signal = analyticSignal(data)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    return instantaneous_phase


def instantaneousFrequency(
    data: ndarray,
    times: ndarray
) -> ndarray:
    instantaneous_phase = instantaneousPhase(data)
    instantaneous_frequency = np.diff(instantaneous_phase) / (2 * np.pi * (times[1:] - times[:-1]))

    return instantaneous_frequency


def interpolateMidpoint(data: ndarray) -> ndarray:
    midpoint_data = (data[1:] + data[:-1]) / 2
    return midpoint_data


def phaseDifference(data: ndarray) -> ndarray:
    """
    Get phase differences between two quantities.

    :param data: 2D numpy array.
        First index, taking values of either 0 or 1, gives quantity within array.
        Second index gives value of quantity at corresponding time.
    :returns: array of phase difference; second quantity minus first quantity
    """
    phase1 = instantaneousPhase(data[0])
    phase2 = instantaneousPhase(data[1])
    phase_difference = phase2 - phase1
    return phase_difference


def imaginaryExponentiation(phase: ndarray) -> ndarray:
    imaginary_exponent = np.exp(phase * 1j)
    return imaginary_exponent


def phaseLockingValue(data: ndarray) -> ndarray:
    """
    Get phase-locking values between two quantities.

    :param data: 2D numpy array.
        First index, taking values of either 0 or 1, gives quantity within array.
        Second index gives value of quantity at corresponding time.
    :returns: array of complex phase-locking values, i.e. exp(i*theta2-theta1)
    """
    assert data.ndim == 2

    phase_difference = phaseDifference(data)
    imaginary_exponent = imaginaryExponentiation(phase_difference)
    return imaginary_exponent


def complexMagnitude(data: ndarray) -> ndarray:
    magnitude = np.absolute(data)
    return magnitude


def complexPhase(data: ndarray) -> ndarray:
    phase = np.angle(data)
    return phase


def realPart(data: ndarray) -> ndarray:
    real_part = np.real(data)
    return real_part


def imaginaryPart(data: ndarray) -> ndarray:
    imaginary_part = np.imag(data)
    return imaginary_part


def linearIntercept(y: ndarray, parameters: ndarray = None) -> ndarray:
    """
    Get y-intercept b of linear regression y=mx+b.

    :param y: 1D array of data for y-axis of plot
    :param parameters: 1D array of data for x-axis of plot.
        Defaults to natural numbers (0 inclusive), up to size of y.
    """
    if parameters is None:
        parameters = np.arange(y.size)

    linear_regression = linregress(parameters, y)
    intercept = linear_regression[1]

    return intercept


def linearSlope(y: ndarray, parameters: ndarray = None) -> ndarray:
    """
    Get slope m of linear regression y=mx+b.

    :param y: 1D array of data for y-axis of plot
    :param parameters: 1D array of data for x-axis of plot.
        Defaults to natural numbers (0 inclusive), up to size of y.
    """
    if parameters is None:
        parameters = np.arange(y.size)

    linear_regression = linregress(parameters, y)
    slope = linear_regression[0]
    return slope


def linearSlopeTimesIntercept(y: ndarray, parameters: ndarray = None) -> ndarray:
    """
    Get m*b of linear regression y=mx+b.

    :param y: 1D array of data for y-axis of plot
    :param parameters: 1D array of data for x-axis of plot.
        Defaults to natural numbers (0 inclusive), up to size of y.
    """
    if parameters is None:
        parameters = np.arange(y.size)

    linear_regression = linregress(parameters, y)
    slope, intercept = linear_regression[0:2]

    multiplied = slope * intercept
    return multiplied


def quadraticHighestOrder(y: ndarray, parameters: ndarray = None) -> ndarray:
    """
    Get highest-order coefficient a2 of regression y=a2*x^2+a1*x+a0.

    :param y: 1D array of data for y-axis of plot
    :param parameters: 1D array of data for x-axis of plot.
        Defaults to natural numbers (0 inclusive), up to size of y.
    """
    if parameters is None:
        parameters = np.arange(y.size)

    quadratic_coefficients = polyfit(parameters, y, 2)
    highest_order = quadratic_coefficients[0]
    return highest_order


def globalPeakIsolation(y: ndarray, parameters: ndarray = None) -> ndarray:
    """
    Get peak width/isolation for global maximum.

    :param y: 1D array of data for y-axis of plot
    :param parameters: 1D array of data for x-axis of plot.
        Defaults to natural numbers (0 inclusive), up to size of y.
    """
    if parameters is None:
        parameters = np.arange(y.size)

    max_index = np.argmax(y)

    isolations = peak_widths(y, [max_index])
    global_isolation = isolations[0][0]
    return global_isolation


def globalPeakProminence(y: ndarray, parameters: ndarray = None) -> ndarray:
    """
    Get peak prominence for global maximum.

    :param y: 1D array of data for y-axis of plot
    :param parameters: 1D array of data for x-axis of plot.
        Defaults to natural numbers (0 inclusive), up to size of y.
    """
    if parameters is None:
        parameters = np.arange(y.size)

    max_index = np.argmax(y)

    prominences = peak_prominences(y, [max_index])
    global_prominence = prominences[0][0]
    return global_prominence
