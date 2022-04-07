import numpy as np
from numpy import ndarray
from scipy.fft import rfft, rfftfreq
from scipy.signal import correlate, find_peaks, hilbert
from scipy.stats import stats


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

    modulus = np.absolute(mean)
    return modulus


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


def phaseDifference(data1, data2) -> ndarray:
    phase1 = instantaneousPhase(data1)
    phase2 = instantaneousPhase(data2)
    phase_difference = phase2 - phase1
    return phase_difference


def imaginaryExponentiation(phase: ndarray) -> ndarray:
    imaginary_exponent = np.exp(phase * 1j)
    return imaginary_exponent


def phaseLockingValue(data1, data2):
    assert data1.shape == data2.shape

    phase_difference = phaseDifference(data1, data2)
    imaginary_exponent = imaginaryExponentiation(phase_difference)
    return imaginary_exponent
