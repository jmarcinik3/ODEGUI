import numpy as np
from numpy import ndarray
from scipy import fft, signal
from scipy.stats import stats


def oscillationFrequency(
    data: ndarray,
    times: ndarray,
    calculation_method: str = "autocorrelation",
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
        "autocorrelation" uses peak-to-peak separation for autocorrelation of waveform.
    :param condensing_method: method used to "average" frequencies.
        "average" uses arithmetic mean of frequencies.
        "maximum" uses maximum of frequencies.
        "minimum" uses minium of frequencies.
        "initial" uses first frequency in frequencies.
        "final" uses last frequency in frequencies.
    """
    calculation_method = calculation_method.lower()
    condensing_method = condensing_method.lower()
    
    if "autocorrelation" in calculation_method:
        results_count = data.size
        autocorrelation = signal.correlate(
            data,
            data,
            mode="same"
        )[results_count // 2:]
        lags = signal.correlation_lags(
            results_count,
            results_count,
            mode="same"
        )[results_count // 2:]

        argrelmax_correlation = signal.argrelmax(autocorrelation)[0]

        if argrelmax_correlation.size >= 1:
            argrelmax_lags = lags[argrelmax_correlation]
            delta_lags = argrelmax_lags[1:] - argrelmax_lags[:-1]
            delta_time = times[1] - times[0]
            frequencies = 1 / (delta_lags * delta_time)
            
            fourier_temp = fourierTransform(autocorrelation)
            fourier_frequencies = fourierFrequencies(times)
            argrelmax_fourier = signal.argrelmax(fourier_temp)
            relmax_frequencies = fourier_frequencies[argrelmax_fourier]
        else:
            frequencies = np.array([])
    elif "separation" in calculation_method:
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
            maxima_indicies = signal.find_peaks(data)[0]
            extrema_indicies = np.append(extrema_indicies, maxima_indicies)
        if "min" in calculation_method or "extrema" in calculation_method:
            minima_indicies = signal.find_peaks(-data)[0]
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


def fourierTransform(data: ndarray) -> ndarray:
    """
    Calculate Fourier tranform for array.

    :param data: array to calculate Fourier transform for
    """
    fourier_data = fft.rfft(data)
    fourier_data = abs(fourier_data)
    return fourier_data


def fourierFrequencies(times: ndarray) -> ndarray:
    """
    Calculate Fourier frequencies from times.

    :param times: array of times to calculate frequencies from
    """
    times_count = times.size

    initial_time = times[0]
    final_time = times[-1]
    time_resolution = (final_time - initial_time) / (times_count - 1)
    frequencies = fft.rfftfreq(times_count, time_resolution)

    return frequencies


def correlationTransform(data: ndarray) -> ndarray:
    """
    Get autocorrelation function of given data.
    
    :param data: array for function to retrieve autocorrelation from
    """
    data_count = data.size
    correlation = signal.correlate(
        data,
        data,
        mode="same"
    )[data_count // 2:]
    
    return correlation


def correlationLags(times: ndarray) -> ndarray:
    """
    Get lag times association with autocorrelation function.
    
    :param times: array of times to retrieve lag times from
    """
    times_count = times.size
    """lags = signal.correlation_lags(
        times_count,
        times_count,
        mode="same"
    )[times_count // 2:]"""
    
    lag_times = times[:times_count // 2] - times[0]
    return lag_times