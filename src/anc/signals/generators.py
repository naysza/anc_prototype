# src/anc/signals/generators.py

"""
Signal generation utilities for the ANC project.

This module converts a signal specification (configuration dictionary)
into a common Signal object containing a discrete-time sequence.

Supported signal types:
    - sinusoid
    - multi_tone
    - white_noise
    - colored_noise
    - impulse
    - burst

The generated signal is represented as:

    Signal(data=<numpy array>, fs=<sampling frequency>, name=<name>)

The module is intentionally limited to signal generation.
Filtering, correlation, Wiener estimation, MSE calculation, and
experiment orchestration belong to other modules.
"""

from __future__ import annotations

import warnings

import numpy as np

from anc.core.interfaces import Signal


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def _make_sample_indices(fs: float, duration: float) -> np.ndarray:
    """
    Create discrete-time sample indices.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.

    duration : float
        Signal duration in seconds.

    Returns
    -------
    np.ndarray
        Integer sample indices:

            n = 0, 1, 2, ..., N-1

        where approximately:

            N = fs * duration
    """

    if fs <= 0:
        raise ValueError("Sampling frequency 'fs' must be greater than 0.")

    if duration <= 0:
        raise ValueError("Signal 'duration' must be greater than 0.")

    n_samples = int(round(fs * duration))

    if n_samples <= 0:
        raise ValueError(
            "The combination of 'fs' and 'duration' produces no samples."
        )

    return np.arange(n_samples)


def _validate_frequency(frequency: float, fs: float) -> None:
    """
    Validate a frequency parameter.

    Frequencies at or above Nyquist are allowed because aliasing may be
    intentionally studied in an experiment. A warning is issued rather
    than rejecting the configuration.
    """

    if frequency < 0:
        raise ValueError("Frequency must be non-negative.")

    nyquist = fs / 2.0

    if frequency > nyquist:
        warnings.warn(
            f"Frequency {frequency} Hz is above the Nyquist frequency "
            f"({nyquist} Hz). The generated discrete-time signal will "
            f"contain aliasing.",
            UserWarning,
            stacklevel=2,
        )


def _get_required(config: dict, key: str):
    """
    Return a required configuration value.

    Raises
    ------
    ValueError
        If the required key is missing.
    """

    if key not in config:
        raise ValueError(
            f"Missing required signal configuration parameter: '{key}'."
        )

    return config[key]


# ---------------------------------------------------------------------
# Individual signal generators
# ---------------------------------------------------------------------

def _generate_sinusoid(
    n: np.ndarray,
    fs: float,
    amplitude: float,
    frequency: float,
) -> np.ndarray:
    """
    Generate a discrete-time sinusoidal signal.

    Mathematical representation:

        x[n] = A sin(2*pi*f*n/fs)

    Parameters
    ----------
    n : np.ndarray
        Discrete-time sample indices.

    fs : float
        Sampling frequency in Hz.

    amplitude : float
        Sinusoid amplitude.

    frequency : float
        Sinusoid frequency in Hz.

    Returns
    -------
    np.ndarray
        Generated discrete-time sinusoid.
    """

    _validate_frequency(frequency, fs)

    return amplitude * np.sin(
        2.0 * np.pi * frequency * n / fs
    )


def _generate_multi_tone(
    n: np.ndarray,
    fs: float,
    amplitudes: list[float],
    frequencies: list[float],
) -> np.ndarray:
    """
    Generate a sum of sinusoidal tones.

    Mathematical representation:

        x[n] = sum_i A_i sin(2*pi*f_i*n/fs)

    Parameters
    ----------
    n : np.ndarray
        Discrete-time sample indices.

    fs : float
        Sampling frequency in Hz.

    amplitudes : list[float]
        Amplitude of each tone.

    frequencies : list[float]
        Frequency of each tone in Hz.

    Returns
    -------
    np.ndarray
        Generated multi-tone signal.
    """

    if len(amplitudes) != len(frequencies):
        raise ValueError(
            "'amplitudes' and 'frequencies' must contain the same "
            "number of values."
        )

    if len(frequencies) == 0:
        raise ValueError(
            "'frequencies' and 'amplitudes' cannot be empty."
        )

    signal = np.zeros(len(n), dtype=float)

    for amplitude, frequency in zip(amplitudes, frequencies):
        signal += _generate_sinusoid(
            n=n,
            fs=fs,
            amplitude=float(amplitude),
            frequency=float(frequency),
        )

    return signal


def _generate_white_noise(
    n_samples: int,
    amplitude: float,
    seed: int | None,
) -> np.ndarray:
    """
    Generate zero-mean Gaussian white noise.

    Parameters
    ----------
    n_samples : int
        Number of samples.

    amplitude : float
        Standard deviation of the generated Gaussian noise.

    seed : int or None
        Random seed used for reproducibility.

    Returns
    -------
    np.ndarray
        White-noise sequence.
    """

    if amplitude < 0:
        raise ValueError("Noise amplitude must be non-negative.")

    rng = np.random.default_rng(seed)

    return amplitude * rng.standard_normal(n_samples)


def _generate_colored_noise(
    n_samples: int,
    amplitude: float,
    seed: int | None,
    color: str = "pink",
) -> np.ndarray:
    """
    Generate an approximately colored-noise sequence.

    The PS requires support for colored noise but does not prescribe
    one specific coloring method. This implementation uses frequency-
    domain spectral shaping.

    Supported colors:
        - pink
        - brown

    Parameters
    ----------
    n_samples : int
        Number of samples.

    amplitude : float
        Scaling factor applied to the resulting noise.

    seed : int or None
        Random seed.

    color : str
        Type of colored noise.

    Returns
    -------
    np.ndarray
        Colored-noise sequence.
    """

    if amplitude < 0:
        raise ValueError("Noise amplitude must be non-negative.")

    color = color.lower()

    if color not in {"pink", "brown"}:
        raise ValueError(
            f"Unsupported colored-noise type '{color}'. "
            "Supported values are 'pink' and 'brown'."
        )

    rng = np.random.default_rng(seed)

    # Start with white Gaussian noise.
    white = rng.standard_normal(n_samples)

    # Transform into frequency domain.
    spectrum = np.fft.rfft(white)

    frequencies = np.fft.rfftfreq(n_samples)

    # Avoid division by zero at DC.
    frequencies[0] = frequencies[1] if len(frequencies) > 1 else 1.0

    if color == "pink":
        # Approximately 1/f power behavior.
        shaping = 1.0 / np.sqrt(frequencies)

    else:  # brown
        # Approximately 1/f^2 power behavior.
        shaping = 1.0 / frequencies

    shaped_spectrum = spectrum * shaping

    colored = np.fft.irfft(
        shaped_spectrum,
        n=n_samples,
    )

    # Normalize so that the requested amplitude corresponds to
    # approximately unit standard deviation before scaling.
    std = np.std(colored)

    if std > 0:
        colored = colored / std

    return amplitude * colored


def _generate_impulse(
    n_samples: int,
    amplitude: float,
) -> np.ndarray:
    """
    Generate a discrete-time impulse.

    Mathematical representation:

        x[n] = A, n = 0
        x[n] = 0, otherwise

    Parameters
    ----------
    n_samples : int
        Number of samples.

    amplitude : float
        Impulse amplitude.

    Returns
    -------
    np.ndarray
        Impulse sequence.
    """

    signal = np.zeros(n_samples, dtype=float)

    signal[0] = amplitude

    return signal


def _generate_burst(
    n: np.ndarray,
    fs: float,
    amplitude: float,
    frequency: float,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    """
    Generate a sinusoidal burst.

    The sinusoid exists only between start_time and end_time.

    Parameters
    ----------
    n : np.ndarray
        Discrete-time sample indices.

    fs : float
        Sampling frequency in Hz.

    amplitude : float
        Burst amplitude.

    frequency : float
        Burst frequency in Hz.

    start_time : float
        Burst start time in seconds.

    end_time : float
        Burst end time in seconds.

    Returns
    -------
    np.ndarray
        Burst signal.
    """

    if start_time < 0:
        raise ValueError("'start_time' must be non-negative.")

    if end_time <= start_time:
        raise ValueError(
            "'end_time' must be greater than 'start_time'."
        )

    _validate_frequency(frequency, fs)

    time = n / fs

    mask = (time >= start_time) & (time < end_time)

    signal = np.zeros(len(n), dtype=float)

    signal[mask] = amplitude * np.sin(
        2.0 * np.pi * frequency * time[mask]
    )

    return signal


# ---------------------------------------------------------------------
# Public signal-generation interface
# ---------------------------------------------------------------------

def generate_signal(config: dict) -> Signal:
    """
    Generate a discrete-time signal from a configuration dictionary.

    This is the main public interface of the module.

    The caller provides a signal specification rather than an existing
    sequence. The function creates the numerical sequence and returns
    it using the common Signal representation.

    Parameters
    ----------
    config : dict
        Signal-generation configuration.

    Required parameters
    -------------------
    type : str
        Signal type.

    fs : float
        Sampling frequency in Hz.

    duration : float
        Signal duration in seconds.

    Optional parameters
    -------------------
    amplitude : float
        Signal amplitude. Defaults to 1.0.

    frequency : float
        Frequency for sinusoid/burst signals.

    frequencies : list
        Frequencies for multi-tone signals.

    amplitudes : list
        Amplitudes for multi-tone signals.

    seed : int or None
        Random seed for noise signals.

    color : str
        Colored-noise type.

    start_time : float
        Start time for a burst.

    end_time : float
        End time for a burst.

    name : str
        Name stored in the Signal object.

    Returns
    -------
    Signal
        Generated discrete-time signal.

    Examples
    --------
    Sinusoid:

        config = {
            "type": "sinusoid",
            "fs": 1000,
            "duration": 1.0,
            "frequency": 100,
            "amplitude": 1.0,
            "name": "reference_sine",
        }

        signal = generate_signal(config)

    White noise:

        config = {
            "type": "white_noise",
            "fs": 1000,
            "duration": 1.0,
            "amplitude": 1.0,
            "seed": 42,
            "name": "reference_noise",
        }

        signal = generate_signal(config)
    """

    if not isinstance(config, dict):
        raise TypeError(
            "Signal configuration must be provided as a dictionary."
        )

    signal_type = _get_required(config, "type").lower()

    fs = float(_get_required(config, "fs"))
    duration = float(_get_required(config, "duration"))

    n = _make_sample_indices(
        fs=fs,
        duration=duration,
    )

    n_samples = len(n)

    amplitude = float(config.get("amplitude", 1.0))
    seed = config.get("seed", None)
    name = config.get("name", signal_type)

    # -------------------------------------------------------------
    # Select the requested signal generator.
    # -------------------------------------------------------------

    if signal_type == "sinusoid":

        frequency = float(
            _get_required(config, "frequency")
        )

        data = _generate_sinusoid(
            n=n,
            fs=fs,
            amplitude=amplitude,
            frequency=frequency,
        )

    elif signal_type == "multi_tone":

        frequencies = [
            float(value)
            for value in _get_required(config, "frequencies")
        ]

        amplitudes = config.get(
            "amplitudes",
            [amplitude] * len(frequencies),
        )

        amplitudes = [
            float(value)
            for value in amplitudes
        ]

        data = _generate_multi_tone(
            n=n,
            fs=fs,
            amplitudes=amplitudes,
            frequencies=frequencies,
        )

    elif signal_type == "white_noise":

        data = _generate_white_noise(
            n_samples=n_samples,
            amplitude=amplitude,
            seed=seed,
        )

    elif signal_type == "colored_noise":

        color = config.get("color", "pink")

        data = _generate_colored_noise(
            n_samples=n_samples,
            amplitude=amplitude,
            seed=seed,
            color=color,
        )

    elif signal_type == "impulse":

        data = _generate_impulse(
            n_samples=n_samples,
            amplitude=amplitude,
        )

    elif signal_type == "burst":

        frequency = float(
            _get_required(config, "frequency")
        )

        start_time = float(
            config.get("start_time", 0.0)
        )

        end_time = float(
            config.get("end_time", duration)
        )

        data = _generate_burst(
            n=n,
            fs=fs,
            amplitude=amplitude,
            frequency=frequency,
            start_time=start_time,
            end_time=end_time,
        )

    else:
        raise ValueError(
            f"Unsupported signal type '{signal_type}'. "
            "Supported types are: "
            "sinusoid, multi_tone, white_noise, "
            "colored_noise, impulse, burst."
        )

    # -------------------------------------------------------------
    # Return the common Signal representation.
    # -------------------------------------------------------------

    return Signal(
        data=np.asarray(data, dtype=float),
        fs=fs,
        name=name,
    )