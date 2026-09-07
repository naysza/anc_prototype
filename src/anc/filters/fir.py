"""Finite Impulse Response (FIR) filtering.

This module provides a reusable FIR filtering operation for the ANC
project.

The FIR system is defined by:

    y[n] = sum_k h[k] x[n-k]

where:
    x[n] is the input signal,
    h[k] is the FIR impulse response,
    y[n] is the filtered output.
"""

from __future__ import annotations

import numpy as np

from anc.core.interfaces import Signal


def _validate_signal(signal: Signal) -> None:
    """Validate the input signal.

    Parameters
    ----------
    signal:
        Input Signal object.

    Raises
    ------
    TypeError
        If signal is not a Signal instance.
    ValueError
        If the signal contains no samples.
    """
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal instance.")

    if signal.n_samples == 0:
        raise ValueError("signal must contain at least one sample.")


def _validate_coefficients(coefficients) -> np.ndarray:
    """Validate and convert FIR coefficients.

    Parameters
    ----------
    coefficients:
        Sequence of FIR coefficients.

    Returns
    -------
    numpy.ndarray
        One-dimensional floating-point coefficient array.

    Raises
    ------
    ValueError
        If coefficients are empty, not one-dimensional, or contain
        non-finite values.
    TypeError
        If coefficients cannot be converted to numeric values.
    """
    try:
        h = np.asarray(coefficients, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "FIR coefficients must be a numeric sequence."
        ) from exc

    if h.ndim != 1:
        raise ValueError(
            "FIR coefficients must be a one-dimensional sequence."
        )

    if h.size == 0:
        raise ValueError(
            "FIR coefficients must contain at least one coefficient."
        )

    if not np.all(np.isfinite(h)):
        raise ValueError(
            "FIR coefficients must contain only finite values."
        )

    return h


def apply_fir(signal: Signal, coefficients) -> Signal:
    """Apply an FIR filter to a Signal.

    The filtering operation is full linear convolution:

        y[n] = sum_k h[k] x[n-k]

    Parameters
    ----------
    signal:
        Input Signal object containing x[n].

    coefficients:
        One-dimensional sequence containing FIR coefficients h[k].

    Returns
    -------
    Signal
        Filtered output Signal containing y[n].

    Notes
    -----
    For an input signal of length N and an FIR filter of length M,
    the output has length:

        N + M - 1

    The sampling frequency is preserved.
    """
    _validate_signal(signal)
    h = _validate_coefficients(coefficients)

    # Perform full linear convolution.
    y = np.convolve(signal.data, h, mode="full")

    # Create a new Signal while preserving the input sampling frequency.
    output_name = f"{signal.name}_fir"

    return Signal(
        data=np.asarray(y, dtype=float),
        fs=signal.fs,
        name=output_name,
    )