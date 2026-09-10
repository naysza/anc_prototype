from __future__ import annotations

import numpy as np
from scipy.linalg import toeplitz

from anc.core.interfaces import Signal


def _validate_max_lag(max_lag: int, n_samples: int) -> None:
    if not isinstance(max_lag, (int, np.integer)):
        raise TypeError("max_lag must be an integer.")

    if max_lag <= 0:
        raise ValueError("max_lag must be greater than zero.")

    if max_lag > n_samples:
        raise ValueError(
            "max_lag cannot be greater than the signal length."
        )


def calculate_autocorrelation(
    x: Signal,
    max_lag: int,
) -> np.ndarray:
    """
    Calculate biased sample autocorrelation.

    Rxx[k] = E{x[n]x[n-k]}

    Returns lags:
        k = 0, 1, ..., max_lag - 1
    """

    if not isinstance(x, Signal):
        raise TypeError("x must be a Signal instance.")

    data = x.data
    n_samples = data.size

    _validate_max_lag(max_lag, n_samples)

    correlation = np.correlate(
        data,
        data,
        mode="full",
    )

    center = n_samples - 1

    rxx = correlation[
        center:center + max_lag
    ]

    return rxx / n_samples


def calculate_crosscorrelation(
    x: Signal,
    d: Signal,
    max_lag: int,
) -> np.ndarray:
    """
    Calculate biased sample cross-correlation using the
    Wiener-filter convention:

        Rxd[k] = E{x[n-k] d[n]}

    Returns:
        Rxd[0], Rxd[1], ..., Rxd[max_lag-1]

    This convention is consistent with the causal FIR model:

        d[n] = sum_k w[k] x[n-k]

    and the Wiener-Hopf equation:

        R w = p
    """

    if not isinstance(x, Signal):
        raise TypeError("x must be a Signal instance.")

    if not isinstance(d, Signal):
        raise TypeError("d must be a Signal instance.")

    if x.n_samples != d.n_samples:
        raise ValueError(
            "Reference and desired signals must have the same length."
        )

    if not np.isclose(x.fs, d.fs):
        raise ValueError(
            "Reference and desired signals must have the same sampling rate."
        )

    n_samples = x.n_samples

    _validate_max_lag(max_lag, n_samples)

    # np.correlate(d, x) gives:
    #
    #   sum_n d[n] x[n-k]
    #
    # at positive lag k.
    #
    # This is exactly the Wiener convention:
    #
    #   Rxd[k] = E{x[n-k] d[n]}
    correlation = np.correlate(
        d.data,
        x.data,
        mode="full",
    )

    center = n_samples - 1

    rxd = correlation[
        center:center + max_lag
    ]

    return rxd / n_samples

def construct_wiener_matrix(
    rxx: np.ndarray,
) -> np.ndarray:
    """
    Construct the Wiener correlation matrix R.

    R is an M x M symmetric Toeplitz matrix.
    """

    rxx = np.asarray(rxx, dtype=float)

    if rxx.ndim != 1:
        raise ValueError("rxx must be one-dimensional.")

    if rxx.size == 0:
        raise ValueError("rxx cannot be empty.")

    if not np.all(np.isfinite(rxx)):
        raise ValueError("rxx must contain finite values.")

    return toeplitz(rxx)


def construct_cross_vector(
    rxd: np.ndarray,
) -> np.ndarray:
    """
    Construct the Wiener cross-correlation vector p.

    Returns shape:
        (M, 1)
    """

    rxd = np.asarray(rxd, dtype=float)

    if rxd.ndim != 1:
        raise ValueError("rxd must be one-dimensional.")

    if rxd.size == 0:
        raise ValueError("rxd cannot be empty.")

    if not np.all(np.isfinite(rxd)):
        raise ValueError("rxd must contain finite values.")

    return rxd.reshape(-1, 1)