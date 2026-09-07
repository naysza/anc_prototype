from __future__ import annotations

import numpy as np


def calculate_mse(
    desired: np.ndarray,
    estimate: np.ndarray,
) -> float:
    """
    Calculate mean squared error:

        MSE = E[(d[n] - d_hat[n])^2]
    """

    desired = np.asarray(desired, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    if desired.ndim != 1 or estimate.ndim != 1:
        raise ValueError("desired and estimate must be one-dimensional.")

    if desired.size != estimate.size:
        raise ValueError(
            "desired and estimate must have the same length."
        )

    if desired.size == 0:
        raise ValueError("Signals cannot be empty.")

    if not np.all(np.isfinite(desired)):
        raise ValueError("desired contains non-finite values.")

    if not np.all(np.isfinite(estimate)):
        raise ValueError("estimate contains non-finite values.")

    error = desired - estimate

    return float(np.mean(error ** 2))


def calculate_error(
    desired: np.ndarray,
    estimate: np.ndarray,
) -> np.ndarray:
    """
    Calculate instantaneous estimation error:

        e[n] = d[n] - d_hat[n]
    """

    desired = np.asarray(desired, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    if desired.shape != estimate.shape:
        raise ValueError(
            "desired and estimate must have identical shapes."
        )

    return desired - estimate