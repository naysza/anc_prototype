from __future__ import annotations

import numpy as np


def solve_wiener(
    R: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    """
    Solve the Wiener-Hopf equation:

        R w_opt = p

    using a numerical linear-system solver.
    """

    R = np.asarray(R, dtype=float)
    p = np.asarray(p, dtype=float)

    if R.ndim != 2:
        raise ValueError("R must be a two-dimensional matrix.")

    if R.shape[0] != R.shape[1]:
        raise ValueError("R must be square.")

    if p.ndim == 2 and p.shape[1] == 1:
        p_vector = p[:, 0]
    elif p.ndim == 1:
        p_vector = p
    else:
        raise ValueError(
            "p must have shape (M,) or (M, 1)."
        )

    if R.shape[0] != p_vector.size:
        raise ValueError(
            "R and p dimensions are incompatible."
        )

    if not np.all(np.isfinite(R)):
        raise ValueError("R must contain only finite values.")

    if not np.all(np.isfinite(p_vector)):
        raise ValueError("p must contain only finite values.")

    try:
        w_opt = np.linalg.solve(R, p_vector)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "Wiener correlation matrix R is singular or ill-conditioned."
        ) from exc

    return np.asarray(w_opt, dtype=float)


def estimate_signal(
    x: np.ndarray,
    w_opt: np.ndarray,
) -> np.ndarray:
    """
    Estimate d[n] using the causal FIR Wiener filter:

        d_hat[n] = w^T x_n

    where:

        x_n = [x[n], x[n-1], ..., x[n-M+1]]
    """

    x = np.asarray(x, dtype=float)
    w_opt = np.asarray(w_opt, dtype=float).reshape(-1)

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")

    if w_opt.ndim != 1:
        raise ValueError("w_opt must be one-dimensional.")

    if x.size == 0:
        raise ValueError("x cannot be empty.")

    if w_opt.size == 0:
        raise ValueError("w_opt cannot be empty.")

    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values.")

    if not np.all(np.isfinite(w_opt)):
        raise ValueError("w_opt contains non-finite values.")

    n_samples = x.size
    filter_length = w_opt.size

    estimate = np.zeros(n_samples, dtype=float)

    for n in range(n_samples):
        for k in range(filter_length):
            index = n - k

            if index >= 0:
                estimate[n] += w_opt[k] * x[index]

    return estimate


def wiener_filter(
    R: np.ndarray,
    p: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience function:

        R,p → w_opt → d_hat
    """

    w_opt = solve_wiener(R, p)

    d_hat = estimate_signal(
        x,
        w_opt,
    )

    return w_opt, d_hat