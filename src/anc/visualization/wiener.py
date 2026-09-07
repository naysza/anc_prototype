from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_wiener_coefficients(
    w_opt: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Plot Wiener-optimal coefficients.
    """

    w_opt = np.asarray(w_opt)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(9, 5),
    )

    indices = np.arange(w_opt.size)

    ax.stem(
        indices,
        w_opt,
        basefmt=" ",
    )

    ax.set_title("Wiener/MMSE Optimal Filter Coefficients")
    ax.set_xlabel("Coefficient index")
    ax.set_ylabel("Coefficient value")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_estimation(
    desired: np.ndarray,
    estimate: np.ndarray,
    output_path: str | Path,
    max_samples: int = 1000,
) -> None:
    """
    Plot desired signal against Wiener estimate.
    """

    desired = np.asarray(desired)
    estimate = np.asarray(estimate)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(
        desired.size,
        estimate.size,
        max_samples,
    )

    indices = np.arange(n)

    fig, ax = plt.subplots(
        figsize=(10, 5),
    )

    ax.plot(
        indices,
        desired[:n],
        label="Desired d[n]",
    )

    ax.plot(
        indices,
        estimate[:n],
        label="Estimated d_hat[n]",
        alpha=0.8,
    )

    ax.set_title("Desired Signal vs Wiener Estimate")
    ax.set_xlabel("Sample n")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_error(
    error: np.ndarray,
    output_path: str | Path,
    max_samples: int = 1000,
) -> None:
    """
    Plot Wiener estimation error.
    """

    error = np.asarray(error)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(
        error.size,
        max_samples,
    )

    indices = np.arange(n)

    fig, ax = plt.subplots(
        figsize=(10, 5),
    )

    ax.plot(
        indices,
        error[:n],
    )

    ax.set_title("Wiener Estimation Error")
    ax.set_xlabel("Sample n")
    ax.set_ylabel("e[n]")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)