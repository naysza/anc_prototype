from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_correlations(
    rxx: np.ndarray,
    rxd: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Plot Rxx and Rxd versus lag.
    """

    rxx = np.asarray(rxx)
    rxd = np.asarray(rxd)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lags = np.arange(rxx.size)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
    )

    axes[0].stem(
        lags,
        rxx,
        basefmt=" ",
    )
    axes[0].set_title("Autocorrelation Rxx[k]")
    axes[0].set_ylabel("Rxx[k]")
    axes[0].grid(True)

    axes[1].stem(
        lags,
        rxd,
        basefmt=" ",
    )
    axes[1].set_title("Cross-correlation Rxd[k]")
    axes[1].set_xlabel("Lag k")
    axes[1].set_ylabel("Rxd[k]")
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_correlation_matrix(
    R: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Visualize the Wiener correlation matrix.
    """

    R = np.asarray(R)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(7, 6),
    )

    image = ax.imshow(
        R,
        aspect="auto",
    )

    ax.set_title("Wiener Correlation Matrix R")
    ax.set_xlabel("Coefficient index")
    ax.set_ylabel("Coefficient index")

    fig.colorbar(
        image,
        ax=ax,
        label="Correlation",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)