from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Signal:
    """
    Common representation for all discrete-time signals
    used throughout the ANC project.
    """

    data: np.ndarray
    fs: float
    name: str = ""

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float)

        if self.data.ndim != 1:
            raise ValueError("Signal data must be one-dimensional.")

        if self.data.size == 0:
            raise ValueError("Signal data must contain at least one sample.")

        if not np.all(np.isfinite(self.data)):
            raise ValueError("Signal data must contain only finite values.")

        if not np.isfinite(self.fs) or self.fs <= 0:
            raise ValueError("Sampling frequency fs must be positive and finite.")

    @property
    def n_samples(self) -> int:
        """Return the number of samples."""
        return self.data.size

    @property
    def duration(self) -> float:
        """Return duration in seconds."""
        return self.n_samples / self.fs


class Experiment:
    """
    Base interface for ANC experiments.
    """

    def run(self, config: dict):
        raise NotImplementedError(
            "Experiment subclasses must implement run()."
        )