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

    @property
    def n_samples(self) -> int:
        """Return the number of samples in the signal."""
        return len(self.data)

    @property
    def duration(self) -> float:
        """Return the signal duration in seconds."""
        return self.n_samples / self.fs


class Experiment:
    """
    Base interface for all experiments in the ANC project.
    """

    def run(self, config: dict):
        """
        Run the experiment using the supplied configuration.

        Subclasses must implement this method.
        """
        raise NotImplementedError(
            "Experiment subclasses must implement run()."
        )