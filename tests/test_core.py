import numpy as np
import pytest

from src.anc.core.interfaces import Signal, Experiment


def test_signal_creation():
    data = np.array([1.0, 2.0, 3.0])

    signal = Signal(
        data=data,
        fs=1000,
        name="test_signal",
    )

    assert isinstance(signal.data, np.ndarray)
    np.testing.assert_array_equal(
        signal.data,
        data,
    )

    assert signal.fs == 1000
    assert signal.name == "test_signal"


def test_signal_n_samples():
    data = np.zeros(500)

    signal = Signal(
        data=data,
        fs=1000,
        name="test_signal",
    )

    assert signal.n_samples == 500


def test_signal_duration():
    data = np.zeros(1000)

    signal = Signal(
        data=data,
        fs=1000,
        name="test_signal",
    )

    assert signal.duration == 1.0


def test_experiment_base_class():
    experiment = Experiment()

    with pytest.raises(NotImplementedError):
        experiment.run({})