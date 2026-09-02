import numpy as np
import pytest

from src.anc.signals.generators import generate_signal
from src.anc.core.interfaces import Signal


def test_generate_sinusoid():
    config = {
        "type": "sinusoid",
        "fs": 1000,
        "duration": 1.0,
        "frequency": 100,
        "amplitude": 1.0,
        "name": "test_sinusoid",
    }

    signal = generate_signal(config)

    assert isinstance(signal, Signal)
    assert signal.fs == 1000
    assert signal.n_samples == 1000
    assert signal.name == "test_sinusoid"

    expected = np.sin(
        2 * np.pi * 100 * np.arange(1000) / 1000
    )

    np.testing.assert_allclose(
        signal.data,
        expected,
    )


def test_white_noise_is_reproducible():
    config = {
        "type": "white_noise",
        "fs": 1000,
        "duration": 1.0,
        "amplitude": 1.0,
        "seed": 42,
    }

    signal_1 = generate_signal(config)
    signal_2 = generate_signal(config)

    np.testing.assert_array_equal(
        signal_1.data,
        signal_2.data,
    )


def test_impulse():
    config = {
        "type": "impulse",
        "fs": 1000,
        "duration": 1.0,
        "amplitude": 2.0,
    }

    signal = generate_signal(config)

    assert signal.data[0] == 2.0
    assert np.all(signal.data[1:] == 0.0)


def test_invalid_signal_type():
    config = {
        "type": "unknown",
        "fs": 1000,
        "duration": 1.0,
    }

    with pytest.raises(ValueError):
        generate_signal(config)


def test_invalid_sampling_frequency():
    config = {
        "type": "sinusoid",
        "fs": 0,
        "duration": 1.0,
        "frequency": 100,
    }

    with pytest.raises(ValueError):
        generate_signal(config)