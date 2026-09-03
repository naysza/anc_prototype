import numpy as np
import pytest

from src.anc.core.interfaces import Signal
from src.anc.signals.generators import generate_signal


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

    n = np.arange(1000)

    expected = np.sin(
        2 * np.pi * 100 * n / 1000
    )

    np.testing.assert_allclose(
        signal.data,
        expected,
    )


def test_generate_multi_tone():
    config = {
        "type": "multi_tone",
        "fs": 2000,
        "duration": 1.0,
        "frequencies": [100, 250],
        "amplitudes": [1.0, 0.5],
    }

    signal = generate_signal(config)

    assert isinstance(signal, Signal)
    assert signal.n_samples == 2000

    n = np.arange(2000)

    expected = (
        np.sin(2 * np.pi * 100 * n / 2000)
        + 0.5 * np.sin(2 * np.pi * 250 * n / 2000)
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


def test_colored_noise():
    config = {
        "type": "colored_noise",
        "fs": 1000,
        "duration": 1.0,
        "amplitude": 1.0,
        "seed": 42,
        "color": "pink",
    }

    signal = generate_signal(config)

    assert isinstance(signal, Signal)
    assert signal.n_samples == 1000
    assert np.all(np.isfinite(signal.data))


def test_impulse():
    config = {
        "type": "impulse",
        "fs": 1000,
        "duration": 1.0,
        "amplitude": 2.0,
    }

    signal = generate_signal(config)

    assert isinstance(signal, Signal)
    assert signal.data[0] == 2.0
    assert np.all(signal.data[1:] == 0.0)


def test_burst():
    config = {
        "type": "burst",
        "fs": 1000,
        "duration": 1.0,
        "frequency": 100,
        "amplitude": 1.0,
        "start_time": 0.2,
        "end_time": 0.5,
    }

    signal = generate_signal(config)

    assert isinstance(signal, Signal)
    assert signal.n_samples == 1000

    # Before burst
    assert np.allclose(
        signal.data[:200],
        0.0,
    )

    # After burst
    assert np.allclose(
        signal.data[500:],
        0.0,
    )

    # There should be non-zero samples during the burst.
    assert np.any(
        np.abs(signal.data[200:500]) > 0
    )


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


def test_invalid_duration():
    config = {
        "type": "sinusoid",
        "fs": 1000,
        "duration": 0,
        "frequency": 100,
    }

    with pytest.raises(ValueError):
        generate_signal(config)


def test_multitone_length_mismatch():
    config = {
        "type": "multi_tone",
        "fs": 1000,
        "duration": 1.0,
        "frequencies": [100, 200],
        "amplitudes": [1.0],
    }

    with pytest.raises(ValueError):
        generate_signal(config)