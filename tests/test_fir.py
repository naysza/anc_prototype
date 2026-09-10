import numpy as np
import pytest

from anc.core.interfaces import Signal
from anc.filters import apply_fir


def test_known_convolution():
    """Check FIR against a manually known convolution."""
    x = Signal(
        data=np.array([1.0, 2.0, 3.0]),
        fs=1000,
        name="test",
    )

    h = [1.0, 1.0]

    y = apply_fir(x, h)

    expected = np.array([1.0, 3.0, 5.0, 3.0])

    assert np.allclose(y.data, expected)


def test_impulse_response():
    """An impulse input should reproduce the FIR coefficients."""
    x = Signal(
        data=np.array([1.0, 0.0, 0.0, 0.0]),
        fs=1000,
        name="impulse",
    )

    h = [0.2, 0.5, 0.2]

    y = apply_fir(x, h)

    expected = np.array(
        [0.2, 0.5, 0.2, 0.0, 0.0, 0.0]
    )

    assert np.allclose(y.data, expected)


def test_identity_filter():
    """A one-tap [1] FIR should leave the signal unchanged."""
    x = Signal(
        data=np.array([1.0, 4.0, 2.0, 7.0]),
        fs=1000,
        name="identity",
    )

    y = apply_fir(x, [1.0])

    assert np.allclose(y.data, x.data)


def test_output_length():
    """Full convolution length must be N + M - 1."""
    x = Signal(
        data=np.array([1.0, 2.0, 3.0, 4.0]),
        fs=1000,
        name="length_test",
    )

    h = [0.5, 0.25, 0.1]

    y = apply_fir(x, h)

    assert y.n_samples == 4 + 3 - 1


def test_sampling_frequency_preserved():
    """FIR filtering must preserve the input sampling frequency."""
    x = Signal(
        data=np.array([1.0, 2.0, 3.0]),
        fs=8000,
        name="fs_test",
    )

    y = apply_fir(x, [1.0, 0.5])

    assert y.fs == 8000


def test_output_name():
    """Output should use the expected naming convention."""
    x = Signal(
        data=np.array([1.0, 2.0]),
        fs=1000,
        name="reference",
    )

    y = apply_fir(x, [1.0])

    assert y.name == "reference_fir"


def test_empty_coefficients():
    """Empty FIR coefficient list should be rejected."""
    x = Signal(
        data=np.array([1.0, 2.0, 3.0]),
        fs=1000,
        name="test",
    )

    with pytest.raises(ValueError):
        apply_fir(x, [])


def test_non_numeric_coefficients():
    """Non-numeric coefficients should be rejected."""
    x = Signal(
        data=np.array([1.0, 2.0, 3.0]),
        fs=1000,
        name="test",
    )

    with pytest.raises(TypeError):
        apply_fir(x, ["a", "b"])


def test_nan_coefficients():
    """NaN coefficients should be rejected."""
    x = Signal(
        data=np.array([1.0, 2.0, 3.0]),
        fs=1000,
        name="test",
    )

    with pytest.raises(ValueError):
        apply_fir(x, [1.0, np.nan])


def test_infinite_coefficients():
    """Infinite coefficients should be rejected."""
    x = Signal(
        data=np.array([1.0, 2.0, 3.0]),
        fs=1000,
        name="test",
    )

    with pytest.raises(ValueError):
        apply_fir(x, [1.0, np.inf])


def test_invalid_signal():
    """Input must be a Signal object."""
    with pytest.raises(TypeError):
        apply_fir([1.0, 2.0, 3.0], [1.0])