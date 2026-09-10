import numpy as np
import pytest

from anc.metrics.mse import (
    calculate_error,
    calculate_mse,
)


def test_error():
    desired = np.array(
        [1.0, 2.0, 3.0]
    )

    estimate = np.array(
        [0.5, 2.5, 2.0]
    )

    error = calculate_error(
        desired,
        estimate,
    )

    expected = np.array(
        [0.5, -0.5, 1.0]
    )

    np.testing.assert_allclose(
        error,
        expected,
    )


def test_mse():
    desired = np.array(
        [1.0, 2.0, 3.0]
    )

    estimate = np.array(
        [0.0, 2.0, 2.0]
    )

    mse = calculate_mse(
        desired,
        estimate,
    )

    expected = (1.0 + 0.0 + 1.0) / 3.0

    assert mse == pytest.approx(expected)


def test_mse_length_mismatch():
    with pytest.raises(ValueError):
        calculate_mse(
            np.ones(5),
            np.ones(4),
        )