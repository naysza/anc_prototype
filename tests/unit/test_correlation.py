import numpy as np
import pytest

from anc.core.interfaces import Signal
from anc.stats.correlation import (
    calculate_autocorrelation,
    calculate_crosscorrelation,
    construct_cross_vector,
    construct_wiener_matrix,
)


def test_autocorrelation_known_signal():
    signal = Signal(
        data=np.array([1.0, 2.0, 3.0]),
        fs=1.0,
    )

    rxx = calculate_autocorrelation(
        signal,
        max_lag=2,
    )

    expected = np.array(
        [
            14.0 / 3.0,
            8.0 / 3.0,
        ]
    )

    np.testing.assert_allclose(
        rxx,
        expected,
    )


def test_crosscorrelation_known_signal():
    x = Signal(
        data=np.array([1.0, 2.0, 3.0]),
        fs=1.0,
    )

    d = Signal(
        data=np.array([0.0, 1.0, 1.0]),
        fs=1.0,
    )

    rxd = calculate_crosscorrelation(
        x,
        d,
        max_lag=2,
    )

    # Wiener convention:
    #
    # Rxd[0] = E{x[n] d[n]}
    #        = (0*1 + 1*2 + 1*3) / 3
    #        = 5/3
    #
    # Rxd[1] = E{x[n-1] d[n]}
    #        = (0*0 + 1*1 + 2*1) / 3
    #        = 1
    expected = np.array(
        [
            5.0 / 3.0,
            1.0,
        ]
    )

    np.testing.assert_allclose(
        rxd,
        expected,
    )


def test_wiener_matrix():
    rxx = np.array(
        [
            1.0,
            0.5,
            0.25,
        ]
    )

    R = construct_wiener_matrix(rxx)

    expected = np.array(
        [
            [1.0, 0.5, 0.25],
            [0.5, 1.0, 0.5],
            [0.25, 0.5, 1.0],
        ]
    )

    np.testing.assert_allclose(
        R,
        expected,
    )


def test_cross_vector():
    rxd = np.array(
        [
            1.0,
            0.5,
            0.25,
        ]
    )

    p = construct_cross_vector(rxd)

    expected = np.array(
        [
            [1.0],
            [0.5],
            [0.25],
        ]
    )

    np.testing.assert_allclose(
        p,
        expected,
    )


def test_invalid_lag():
    signal = Signal(
        data=np.ones(10),
        fs=1.0,
    )

    with pytest.raises(ValueError):
        calculate_autocorrelation(
            signal,
            max_lag=0,
        )

    with pytest.raises(ValueError):
        calculate_autocorrelation(
            signal,
            max_lag=11,
        )


def test_crosscorrelation_requires_equal_lengths():
    x = Signal(
        data=np.ones(10),
        fs=1.0,
    )

    d = Signal(
        data=np.ones(9),
        fs=1.0,
    )

    with pytest.raises(ValueError):
        calculate_crosscorrelation(
            x,
            d,
            max_lag=3,
        )


def test_crosscorrelation_requires_same_sampling_rate():
    x = Signal(
        data=np.ones(10),
        fs=1000.0,
    )

    d = Signal(
        data=np.ones(10),
        fs=2000.0,
    )

    with pytest.raises(ValueError):
        calculate_crosscorrelation(
            x,
            d,
            max_lag=3,
        )