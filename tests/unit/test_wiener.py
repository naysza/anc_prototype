import numpy as np
import pytest

from anc.estimation.wiener import (
    estimate_signal,
    solve_wiener,
)


def test_solve_wiener():
    R = np.array(
        [
            [2.0, 0.5],
            [0.5, 1.0],
        ]
    )

    p = np.array(
        [
            [1.0],
            [0.5],
        ]
    )

    w = solve_wiener(
        R,
        p,
    )

    expected = np.linalg.solve(
        R,
        p[:, 0],
    )

    np.testing.assert_allclose(
        w,
        expected,
    )


def test_estimate_signal():
    x = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ]
    )

    w = np.array(
        [
            2.0,
            0.5,
        ]
    )

    estimate = estimate_signal(
        x,
        w,
    )

    expected = np.array(
        [
            2.0,
            4.5,
            7.0,
            9.5,
        ]
    )

    np.testing.assert_allclose(
        estimate,
        expected,
    )


def test_wiener_dimension_mismatch():
    R = np.eye(3)
    p = np.ones(2)

    with pytest.raises(ValueError):
        solve_wiener(
            R,
            p,
        )


def test_singular_matrix_rejected():
    R = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )

    p = np.ones(2)

    with pytest.raises(np.linalg.LinAlgError):
        solve_wiener(
            R,
            p,
        )