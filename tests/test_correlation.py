import numpy as np
import pytest

from src.anc.core.interfaces import Signal
from src.anc.stats.correlation import (
    calculate_autocorrelation,
    calculate_crosscorrelation,
    construct_wiener_matrix,
    construct_cross_vector
)

def test_autocorrelation():
    # Generate simple test signal [1, 2, 3]
    data = np.array([1.0, 2.0, 3.0])
    N = len(data)
    sig = Signal(data=data, fs=1.0)
    
    # max_lag = 2: calculates E{x[n]x[n-0]} and E{x[n]x[n-1]}
    rxx = calculate_autocorrelation(sig, max_lag=2)
    
    # Expected unnormalized: [1*1+2*2+3*3, 1*2+2*3] = [14, 8]
    # Normalized by N=3: [14/3, 8/3]
    expected = np.array([14.0/3.0, 8.0/3.0])
    
    np.testing.assert_almost_equal(rxx, expected)

def test_crosscorrelation():
    data_x = np.array([1.0, 2.0, 3.0])
    data_d = np.array([0.0, 1.0, 1.0])
    
    sig_x = Signal(data=data_x, fs=1.0)
    sig_d = Signal(data=data_d, fs=1.0)
    
    # max_lag = 2: calculates E{x[n]d[n-0]} and E{x[n]d[n-1]}
    rxd = calculate_crosscorrelation(sig_x, sig_d, max_lag=2)
    
    # Expected unnormalized: sum(x[n]d[n-k])
    # k=0: 1*0 + 2*1 + 3*1 = 5
    # k=1: x shifted right by 1 relative to d (or d shifted left)
    # wait, x[n]d[n-k]: 
    # n=1: x[1]d[0] = 2*0 = 0
    # n=2: x[2]d[1] = 3*1 = 3
    # sum = 3
    expected = np.array([5.0/3.0, 3.0/3.0])
    
    np.testing.assert_almost_equal(rxd, expected)

def test_wiener_matrix():
    rxx = np.array([1.0, 0.5, 0.25])
    R = construct_wiener_matrix(rxx)
    
    # Should be a symmetric Toeplitz matrix
    expected = np.array([
        [1.0, 0.5, 0.25],
        [0.5, 1.0, 0.5],
        [0.25, 0.5, 1.0]
    ])
    np.testing.assert_almost_equal(R, expected)

def test_cross_vector():
    rxd = np.array([1.0, 0.5, 0.25])
    p = construct_cross_vector(rxd)
    
    expected = np.array([[1.0], [0.5], [0.25]])
    np.testing.assert_almost_equal(p, expected)
