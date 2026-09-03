import numpy as np
from scipy.linalg import toeplitz
from src.anc.core.interfaces import Signal

def calculate_autocorrelation(x: Signal, max_lag: int) -> np.ndarray:
    """
    Calculate the biased sample autocorrelation estimate Rxx[k] = E{x[n]x[n-k]}.
    
    The biased estimate (dividing by N for all lags) ensures that the
    resulting correlation matrix is positive semi-definite.
    
    Parameters
    ----------
    x : Signal
        The reference signal.
    max_lag : int
        The maximum lag to compute (returns array of length max_lag).
        
    Returns
    -------
    np.ndarray
        Autocorrelation array of length max_lag.
    """
    data = x.data
    N = len(data)
    
    if max_lag > N:
        raise ValueError("max_lag cannot be greater than the signal length.")
        
    # Compute full correlation: sum(x[n] * x[n-k])
    corr = np.correlate(data, data, mode='full')
    center = N - 1
    
    # Extract positive lags 0 to max_lag - 1
    rxx = corr[center:center + max_lag]
    
    # Biased estimate: divide by N
    return rxx / N

def calculate_crosscorrelation(x: Signal, d: Signal, max_lag: int) -> np.ndarray:
    """
    Calculate the biased sample cross-correlation estimate Rxd[k] = E{x[n]d[n-k]}.
    
    Parameters
    ----------
    x : Signal
        The reference signal.
    d : Signal
        The desired signal.
    max_lag : int
        The maximum lag to compute (returns array of length max_lag).
        
    Returns
    -------
    np.ndarray
        Cross-correlation array of length max_lag.
    """
    x_data = x.data
    d_data = d.data
    
    if len(x_data) != len(d_data):
        raise ValueError("Reference and desired signals must have the same length.")
        
    N = len(x_data)
    
    if max_lag > N:
        raise ValueError("max_lag cannot be greater than the signal length.")
        
    # Compute full cross-correlation. 
    # Note: np.correlate(a, b, 'full') computes sum(a[n] * b[n-k]).
    # We want E{x[n] * d[n-k]}.
    corr = np.correlate(x_data, d_data, mode='full')
    center = N - 1
    
    # Extract positive lags 0 to max_lag - 1
    rxd = corr[center:center + max_lag]
    
    # Biased estimate: divide by N
    return rxd / N

def construct_wiener_matrix(rxx: np.ndarray) -> np.ndarray:
    """
    Construct the M x M symmetric Toeplitz correlation matrix R from Rxx.
    
    Parameters
    ----------
    rxx : np.ndarray
        Autocorrelation vector of length M.
        
    Returns
    -------
    np.ndarray
        M x M Toeplitz matrix R.
    """
    return toeplitz(rxx)

def construct_cross_vector(rxd: np.ndarray) -> np.ndarray:
    """
    Construct the M x 1 cross-correlation column vector p from Rxd.
    
    Parameters
    ----------
    rxd : np.ndarray
        Cross-correlation vector of length M.
        
    Returns
    -------
    np.ndarray
        M x 1 column vector p.
    """
    return rxd.reshape(-1, 1)
