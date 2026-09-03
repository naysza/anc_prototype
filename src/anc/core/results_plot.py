import matplotlib.pyplot as plt
import os
from typing import Any
import numpy as np

class BasicPlotter:
    """A basic plotter implementation for saving signal visualizations."""
    
    def plot(self, data: Any, title: str, save_path: str = None) -> None:
        plt.figure(figsize=(10, 4))
        plt.plot(data)
        plt.title(title)
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        plt.close()

def plot_correlation(rxx: np.ndarray, rxd: np.ndarray, save_path: str = None) -> None:
    """Plot autocorrelation and cross-correlation."""
    lags = np.arange(len(rxx))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(lags, rxx, marker='o', markersize=4, linestyle='-', alpha=0.7)
    plt.title('Autocorrelation Rxx[k]')
    plt.xlabel('Lag k')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(lags, rxd, marker='o', markersize=4, linestyle='-', alpha=0.7, color='orange')
    plt.title('Cross-correlation Rxd[k]')
    plt.xlabel('Lag k')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.close()

