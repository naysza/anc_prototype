import matplotlib.pyplot as plt
import os
from typing import Any
from anc.core.interfaces import Visualization

class BasicPlotter(Visualization):
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
