import numpy as np
from typing import Any, Dict
from anc.core.interfaces import Configuration, Experiment, Result
from anc.core.results_plot import BasicPlotter
from anc.core.results_io import save_result_json
import os

class DictConfiguration(Configuration):
    """Simple configuration implementation using a dictionary."""
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        
    def load(self, filepath: str) -> None:
        pass # Not used here, main handles it

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

class DummyResult(Result):
    def __init__(self, output_signal: np.ndarray, metrics: Dict[str, float]):
        self.output_signal = output_signal
        self.metrics = metrics
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics,
            "output_length": len(self.output_signal)
        }

class DummyANCExperiment(Experiment):
    """A dummy experiment implementation to prove the architecture works."""
    
    def run(self, config: Configuration) -> Result:
        print("Running dummy experiment driven by config...")
        
        # 1. Read config params (no hardcoded parameters)
        signal_params = config.get('signal_params', {})
        processing_params = config.get('processing_params', {})
        output_params = config.get('output', {})
        
        # 2. Mock generation
        duration = signal_params.get('duration', 1.0)
        sampling_rate = signal_params.get('sampling_rate', 8000)
        t = np.linspace(0, duration, int(sampling_rate * duration))
        
        # 3. Mock processing
        print(f"Applying {processing_params.get('filter_type', 'unknown')} filter...")
        output_signal = np.sin(2 * np.pi * signal_params.get('frequency', 1000) * t)
        
        result = DummyResult(output_signal, {"mse": 0.05, "snr": 15.2})
        
        # 4. Result saving & visualization mechanism
        if output_params.get('save_results', False):
            out_dir = output_params.get('output_dir', 'outputs')
            save_result_json(result.to_dict(), os.path.join(out_dir, 'result.json'))
            print("Results saved.")
            
        if output_params.get('plot_signals', False):
            out_dir = output_params.get('output_dir', 'outputs')
            plotter = BasicPlotter()
            plotter.plot(output_signal, "Filtered Output Signal", os.path.join(out_dir, 'output_plot.png'))
            print("Visualization saved.")
            
        return result
