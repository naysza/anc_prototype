import argparse
import sys
import os

from src.anc.core.interfaces import Experiment, Signal
from src.anc.signals.generators import generate_signal
from src.anc.stats.correlation import (
    calculate_autocorrelation,
    calculate_crosscorrelation,
    construct_wiener_matrix,
    construct_cross_vector
)
from src.anc.core.results_plot import plot_correlation
from src.anc.core.results_io import save_result_json

class CorrelationExperiment(Experiment):
    """Experiment to test Task 2A: Correlation & Statistical Signal Analysis."""
    
    def run(self, config: dict):
        print("Running Correlation Analysis Experiment (Task 2A)...")
        
        # 1. Setup Signal Generators
        # Reference signal x[n]
        x_config = config.get('x_signal', {
            "type": "white_noise",
            "fs": 8000,
            "duration": 0.5,
            "amplitude": 1.0,
            "seed": 42
        })
        x_signal = generate_signal(x_config)
        
        # Desired signal d[n]
        d_config = config.get('d_signal', {
            "type": "white_noise",
            "fs": 8000,
            "duration": 0.5,
            "amplitude": 0.5,
            "seed": 1337
        })
        d_signal = generate_signal(d_config)
        
        max_lag = config.get('max_lag', 64)
        print(f"Signals generated (N={x_signal.n_samples}). Computing correlations for max_lag={max_lag}...")
        
        # 2. Compute Correlations
        rxx = calculate_autocorrelation(x_signal, max_lag)
        rxd = calculate_crosscorrelation(x_signal, d_signal, max_lag)
        
        # 3. Construct Matrix R and vector p
        R_matrix = construct_wiener_matrix(rxx)
        p_vector = construct_cross_vector(rxd)
        
        print("Constructed Toeplitz matrix R and cross-correlation vector p.")
        
        # 4. Save and Visualize
        output_params = config.get('output', {})
        out_dir = output_params.get('output_dir', 'outputs/2a_correlation')
        
        if output_params.get('save_results', True):
            results = {
                "rxx_first_10": rxx[:10].tolist(),
                "rxd_first_10": rxd[:10].tolist(),
                "R_shape": R_matrix.shape,
                "p_shape": p_vector.shape
            }
            save_result_json(results, os.path.join(out_dir, 'correlation_results.json'))
            print(f"Results saved to {out_dir}/correlation_results.json")
            
        if output_params.get('plot_signals', True):
            plot_correlation(rxx, rxd, os.path.join(out_dir, 'correlation_plot.png'))
            print(f"Correlation plot saved to {out_dir}/correlation_plot.png")
            
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 2A Experiment")
    # Provide a simple manual override for running directly without CLI yaml for now.
    experiment = CorrelationExperiment()
    
    experiment.run({})

