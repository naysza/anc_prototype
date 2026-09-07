from __future__ import annotations

from pathlib import Path

import numpy as np

from anc.core.interfaces import Experiment, Signal
from anc.core.results import (
    prepare_output_dir,
    save_array,
    save_json,
)
from anc.core.config import load_config
from anc.filters.fir import apply_fir
from anc.metrics.mse import calculate_error, calculate_mse
from anc.signals.generators import generate_signal
from anc.stats.correlation import (
    calculate_autocorrelation,
    calculate_crosscorrelation,
    construct_cross_vector,
    construct_wiener_matrix,
)
from anc.estimation.wiener import solve_wiener, estimate_signal
from anc.visualization.correlation import (
    plot_correlation_matrix,
    plot_correlations,
)
from anc.visualization.wiener import (
    plot_error,
    plot_estimation,
    plot_wiener_coefficients,
)


class Module2Experiment(Experiment):
    """
    Complete Module 2 Wiener/MMSE experiment.

    Pipeline:

        x[n]
          ↓
        known FIR
          ↓
        d[n]

        x,d
          ↓
        Rxx/Rxd
          ↓
        R,p
          ↓
        Rw=p
          ↓
        w_opt
          ↓
        d_hat
          ↓
        e
          ↓
        MSE
    """

    def run(self, config: dict) -> dict:
        experiment_config = config["experiment"]
        signal_config = config["signals"]["x"]

        filter_length = int(
            config["correlation"]["filter_length"]
        )

        output_dir = prepare_output_dir(
            config["output"]["directory"]
        )

        # ---------------------------------------------------------
        # 1. Generate reference signal x[n]
        # ---------------------------------------------------------

        x = generate_signal(signal_config)

        # ---------------------------------------------------------
        # 2. Create controlled desired signal d[n]
        # ---------------------------------------------------------

        true_coefficients = np.array(
            [
                0.8,
                -0.4,
                0.25,
                0.1,
                -0.05,
                0.03,
                0.02,
                -0.01,
            ],
            dtype=float,
        )

        if true_coefficients.size != filter_length:
            raise ValueError(
                "Configured filter_length must equal the number "
                "of controlled FIR coefficients."
            )

        desired_full = apply_fir(
            x,
            true_coefficients,
        )

        # Keep d[n] the same length as x[n].
        d = Signal(
            data=desired_full.data[: x.n_samples],
            fs=x.fs,
            name="desired",
        )

        # ---------------------------------------------------------
        # 3. Correlation analysis
        # ---------------------------------------------------------

        rxx = calculate_autocorrelation(
            x,
            max_lag=filter_length,
        )

        rxd = calculate_crosscorrelation(
            x,
            d,
            max_lag=filter_length,
        )

        R = construct_wiener_matrix(rxx)
        p = construct_cross_vector(rxd)

        # ---------------------------------------------------------
        # 4. Wiener solution
        # ---------------------------------------------------------

        w_opt = solve_wiener(
            R,
            p,
        )

        # ---------------------------------------------------------
        # 5. Estimated desired signal
        # ---------------------------------------------------------

        d_hat = estimate_signal(
            x.data,
            w_opt,
        )

        # ---------------------------------------------------------
        # 6. Error and MSE
        # ---------------------------------------------------------

        error = calculate_error(
            d.data,
            d_hat,
        )

        mse = calculate_mse(
            d.data,
            d_hat,
        )

        coefficient_error = float(
            np.linalg.norm(
                w_opt - true_coefficients
            )
        )

        # Ignore initial FIR transient for a more meaningful
        # steady-state coefficient/performance diagnostic.
        warmup = filter_length - 1

        if warmup < d.n_samples:
            steady_state_mse = calculate_mse(
                d.data[warmup:],
                d_hat[warmup:],
            )
        else:
            steady_state_mse = mse

        # ---------------------------------------------------------
        # 7. Save numerical artifacts
        # ---------------------------------------------------------

        save_array(
            output_dir,
            "x.npy",
            x.data,
        )

        save_array(
            output_dir,
            "d.npy",
            d.data,
        )

        save_array(
            output_dir,
            "rxx.npy",
            rxx,
        )

        save_array(
            output_dir,
            "rxd.npy",
            rxd,
        )

        save_array(
            output_dir,
            "R.npy",
            R,
        )

        save_array(
            output_dir,
            "p.npy",
            p,
        )

        save_array(
            output_dir,
            "w_opt.npy",
            w_opt,
        )

        save_array(
            output_dir,
            "true_coefficients.npy",
            true_coefficients,
        )

        save_array(
            output_dir,
            "d_hat.npy",
            d_hat,
        )

        save_array(
            output_dir,
            "error.npy",
            error,
        )

        # ---------------------------------------------------------
        # 8. Save summary
        # ---------------------------------------------------------

        summary = {
            "module": "2",
            "experiment": experiment_config["name"],
            "n_samples": x.n_samples,
            "sampling_frequency": x.fs,
            "filter_length": filter_length,
            "mse": mse,
            "steady_state_mse": steady_state_mse,
            "coefficient_l2_error": coefficient_error,
            "true_coefficients": true_coefficients.tolist(),
            "w_opt": w_opt.tolist(),
            "R_shape": list(R.shape),
            "p_shape": list(p.shape),
        }

        save_json(
            output_dir,
            "summary.json",
            summary,
        )

        # ---------------------------------------------------------
        # 9. Visualizations
        # ---------------------------------------------------------

        plot_correlations(
            rxx,
            rxd,
            output_dir / "correlations.png",
        )

        plot_correlation_matrix(
            R,
            output_dir / "correlation_matrix.png",
        )

        plot_wiener_coefficients(
            w_opt,
            output_dir / "wiener_coefficients.png",
        )

        plot_estimation(
            d.data,
            d_hat,
            output_dir / "desired_vs_estimated.png",
        )

        plot_error(
            error,
            output_dir / "error.png",
        )

        print()
        print("=" * 60)
        print("MODULE 2 — WIENER/MMSE")
        print("=" * 60)
        print(f"Samples              : {x.n_samples}")
        print(f"Filter length        : {filter_length}")
        print(f"MSE                  : {mse:.8e}")
        print(f"Steady-state MSE     : {steady_state_mse:.8e}")
        print(f"Coefficient L2 error : {coefficient_error:.8e}")
        print()
        print("True coefficients:")
        print(true_coefficients)
        print()
        print("Wiener coefficients:")
        print(w_opt)
        print()
        print(f"Artifacts saved to: {output_dir}")
        print("=" * 60)

        return summary


def run_module2(config_path: str) -> dict:
    """
    Load YAML configuration and execute Module 2.
    """

    config = load_config(config_path)

    experiment = Module2Experiment()

    return experiment.run(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Module 2 Wiener/MMSE experiment."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to Module 2 YAML configuration.",
    )

    args = parser.parse_args()

    run_module2(args.config)