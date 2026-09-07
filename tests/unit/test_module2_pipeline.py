import numpy as np

from experiments.module2 import Module2Experiment


def test_module2_end_to_end(tmp_path):
    config = {
        "experiment": {
            "name": "test_wiener",
        },
        "signals": {
            "x": {
                "type": "white_noise",
                "fs": 8000,
                "duration": 1.0,
                "amplitude": 1.0,
                "seed": 42,
                "name": "reference",
            }
        },
        "correlation": {
            "filter_length": 8,
        },
        "output": {
            "directory": str(tmp_path),
        },
    }

    experiment = Module2Experiment()

    summary = experiment.run(config)

    assert summary["filter_length"] == 8

    assert summary["R_shape"] == [8, 8]
    assert summary["p_shape"] == [8, 1]

    assert summary["mse"] >= 0.0

    output_files = [
        "x.npy",
        "d.npy",
        "rxx.npy",
        "rxd.npy",
        "R.npy",
        "p.npy",
        "w_opt.npy",
        "true_coefficients.npy",
        "d_hat.npy",
        "error.npy",
        "summary.json",
        "correlations.png",
        "correlation_matrix.png",
        "wiener_coefficients.png",
        "desired_vs_estimated.png",
        "error.png",
    ]

    for filename in output_files:
        assert (tmp_path / filename).exists()

    w_opt = np.load(
        tmp_path / "w_opt.npy"
    )

    true_coefficients = np.load(
        tmp_path / "true_coefficients.npy"
    )

    assert w_opt.shape == true_coefficients.shape

    assert np.all(
        np.isfinite(w_opt)
    )
