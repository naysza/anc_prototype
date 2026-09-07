import numpy as np

from experiments.module2 import Module2Experiment


def test_module2_wiener_acceptance(tmp_path):
    config = {
        "experiment": {
            "name": "module2_acceptance",
        },
        "signals": {
            "x": {
                "type": "white_noise",
                "fs": 8000,
                "duration": 2.0,
                "amplitude": 1.0,
                "seed": 1234,
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

    result = Module2Experiment().run(config)

    # The known FIR should be recovered closely by
    # the Wiener estimator under the controlled stationary
    # white-noise experiment.
    assert result["coefficient_l2_error"] < 0.15

    # The estimated signal should have low steady-state MSE.
    assert result["steady_state_mse"] < 0.01