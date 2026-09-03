import numpy as np

from src.anc.signals.generators import generate_signal


configs = [
    {
        "type": "sinusoid",
        "fs": 1000,
        "duration": 1.0,
        "frequency": 100,
        "amplitude": 1.0,
        "name": "sinusoid",
    },
    {
        "type": "multi_tone",
        "fs": 2000,
        "duration": 1.0,
        "frequencies": [100, 250, 400],
        "amplitudes": [1.0, 0.5, 0.25],
        "name": "multi_tone",
    },
    {
        "type": "white_noise",
        "fs": 1000,
        "duration": 1.0,
        "amplitude": 1.0,
        "seed": 42,
        "name": "white_noise",
    },
    {
        "type": "colored_noise",
        "fs": 1000,
        "duration": 1.0,
        "amplitude": 1.0,
        "seed": 42,
        "color": "pink",
        "name": "pink_noise",
    },
    {
        "type": "impulse",
        "fs": 1000,
        "duration": 1.0,
        "amplitude": 1.0,
        "name": "impulse",
    },
    {
        "type": "burst",
        "fs": 1000,
        "duration": 1.0,
        "frequency": 100,
        "amplitude": 1.0,
        "start_time": 0.2,
        "end_time": 0.5,
        "name": "burst",
    },
]


for config in configs:
    signal = generate_signal(config)

    print("=" * 50)
    print(f"Signal:    {signal.name}")
    print(f"Type:      {config['type']}")
    print(f"fs:        {signal.fs} Hz")
    print(f"Samples:   {signal.n_samples}")
    print(f"Duration:  {signal.duration:.3f} s")
    print(f"Min:       {np.min(signal.data):.4f}")
    print(f"Max:       {np.max(signal.data):.4f}")
    print(f"Mean:      {np.mean(signal.data):.4f}")
    print(f"Std:       {np.std(signal.data):.4f}")