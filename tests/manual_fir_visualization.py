import matplotlib.pyplot as plt

from anc.core.config import load_config
from anc.signals.generators import generate_signal
from anc.filters import apply_fir


# Load configuration
config = load_config("configs/example.yaml")


# Generate reference signal x[n]
x = generate_signal(config["signal"])


# Read FIR coefficients h[k]
h = config["filter"]["coefficients"]


# Apply FIR filter
y = apply_fir(x, h)


# Create sample indices
x_n = range(x.n_samples)
h_k = range(len(h))
y_n = range(y.n_samples)


# Plot input signal x[n]
plt.figure()
plt.stem(x_n, x.data)
plt.title("Input Signal x[n]")
plt.xlabel("Sample index n")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()


# Plot FIR impulse response h[k]
plt.figure()
plt.stem(h_k, h)
plt.title("FIR Impulse Response h[k]")
plt.xlabel("Filter index k")
plt.ylabel("Coefficient")
plt.grid(True)
plt.show()


# Plot filtered output y[n]
plt.figure()
plt.stem(y_n, y.data)
plt.title("FIR Output y[n]")
plt.xlabel("Sample index n")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()