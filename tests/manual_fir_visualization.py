import matplotlib.pyplot as plt

from anc.core.config import load_config
from anc.signals.generators import generate_signal
from anc.filters import apply_fir


# Load configuration
config = load_config("configs/example.yaml")


# Generate input signal x[n]
x = generate_signal(config["signal"])


# Read FIR coefficients h[k]
h = config["filter"]["coefficients"]


# Apply FIR
y = apply_fir(x, h)


# --------------------------------------------------
# 1. Full input signal
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(x.data)

plt.title("Input Signal x[n] — Full Signal")
plt.xlabel("Sample index n")
plt.ylabel("Amplitude")
plt.grid(True)

plt.show()


# --------------------------------------------------
# 2. Zoomed discrete-time input signal
# --------------------------------------------------

num_samples = min(100, x.n_samples)

plt.figure(figsize=(12, 5))

plt.stem(
    range(num_samples),
    x.data[:num_samples],
)

plt.title("Input Signal x[n] — First 100 Samples")
plt.xlabel("Sample index n")
plt.ylabel("Amplitude")
plt.grid(True)

plt.show()


# --------------------------------------------------
# 3. FIR coefficients
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.stem(
    range(len(h)),
    h,
)

plt.title("FIR Impulse Response h[k]")
plt.xlabel("Filter index k")
plt.ylabel("Coefficient")
plt.grid(True)

plt.show()


# --------------------------------------------------
# 4. Full FIR output
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(y.data)

plt.title("FIR Output y[n] — Full Signal")
plt.xlabel("Sample index n")
plt.ylabel("Amplitude")
plt.grid(True)

plt.show()


# --------------------------------------------------
# 5. Zoomed FIR output
# --------------------------------------------------

num_output_samples = min(100, y.n_samples)

plt.figure(figsize=(12, 5))

plt.stem(
    range(num_output_samples),
    y.data[:num_output_samples],
)

plt.title("FIR Output y[n] — First 100 Samples")
plt.xlabel("Sample index n")
plt.ylabel("Amplitude")
plt.grid(True)

plt.show()