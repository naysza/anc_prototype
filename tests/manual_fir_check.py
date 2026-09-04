import numpy as np

from anc.core.interfaces import Signal
from anc.filters import apply_fir


x = Signal(
    data=np.array([1.0, 2.0, 3.0]),
    fs=1000,
    name="test",
)

h = [0.2, 0.5, 0.2]

y = apply_fir(x, h)

print("Input:")
print(x.data)

print("\nFIR coefficients:")
print(h)

print("\nOutput:")
print(y.data)

print("\nOutput length:")
print(y.n_samples)

print("\nSampling frequency:")
print(y.fs)

print("\nName:")
print(y.name)