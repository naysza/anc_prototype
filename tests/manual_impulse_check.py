import numpy as np

from anc.core.interfaces import Signal
from anc.filters import apply_fir


x = Signal(
    data=np.array([1.0, 0.0, 0.0, 0.0]),
    fs=1000,
    name="impulse",
)

h = [0.2, 0.5, 0.2]

y = apply_fir(x, h)

print("Output:")
print(y.data)

expected = np.array([0.2, 0.5, 0.2, 0.0, 0.0, 0.0])

print("\nExpected:")
print(expected)

print("\nPASS:", np.allclose(y.data, expected))