import numpy as np

from anc.core.interfaces import Signal
from anc.filters import apply_fir


x = Signal(
    data=np.array([1.0, 4.0, 2.0, 7.0]),
    fs=1000,
    name="identity_test",
)

y = apply_fir(x, [1.0])

print("Input:")
print(x.data)

print("\nOutput:")
print(y.data)

print("\nPASS:", np.allclose(x.data, y.data))