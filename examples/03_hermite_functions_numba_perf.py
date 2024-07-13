"""
This script compares the performance of the Hermite functions implemented in NumPy and
Numba.

"""

# === Imports ===

import numpy as np
import perfplot

from robust_hermite_ft import hermite_function_basis

# === Setup ===

# the x-values to evaluate the Hermite functions
X_FROM = -5.0
X_TO = 5.0
NUM_X = 1_001

# the scaling factor alpha to use
ALPHA = 1.0

# the orders of the Hermite functions to plot
ORDERS = [int(2 ** (i * 0.3333)) for i in range(0, 35)]

# the Numba-based version of the Hermite functions is pre-compiled to avoid the
# compilation overhead in the performance comparison
hermite_function_basis(
    x=np.linspace(start=X_FROM, stop=X_TO, num=NUM_X),
    n=ORDERS[0],
    alpha=ALPHA,
    jit=True,
)

# === Performance Comparison ===

perfplot.show(
    setup=lambda n: (np.linspace(start=X_FROM, stop=X_TO, num=NUM_X), n, ALPHA),
    kernels=[
        lambda x, n, alpha: hermite_function_basis(x=x, n=n, alpha=alpha, jit=False),
        lambda x, n, alpha: hermite_function_basis(x=x, n=n, alpha=alpha, jit=True),
    ],
    labels=["NumPy", "Numba JIT"],
    n_range=ORDERS,
    logx=True,
    logy=True,
    xlabel="Order of Hermite Function",
    title=f"Performance of Hermite Function for {NUM_X} x-values",
    equality_check=None,
    show_progress=True,
    target_time_per_measurement=0.1,
)
