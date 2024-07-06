``robust_hermite_ft``
=====================

You want to compute the Fourier transform of a signal, but your signal can be corrupted
by outliers? If so, this package is for you even though you will have to say goodbye to
the *"fast"* in *Fast Fourier Transform* 🏃🙅‍♀️

🏗️🚧 👷👷‍♂️👷‍♀️🏗️🚧

Currently under construction. Please come back later.

〰️ Hermite functions
---------------------

The module ``hermite_functions`` offers a numerically stable way to evaluate Hermite
functions or arbitrary order and argument - that can be scaled with a factor
:math:`\\alpha`

.. image:: docs/hermite_functions/DilatedHermiteFunctions_DifferentScales.png
    :align: center

The Hermite functions are defined as

.. image:: docs\hermite_functions\equations\DilatedHermiteFunctions.png
    :align: center

By making use of logarithm tricks, the evaluation that might involve infinitely high
polynomial values and at the same time infinitely small Gaussians - that are on top of
that scaled by an infinitely high factorial - can be computed safely and yield accurate
results.

For doing so, the equation is rewritten in logarithmic form as

.. image:: docs/hermite_functions/equations/LogDilatedHermiteFunctions.png
    :align: center

where the evaluation of the natural logarithm of the Hermite polynomials is achieved by
making use of the
`logsumexp trick <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html>`_.

This approach is tested against a symbolic evaluation with ``sympy`` that uses 100 digits of precision.