``robust_hermite_ft``
=====================
.. image:: https://img.shields.io/badge/python-3.9-blue.svg
    :target: https://www.python.org/downloads/release/python-390/
.. image:: https://img.shields.io/badge/python-3.10-blue.svg
    :target: https://www.python.org/downloads/release/python-3100/
.. image:: https://img.shields.io/badge/python-3.11-blue.svg
    :target: https://www.python.org/downloads/release/python-3110/
.. image:: https://img.shields.io/badge/python-3.12-blue.svg
    :target: https://www.python.org/downloads/release/python-3120/
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
.. image:: https://img.shields.io/badge/code%20style-isort-000000.svg
    :target: https://pycqa.github.io/isort/
    

You want to compute the Fourier transform of a signal, but your signal can be corrupted
by outliers? If so, this package is for you even though you will have to say goodbye to
the *"fast"* in *Fast Fourier Transform* 🏃🙅‍♀️

🏗️🚧 👷👷‍♂️👷‍♀️🏗️🚧

Currently under construction. Please check back later.

〰️ Hermite functions
---------------------

Being the eigenfunctions of the Fourier transform, Hermite functions are excellent
candidates for the basis functions for a Least Squares Regression approach to the Fourier
transform. However, their evaluation can be a bit tricky.

The module ``hermite_functions`` offers a numerically stable way to evaluate Hermite
functions or arbitrary order :math:`n` and argument - that can be scaled with a factor
:math:`{\alpha}`

.. image:: docs/hermite_functions/DilatedHermiteFunctions_DifferentScales.png
    :width: 1000px
    :align: center

The Hermite functions are defined as

.. image:: docs/hermite_functions/equations/DilatedHermiteFunctions.png
    :width: 500px
    :align: left

with the Hermite polynomials

.. image:: docs/hermite_functions/equations/DilatedHermitePolynomials.png
    :width: 681px
    :align: left

By making use of logarithm tricks, the evaluation that might involve infinitely high
polynomial values and at the same time infinitely small Gaussians - that are on top of
that scaled by an infinitely high factorial - can be computed safely and yield accurate
results.

For doing so, the relation between the dilated and the non-dilated Hermite functions

.. image:: docs/hermite_functions/equations/HermiteFunctions_UndilatedToDilated.png
    :width: 321px
    :align: left

and the recurrence relation for the Hermite functions

.. image:: docs/hermite_functions/equations/HermiteFunctions_RecurrenceRelation.png
    :width: 699px
    :align: left

are used, but not directly. Instead, the latest evaluated Hermite function is kept at a
value of either -1, 0, or +1 during the recursion and the logarithm of a correction
factor is tracked and applied when the respective Hermite function is finally evaluated
and stored. This approach is based on [1_].

This approach is tested against a symbolic evaluation with ``sympy`` that uses 200
digits of precision and it can be shown that even orders as high as 2,000 can still be
computed even though neither the polynomial, the Gaussian nor the factorial can be
evaluated for this anymore. The factorial for example would already have overflown for
orders of 170 in ``float64``-precision.

.. image:: docs/hermite_functions/DilatedHermiteFunctions_Stability.png
    :width: 1000px
    :align: center

As a sanity check, their orthogonality is part of the tests together with a test for
the fact that the absolute values of the Hermite functions for real input cannot exceed
the value :math:`\frac{\pi^{-\frac{1}{4}}}{\sqrt{\alpha}}`.

References
----------
.. [1] Bunck B. F., A fast algorithm for evaluation of normalized Hermite
    functions, BIT Numer Math (2009), 49, pp. 281–295, DOI:
    `<https://doi.org/10.1007/s10543-009-0216-1>`_