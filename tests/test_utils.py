"""
This test suite implements the tests for the module :mod:`_utils`.

"""

# === Imports ===

from typing import Literal, Union

import pytest
from psutil import Process

from robust_hermite_ft._utils import _get_num_workers

# === Types ===

# the type of the number of workers that need to be evaluated dynamically with
# ``psutil``
DynamicWorkers = Literal["__dynamic__"]

# === Constants ===

# the value of the number of workers that need to be evaluated dynamically with
# ``psutil``
DYNAMIC_WORKERS = "__dynamic__"


# === Tests ===


@pytest.mark.parametrize(
    "workers, expected",
    [
        (  # Test 0) 1 worker requested
            1,
            1,
        ),
        (  # Test 1) 1000 workers requested which should be limited to the maximum
            1_000,
            "__dynamic__",
        ),
        (  # Test 2) 0 workers requested which should be limited to the minimum
            0,
            1,
        ),
        (  # Test 3) -1 workers requested which should be limited to the maximum
            -1,
            "__dynamic__",
        ),
        (  # Test 4) -2 workers requested which should raise a ValueError
            -2,
            ValueError("Expected 'workers' to be greater or equal to -1"),
        ),
    ],
)
def test_get_num_workers(workers: int, expected: Union[int, DynamicWorkers, Exception]):
    """
    Tests that the function :func:`_get_num_workers` returns the expected number of
    workers or raises the expected exception.

    """

    # in the case of a ValueError, the exception is raised
    if isinstance(expected, Exception):
        # the function is called and the exception is checked
        with pytest.raises(type(expected), match=str(expected)):
            _get_num_workers(workers=workers)

        return

    # the number of workers is determined
    num_workers = _get_num_workers(workers=workers)

    # the number of workers is checked
    # for the dynamic case, the number of workers is determined with ``psutil``
    if expected == DYNAMIC_WORKERS:
        # the number of workers is determined dynamically
        process = Process()
        expected = len(process.cpu_affinity())  # type: ignore
        del process

    # the check is performed
    assert num_workers == expected
