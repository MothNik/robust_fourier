"""
Module :mod:`_utils.parallel_helpers`

This module provides functionalities to handle parallel computations, e.g.,

- obtaining the number of threads available for the process

"""

# === Imports ===

import psutil

# === Functions ===


def _get_num_workers(workers: int) -> int:
    """
    Gets the number of available workers for the process calling this function.

    Parameters
    ----------
    workers : :class:`int`
        Number of workers requested.

    Returns
    -------
    workers : :class:`int`
        Number of workers available.

    """

    # the number of workers may not be less than -1
    if workers < -1:
        raise ValueError(
            f"Expected 'workers' to be greater or equal to -1 but got {workers}."
        )

    # then, the maximum number of workers is determined ...
    # NOTE: the following does not count the number of total threads, but the number of
    #       threads available to the process calling this function
    process = psutil.Process()
    max_workers = len(process.cpu_affinity())  # type: ignore
    del process

    # ... and overwrites the number of workers if it is set to -1
    workers = max_workers if workers == -1 else workers

    # the number of workers is limited between 1 and the number of available threads
    return max(1, min(workers, max_workers))
