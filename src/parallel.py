"""Small process-based parallelism helper.

The numerical backends are single-threaded NumPy; the pipeline's parallelism
comes from running independent (dataset, arch, method, seed) jobs in separate
processes.  Four workers is the default on this machine (4 physical cores;
the CPU also exposes 8 SMT threads, which would oversubscribe the FP units).
Override with CL_JOBS=<n>.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Callable, Iterable, Iterator, Sequence


DEFAULT_JOBS = 4


def configure_single_thread() -> None:
    """Pin BLAS/OpenMP runtimes to one thread per worker.

    Must run before NumPy is imported for the environment variables to take
    effect.  Every experiment script imports this module before numpy.
    """
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[key] = "1"


def available_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def get_jobs(requested: int | None = None) -> int:
    if requested is not None:
        return max(1, int(requested))
    env = os.environ.get("CL_JOBS")
    if env:
        return max(1, int(env))
    return min(DEFAULT_JOBS, available_cpus())


def parallel_map(func: Callable, items: Sequence, jobs: int | None = None,
                 chunksize: int = 1) -> Iterator:
    """Ordered parallel map using fork.

    Workers inherit module globals (pre-built task sequences, frozen
    hyperparameters) because the platform is Linux.  Keeping a single mapping
    function and pre-fork globals avoids pickling large datasets for every run.
    """
    jobs = get_jobs(jobs)
    if jobs <= 1 or len(items) <= 1:
        for item in items:
            yield func(item)
        return

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=jobs) as pool:
        yield from pool.imap(func, items, chunksize=chunksize)


def describe(jobs: int | None = None) -> str:
    return f"{get_jobs(jobs)} worker process(es), one BLAS thread per worker"
