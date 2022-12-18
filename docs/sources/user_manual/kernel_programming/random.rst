.. _rng_kernels:
.. include:: ./../../ext_links.txt

Random number generation in kernel functions
============================================

**Data Parallel Extension for Numba** does not provide support for *random number generation*
from within a kernel function.

Instead random numbers can be seamlessly generated from within an auto-offload function
decorated with the ``@numba_dpex.njit`` decorator.
