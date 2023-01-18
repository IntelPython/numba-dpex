.. _caching:

Caching Mechanism in Numba-dpex
================================

Caching is done by saving the compiled kernel code, the ELF object of the executable code. By using the kernel code, cached kernels have minimal overhead because no compilation is needed.

Unlike Numba, we do not perform file-based caching, instead we use an Least Recently Used (LRU) caching mechanism. However when a kernel needs to be evicted, we utilize numba's file-based caching mechanism described `here <https://numba.pydata.org/numba-doc/latest/developer/caching.html>`_.

Algorithm
==========

The caching mechanism for Numba-dpex works as follows: The cache is an LRU cache backed by an ordered dictionary mapped onto a doubly linked list. The tail of the list contains the most recently used (MRU) kernel and the head of the list contains the least recently used (LRU) kernel. The list  has a fixed size. If a new kernel arrives to be cached and if the size is already on the maximum limit, the algorithm evicts the LRU kernel to make room for the MRU kernel. The evicted item will be serialized and pickled into a file using Numba's caching mechanism.

Everytime whenever a kernel needs to be retrieved from the cache, the mechanism will look for the kernel in the cache and will be loaded if it's already present. However, if the program is seeking for a kernel that has been evicted, the algorithm will load it from the file and enqueue in the cache.

Settings
========

Therefore, we employ similar environment variables as used in Numba, i.e. ``NUMBA_CACHE_DIR`` etc. However we add three more environment variables to control the caching mechanism.

- In order to specify cache capacity, one can use ``NUMBA_DPEX_CACHE_SIZE``. By default, it's set to 10.
- ``NUMBA_DPEX_ENABLE_CACHE`` can be used to enable/disable the caching mechanism. By default it's enabled, i.e. set to 1.
- In order to enable the debugging messages related to caching, one can set ``NUMBA_DPEX_DEBUG_CACHE`` to 1. All environment variables are defined in :file:`numba_dpex/config.py`.
