.. include:: ./../ext_links.txt

Caching Mechanism in ``numba-dpex``
===================================

Caching is done by saving the compiled kernel code, the ELF object of the
executable code. By using the kernel code, cached kernels have minimal overhead
because no compilation is needed.

Unlike Numba*, ``numba-dpex`` does not perform an exclusive file-based caching,
instead a Least Recently Used (LRU) caching mechanism is used. However, when a
kernel needs to be evicted, Numba's file-based caching mechanism is invoked as
described `here
<https://numba.pydata.org/numba-doc/latest/developer/caching.html>`_.

Algorithm
----------

The caching mechanism for ``numba-dpex`` works as follows: The cache is an LRU
cache backed by an ordered dictionary mapped onto a doubly linked list. The tail
of the list contains the most recently used (MRU) kernel and the head of the
list contains the least recently used (LRU) kernel. The list  has a fixed size.
If a new kernel arrives to be cached and if the size is already on the maximum
limit, the algorithm evicts the LRU kernel to make room for the MRU kernel. The
evicted item will be serialized and pickled into a file using Numba's caching
mechanism.

Everytime when a kernel needs to be retrieved from the cache, the mechanism
will look for the kernel in the cache and will be loaded if it's already
present. However, if the program is seeking for a kernel that has been evicted,
the algorithm will load it from the file and enqueue in the cache. As a result,
the amount of file operations are significantly lower than that of Numba.

Settings
---------

Therefore, ``numba-dpex`` employs similar environment variables as used in
Numba, i.e. ``NUMBA_CACHE_DIR`` etc. However there are three more environment
variables to control the caching mechanism.

- In order to specify cache capacity, ``NUMBA_DPEX_CACHE_SIZE`` can be used. By
  default, it's set to 10.

- ``NUMBA_DPEX_ENABLE_CACHE`` can be used to enable/disable the caching
  mechanism. By default it's enabled, i.e. set to 1.

- In order to enable the debugging messages related to caching, the variable
``NUMBA_DPEX_DEBUG_CACHE`` can be set to 1. All environment variables are
defined in :file:`numba_dpex/config.py`.
