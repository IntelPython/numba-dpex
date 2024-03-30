.. include:: ./../../ext_links.txt


Scalar mathematical functions from the Python `math`_ module and the `dpnp`_
library can be used inside a kernel function. During compilation the
mathematical functions get compiled into device-specific intrinsic instructions.


.. csv-table:: Current support matrix of ``math`` module functions
   :file: ./math-functions.csv
   :widths: 30, 70
   :header-rows: 1

.. caution::

   The supported signature for some of the ``math`` module functions in the
   compiled mode differs from CPython. The divergence in behavior is a known
   issue. Please refer https://github.com/IntelPython/numba-dpex/issues/759 for
   updates.

.. csv-table:: Current support matrix of ``dpnp`` functions
   :file: ./dpnp-ufuncs.csv
   :widths: auto
   :header-rows: 1
