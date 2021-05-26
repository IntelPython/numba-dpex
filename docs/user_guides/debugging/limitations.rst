Limitations
===========

Currently, `numba-dppy` provides only initial support of debugging SYCL kernels.
The following functionalities are **not supported**:

  - Printing kernel local variables (e.g. :samp:`info locals`).
  - Stepping over several offloaded functions.


Local variables debugging limitations:

  - No information about variable values while debugging.
  - Information about variable types is limited.
  - Overwriting the value of a variable is not supported.
  - Accessing elements of a complex variable type is not supported.
