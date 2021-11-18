Limitations
===========

Currently Numba-dppy provides only initial support of debugging SYCL* kernels.
The following functionality is **limited** or **not supported**:

  - Information about variable values may not match the actual values while debugging.
  - Information about variable types is limited.
  - Printing kernel arguments is not supported.
  - Overwriting the value of a variable is not supported.
  - Accessing elements of a complex variable type is not supported.
