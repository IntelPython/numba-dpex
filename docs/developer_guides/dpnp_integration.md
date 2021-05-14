# DPNP integration

Currently numba-dppy uses C backend librarly of DPNP.

## Integration with DPNP C backend library

Update DPNP functions enum.
Do not forget add array to `dpnp_ext._dummy_liveness_func([a.size, out.size])`.

Do not forget build numba-dppy with current installed version of DPNP.
--- Headers dependency?


### Types matching for Numba and DPNP

T* -> types.voidptr
size_t -> types.intp
long -> types.int64

We are using void * in case of size_t * as Numba currently does not have
any type to represent size_t *. Since, both the types are pointers,
if the compiler allows there should not be any mismatch in the size of
the container to hold different types of pointer.
