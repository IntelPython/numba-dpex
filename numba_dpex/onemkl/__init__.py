import llvmlite.binding as ll

from ._dpex_lapack_iface import c_helpers

# Register the helper function in _dpexrt_python so that we can insert
# calls to them via llvmlite.
for (
    py_name,
    c_address,
) in c_helpers.items():
    ll.add_symbol(py_name, c_address)
