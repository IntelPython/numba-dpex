"""Tests for numba_dppy._usm_allocators_ext
"""


def test_members():
    from numba_dppy import _usm_allocators_ext

    members = ["c_helpers", "get_external_allocator"]

    for member in members:
        assert hasattr(_usm_allocators_ext, member)


def test_c_helpers():
    from numba_dppy._usm_allocators_ext import c_helpers

    functions = [
        "usmarray_get_ext_allocator",
        "create_allocator",
        "release_allocator",
        "DPRT_MemInfo_new",
        "create_queue",
    ]

    assert len(functions) == len(c_helpers)

    for fn_name in functions:
        assert fn_name in c_helpers
        assert isinstance(c_helpers[fn_name], int)


def test_allocator():
    from ctypes import POINTER, PYFUNCTYPE, Structure, c_int, c_size_t, c_void_p

    from numba_dppy._usm_allocators_ext import c_helpers

    class NRT_ExternalAllocator(Structure):
        _fields_ = [
            ("malloc", PYFUNCTYPE(c_void_p, c_size_t, c_void_p)),
            ("realloc", c_void_p),
            ("free", PYFUNCTYPE(None, c_void_p, c_void_p)),
            ("opaque_data", c_void_p),
        ]

    fnty = PYFUNCTYPE(POINTER(NRT_ExternalAllocator), c_int)
    create_allocator = fnty(c_helpers["create_allocator"])

    fnty = PYFUNCTYPE(None, c_void_p)
    release_allocator = fnty(c_helpers["release_allocator"])

    for usm_type in (0, 1, 2):
        allocator = create_allocator(usm_type)
        assert allocator
        assert allocator.contents.malloc
        assert allocator.contents.realloc is None
        assert allocator.contents.free
        assert allocator.contents.opaque_data

        data = allocator.contents.malloc(10, allocator.contents.opaque_data)
        assert data
        allocator.contents.free(data, allocator.contents.opaque_data)

        release_allocator(allocator)


def test_meminfo():
    from ctypes import POINTER, PYFUNCTYPE, Structure, c_int, c_size_t, c_void_p

    from numba.core.runtime import _nrt_python

    from numba_dppy._usm_allocators_ext import c_helpers

    class MemInfo(Structure):
        _fields_ = [
            ("refct", c_size_t),
            ("dtor", c_void_p),
            ("dtor_info", c_void_p),
            ("data", c_void_p),
            ("size", c_size_t),
            ("external_allocator", c_void_p),
        ]

    fnty = PYFUNCTYPE(POINTER(MemInfo), c_size_t, c_int, c_void_p)
    DPRT_MemInfo_new = fnty(c_helpers["DPRT_MemInfo_new"])

    create_queue = PYFUNCTYPE(c_void_p)(c_helpers["create_queue"])

    fnty = PYFUNCTYPE(None, POINTER(MemInfo))
    MemInfo_release = fnty(_nrt_python.c_helpers["MemInfo_release"])

    for usm_type in (0, 1, 2):
        queue = create_queue()
        mip = DPRT_MemInfo_new(10, usm_type, queue)
        mi = mip.contents

        assert mi.refct == 1
        assert mi.dtor
        assert mi.dtor_info
        assert mi.data
        assert mi.size == 10
        assert not mi.external_allocator

        MemInfo_release(mip)
