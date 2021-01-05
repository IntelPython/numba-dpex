from numba_dppy.ocl.stubs import Stub


class dpnp(Stub):
    """dpnp namespace"""

    _description_ = "<dpnp>"

    class convert_ndarray_to_usm(Stub):
        pass

    class sum(Stub):
        pass

    class eig(Stub):
        pass

    class prod(Stub):
        pass

    class max(Stub):
        pass

    class amax(Stub):
        pass

    class min(Stub):
        pass

    class amin(Stub):
        pass

    class mean(Stub):
        pass

    class median(Stub):
        pass

    class argmax(Stub):
        pass

    class argmin(Stub):
        pass

    class argsort(Stub):
        pass

    class cov(Stub):
        pass

    class dot(Stub):
        pass

    class matmul(Stub):
        pass
