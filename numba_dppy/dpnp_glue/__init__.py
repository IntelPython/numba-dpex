def ensure_dpnp(name):
    try:
        # import dpnp
        from . import dpnp_fptr_interface as dpnp_glue
    except ImportError:
        raise ImportError("dpNP is needed to call np.%s" % name)

DEBUG = None
