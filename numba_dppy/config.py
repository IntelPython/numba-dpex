try:
    import dpctl

    dppy_present = dpctl.has_sycl_platforms() and dpctl.has_gpu_queues()
except:
    dppy_present = False
