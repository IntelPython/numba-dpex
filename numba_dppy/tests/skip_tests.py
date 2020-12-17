import dpctl

def is_gen12(device_type):
    with dpctl.device_context(device_type):
        q = dpctl.get_current_queue()
        device = q.get_sycl_device()
        name = device.get_device_name()
        if "Gen12" in name:
            return True

        return False
