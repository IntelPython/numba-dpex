Set up the machine for debugging
====================================

Graphics driver
---------------

Install drivers using the following guides:

    - `GPGPU Documents / Installation guides`_
    - `Intel® oneAPI Toolkits Installation Guide for Linux* OS / Install Intel GPU Drivers`_

.. _`GPGPU Documents / Installation guides`: https://dgpu-docs.intel.com/installation-guides/index.html
.. _`Intel® oneAPI Toolkits Installation Guide for Linux* OS / Installation Guide / Install Intel GPU Drivers`:
    https://software.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/prerequisites/install-intel-gpu-drivers.html

The user should be in the "video" group (on Ubuntu* 18, Fedora* 30, and SLES* 15 SP1)
or "render" group (on Ubuntu* 19 and higher, CentOS* 8, and Fedora* 31).
An administrator with sudo or root privileges can change the group owner of `/dev/dri/renderD*` and `/dev/dri/card*`
to a group ID used by your user base:

.. code-block:: bash

    sudo usermod -a -G video <username>

.. _NEO-driver:

NEO driver
----------

NEO driver `21.15.19533` or higher is required to make the debugger work correctly.

1) Download the driver from `GitHub <https://github.com/intel/compute-runtime/releases/tag/21.15.19533>`_.

2) Install the NEO driver on the system or locally.

    * To install the driver on the system, use the command:

        .. code-block:: bash

            sudo dpkg -i *.deb

    * To install the driver locally:

        1) Add the path to NEO files in `LD_LIBRARY_PATH`:

            .. code-block:: bash

                cd /path/to/my/neo
                for file in `ls *.deb`; do dpkg -x $file .; done
                export MY_ACTIVE_NEO=/path/to/my/neo/usr/local/lib
                export LD_LIBRARY_PATH=${MY_ACTIVE_NEO}:${MY_ACTIVE_NEO}/intel-opencl:$LD_LIBRARY_PATH

        2) Add environment variables to change the behavior of the Installable Client Driver (ICD).
            ICD uses the system implementation for OpenCL™ by default. To install the driver locally, add all needed from "/etc/OpenCL/vendors/" and custom to `OCL_ICD_FILENAMES`.
            To overwrite the default behavior, use :samp:`export OCL_ICD_VENDORS=`:

            .. code-block:: bash

                export OCL_ICD_FILENAMES=/path/to/my/neo/usr/local/lib/intel-opencl/libigdrcl.so:/optional/from/vendors/libintelocl.so
                export OCL_ICD_VENDORS=

See also:

  - `Intel(R) Graphics Compute Runtime for oneAPI Level Zero and OpenCL(TM) Driver <https://github.com/intel/compute-runtime>`_
  - `Intel(R) Graphics Compute Runtime Releases <https://github.com/intel/compute-runtime/releases>`_
  - `OpenCL ICD Loader <https://github.com/KhronosGroup/OpenCL-ICD-Loader>`_


.. _debugging-machine-dcd-driver:

Debug companion driver (DCD)
----------------------------

1) To install the DCD from oneAPI into the system, use the following command:

    .. code-block:: bash

        sudo dpkg -i /path/to/oneapi/debugger/latest/igfxdcd-*-Linux.deb

2) Activate the driver:

    .. code-block:: bash

        sudo modinfo igfxdcd

Remove the driver from the system if you want to install a different version:

.. code-block:: bash

    sudo dpkg -r igfxdcd

If you are installing DCD for the first time, create keys. For details, see the link at the end of this page.

See also:

  - `Get Started with Intel® Distribution for GDB* on Linux* OS Host <https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-debugging-dpcpp-linux/top.html>`_
  - `Public signature key <https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-debugging-dpcpp-linux/top.html#:~:text=sudo%20modprobe%20igfxdcd-,The%20host%20system%20does%20not%20recognize%20the%20igfxdcd%20signature%20if,gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB,-If%20you%20have>`_
