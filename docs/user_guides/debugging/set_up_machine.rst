Setting up the machine for debugging
====================================

Graphics driver
---------------

You need to install drivers using following guides:

    - `GPGPU Documents / Installation guides`_
    - `Intel® oneAPI Toolkits Installation Guide for Linux* OS / Installation Guide / Install Intel GPU Drivers`_

.. _`GPGPU Documents / Installation guides`: https://dgpu-docs.intel.com/installation-guides/index.html
.. _`Intel® oneAPI Toolkits Installation Guide for Linux* OS / Installation Guide / Install Intel GPU Drivers`:
    https://software.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/prerequisites/install-intel-gpu-drivers.html

It is also important that the user is in the group "video" (on Ubuntu* 18, Fedora* 30, and SLES* 15 SP1)
or "render" (on Ubuntu* 19 and higher, CentOS* 8, and Fedora* 31).
An administrator with sudo or root privilege can change the group owner of `/dev/dri/renderD*` and `/dev/dri/card*`
to a group ID used by your user base:

.. code-block:: bash

    sudo usermod -a -G video <username>

.. _NEO-driver:

NEO driver
----------

NEO driver at least `21.15.19533` version is required to make debugger work correctly.

You can download the driver from the following `link <https://github.com/intel/compute-runtime/releases/tag/21.15.19533>`_.

1) To install the driver on the system, use the command:

.. code-block:: bash

    sudo dpkg -i *.deb

2) To install the NEO driver locally, you need to add the path to NEO files in `LD_LIBRARY_PATH`. Follow the commands below:

.. code-block:: bash

    cd /path/to/my/neo
    for file in `ls *.deb`; do dpkg -x $file .; done
    export MY_ACTIVE_NEO=/path/to/my/neo/usr/local/lib
    export LD_LIBRARY_PATH=${MY_ACTIVE_NEO}:${MY_ACTIVE_NEO}/intel-opencl:$LD_LIBRARY_PATH

The Installable Client Driver (ICD) uses the system implementation for OpenCL by default.
You will also need to add environment variables to change the behavior of the ICD.
Add all needed from "/etc/OpenCL/vendors/" and custom to `OCL_ICD_FILENAMES`.
To overwrite the default behavior, use :samp:`export OCL_ICD_VENDORS=`.

.. code-block:: bash

    export OCL_ICD_FILENAMES=/path/to/my/neo/usr/local/lib/intel-opencl/libigdrcl.so:/optional/from/vendors/libintelocl.so
    export OCL_ICD_VENDORS=

See also:

  - `Intel(R) Graphics Compute Runtime for oneAPI Level Zero and OpenCL(TM) Driver <https://github.com/intel/compute-runtime>`_
  - `Intel(R) Graphics Compute Runtime Releases <https://github.com/intel/compute-runtime/releases>`_
  - `OpenCL ICD Loader <https://github.com/KhronosGroup/OpenCL-ICD-Loader>`_


DCD driver
----------

To install the DCD (Debugger Companion Driver) driver from oneAPI into the system, use the following command:

.. code-block:: bash

    sudo dpkg -i /path/to/oneapi/debugger/latest/igfxdcd-*-Linux.deb

Before working, you must activate it:

.. code-block:: bash

    sudo modinfo igfxdcd

Also, you must remove the driver from the system if you want to install a different version:

.. code-block:: bash

    sudo dpkg -r igfxdcd

If you are installing DCD for the first time, you need to create keys. For more details, see the link at the end of this page.

See also:

  - `Get Started with Intel® Distribution for GDB* on Linux* OS Host <https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-debugging-dpcpp-linux/top.html>`_
  - `Public signature key <https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-debugging-dpcpp-linux/top.html#:~:text=sudo%20modprobe%20igfxdcd-,The%20host%20system%20does%20not%20recognize%20the%20igfxdcd%20signature%20if,gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB,-If%20you%20have>`_
