Setting up the machine for debugging
=======================================

Graphics driver
---------------

You need to install drivers from `here <https://dgpu-docs.intel.com/installation-guides/index.html>`_ and from `here <https://software.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/prerequisites/install-intel-gpu-drivers.html>`_.

It is also important that the user is in the group "video" (on Ubuntu* 18, Fedora* 30, and SLES* 15 SP1) or "render" (on Ubuntu* 19 and higher, CentOS* 8, and Fedora* 31). An administrator with sudo or root privilege can change the group owner of /dev/dri/renderD*/dev/dri/card* to a group ID used by your user base:

.. code-block:: bash

    sudo usermod -a -G video <username> 

NEO driver
---------------

NEO driver at least `21.15.19533` version is required to make debugger work correctly.

Follow the `link <https://github.com/intel/compute-runtime/releases/tag/21.15.19533>`_ below to download the drivers.

1) To install the driver on the system, use the command:

.. code-block:: bash

    sudo dpkg -i *.deb


2) To install the NEO driver locally follow the commands below:

.. code-block:: bash

    cd /path/to/my/neo
    for file in `ls *.deb`; do dpkg -x $file .; done
    export MY_ACTIVE_NEO=/path/to/my/neo/usr/local/lib
    export LD_LIBRARY_PATH=${MY_ACTIVE_NEO}:${MY_ACTIVE_NEO}/intel-opencl:$LD_LIBRARY_PATH

The Installable Client Driver (ICD) uses the system implementation for OpenCL by default. You will also need to add environment variables to change the behavior of the ICD. Info about `OCL_ICD_... <https://github.com/KhronosGroup/OpenCL-ICD-Loader>`_. 
Add all needed from "/etc/OpenCL/vendors/" and custom to `OCL_ICD_FILENAMES`.

.. code-block:: bash

    export OCL_ICD_FILENAMES=/path/to/my/neo/usr/local/lib/intel-opencl/libigdrcl.so:/optional/from/vendors/libintelocl.so
    export OCL_ICD_VENDORS=

DCD driver
---------------

To install the DCD driver from oneapi into the system, use the following command:

.. code-block:: bash

    sudo dpkg -i /path/to/oneapi/debugger/latest/igfxdcd-*-Linux.deb 

Before working, you must activate it:

.. code-block:: bash

    sudo modinfo igfxdcd

Also, you must remove the driver from the system if you want to install a different version:

.. code-block:: bash

    sudo dpkg -r igfxdcd

If you are installing DCD for the first time, you need to create keys. More details `here <https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-debugging-dpcpp-linux/top.html>`_.
