#!/bin/bash

# See https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal.html

check_package_installed() {
  apt list --installed 2>/dev/null "$1" | grep "$1" || echo "$1 not installed"
}

check_package_installed    intel-opencl-icd
check_package_installed    intel-level-zero-gpu
check_package_installed    level-zero
check_package_installed    intel-media-va-driver-non-free
check_package_installed    libmfx1

check_package_installed    libigc-dev
check_package_installed    intel-igc-cm
check_package_installed    libigdfcl-dev
check_package_installed    libigfxcmrt-dev
check_package_installed    level-zero-dev
