#!/bin/bash

set -ex

python -m numba.runtests -b -v -m -- numba_dppy.tests

exit 0
