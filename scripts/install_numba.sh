#!/bin/bash

THEDIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))

. ${THEDIR}/0.env.sh

git clone https://github.com/IntelPython/numba.git

cd numba
python setup.py develop
