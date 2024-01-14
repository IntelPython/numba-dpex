# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import hashlib

from numba.core.serialize import dumps

from numba_dpex.core.types import USMNdArray


def build_key(*args):
    """Constructs key from variable list of args

    Args:
       *args: List of components to construct key
    Return:
       Tuple of args
    """
    return tuple(args)


def create_func_hash(pyfunc):
    """Creates a tuple of sha256 hashes out of code and
    variable bytes extracted from the compiled funtion.

    Args:
       pyfunc: Python function object
    Return:
       Tuple of hashes of code and variable bytes
    """
    codebytes = pyfunc.__code__.co_code
    if pyfunc.__closure__ is not None:
        try:
            cvars = tuple([x.cell_contents for x in pyfunc.__closure__])
            # Note: cloudpickle serializes a function differently depending
            #       on how the process is launched; e.g. multiprocessing.Process
            cvarbytes = dumps(cvars)
        except:
            cvarbytes = b""  # a temporary solution for function template
    else:
        cvarbytes = b""

    return (
        hashlib.sha256(codebytes).hexdigest(),
        hashlib.sha256(cvarbytes).hexdigest(),
    )


def strip_usm_metadata(argtypes):
    """Convert the USMNdArray to an abridged type that disregards the
    usm_type, device, queue, address space attributes.

    Args:
       argtypes: List of types

    Return:
       Tuple of types after removing USM metadata from USMNdArray type
    """

    stripped_argtypes = []
    for argty in argtypes:
        if isinstance(argty, USMNdArray):
            stripped_argtypes.append((argty.ndim, argty.dtype, argty.layout))
        else:
            stripped_argtypes.append(argty)

    return tuple(stripped_argtypes)
