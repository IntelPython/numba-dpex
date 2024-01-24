# SPDX-FileCopyrightText: 2022 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import abc


class KernelInterface(metaclass=abc.ABCMeta):
    """An interface for compute kernel that was generated either from a
    Python function object or as a Numba IR FunctionType object.

    Args:
        metaclass (optional): The interface is derived from abc.ABCMeta.

    Raises:
        NotImplementedError: The interface does not implement any of the
        methods and subclasses are required to implement them.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(
            (subclass, "llvm_module")
            and hasattr(subclass, "device_driver_ir_module")
            and hasattr(subclass, "pyfunc_name")
            and hasattr(subclass, "module_name")
            and hasattr(subclass, "compile")
            and callable(subclass.compile)
        )

    # TODO Add a property for argtypes

    @property
    @abc.abstractmethod
    def llvm_module(self):
        """The LLVM IR Module corresponding to the Kernel instance."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def device_driver_ir_module(self):
        """The module in a device IR (such as SPIR-V or PTX) format."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def pyfunc_name(self):
        """The Python function name corresponding to the Kernel instance."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def module_name(self):
        """The LLVM module name for the compiled kernel."""
        raise NotImplementedError

    @abc.abstractmethod
    def compile(self, target_ctx, typing_ctx, args, debug, compile_flags):
        """Abstract method to compile a Kernel instance."""
        raise NotImplementedError
