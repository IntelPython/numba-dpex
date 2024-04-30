# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import abc


class KernelTemplateInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(
            callable(subclass._generate_kernel_stub_as_string)
            and callable(subclass._generate_kernel_ir)
            and callable(subclass.dump_kernel_string)
            and callable(subclass.dump_kernel_ir)
            and hasattr(subclass, "kernel_ir")
            and hasattr(subclass, "kernel_string")
        )

    @abc.abstractmethod
    def _generate_kernel_stub_as_string(self):
        """Generates as a string a stub for a numba_dpex kernel function"""
        raise NotImplementedError

    @abc.abstractmethod
    def _generate_kernel_ir(self):
        raise NotImplementedError

    @abc.abstractmethod
    def dump_kernel_string(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def py_func(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def kernel_string(self):
        raise NotImplementedError
