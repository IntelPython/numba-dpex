# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numba.core import registry, serialize, dispatcher
from numba import types
from numba.core.errors import UnsupportedError
from numba.core.target_extension import resolve_dispatcher_from_str, target_registry, dispatcher_registry
import dpctl
from numba.core.compiler_lock import global_compiler_lock


class TargetDispatcher(serialize.ReduceMixin, metaclass=dispatcher.DispatcherMeta):
    __numba__ = "py_func"

    target_offload_gpu = "__dppy_offload_gpu__"
    target_offload_cpu = "__dppy_offload_cpu__"
    target_dppy = "dppy"

    def __init__(self, py_func, wrapper, target, parallel_options, compiled=None):

        self.__py_func = py_func
        self.__target = target
        self.__wrapper = wrapper
        self.__compiled = compiled if compiled is not None else {}
        self.__parallel = parallel_options
        self.__doc__ = py_func.__doc__
        self.__name__ = py_func.__name__
        self.__module__ = py_func.__module__

    def __call__(self, *args, **kwargs):
        return self.get_compiled()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.get_compiled(), name)

    def __get__(self, obj, objtype=None):
        return self.get_compiled().__get__(obj, objtype)

    def __repr__(self):
        return self.get_compiled().__repr__()

    @classmethod
    def _rebuild(cls, py_func, wrapper, target, parallel, compiled):
        self = cls(py_func, wrapper, target, parallel, compiled)
        return self

    def get_compiled(self, target=None):
        if target is None:
            target = self.__target

        disp = self.get_current_disp()
        if not disp in self.__compiled.keys():
            with global_compiler_lock:
                if not disp in self.__compiled.keys():
                    self.__compiled[disp] = self.__wrapper(self.__py_func, disp)

        return self.__compiled[disp]

    def __is_with_context_target(self, target):
        return target is None or target == TargetDispatcher.target_dppy

    def get_current_disp(self):
        target = self.__target
        parallel = self.__parallel
        offload = isinstance(parallel, dict) and parallel.get("offload") is True

        if dpctl.is_in_device_context() or offload:
            if not self.__is_with_context_target(target):
                raise UnsupportedError(
                    f"Can't use 'with' context with explicitly specified target '{target}'"
                )
            if parallel is False or (
                isinstance(parallel, dict) and parallel.get("offload") is False
            ):
                raise UnsupportedError(
                    f"Can't use 'with' context with parallel option '{parallel}'"
                )

            from numba_dppy import dppy_offload_dispatcher

            if target is None:
                if dpctl.get_current_device_type() == dpctl.device_type.gpu:
                    return dispatcher_registry[target_registry[
                        TargetDispatcher.target_offload_gpu
                    ]]
                elif dpctl.get_current_device_type() == dpctl.device_type.cpu:
                    return dispatcher_registry[target_registry[
                        TargetDispatcher.target_offload_cpu
                    ]]
                else:
                    if dpctl.is_in_device_context():
                        raise UnsupportedError("Unknown dppy device type")
                    if offload:
                        if dpctl.has_gpu_queues():
                            return dispatcher_registry[target_registry[
                                TargetDispatcher.target_offload_gpu
                            ]]
                        elif dpctl.has_cpu_queues():
                            return dispatcher_registry[target_registry[
                                TargetDispatcher.target_offload_cpu
                            ]]

        if target is None:
            target = "cpu"

        return resolve_dispatcher_from_str(target)


    def _reduce_states(self):
        return dict(
            py_func=self.__py_func,
            wrapper=self.__wrapper,
            target=self.__target,
            parallel=self.__parallel,
            compiled=self.__compiled,
        )
