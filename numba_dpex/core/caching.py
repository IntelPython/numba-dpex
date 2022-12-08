# Copyright 2022 Intel Corporation
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

import hashlib
from abc import ABCMeta, abstractmethod

from numba.core.caching import CacheImpl, IndexDataCacheFile, _Cache
from numba.core.serialize import dumps

from numba_dpex import config


def build_key(sig, pyfunc, codegen, backend=None, device_type=None):
    """Constructs a key from the data object.

    Compute index key for the given signature and codegen. It includes
    a description of the OS, target architecture and hashes of the bytecode
    for the function and, if the function has a __closure__, a hash of the
    cell_contents.

    Args:
        sig (inspect.Signature): The signature object of a python function.
        codegen (numba.core.codegen.Codegen):
            The codegen object found from the target context.
        backend (enum, optional): A 'backend_type' enum. Defaults to None.
        device_type (enum, optional): A 'device_type' enum. Defaults to None.

    Returns:
        tuple: A tuple of signature, magic_tuple of codegen and another tuple of
                hashcodes from bytecode and cell_contents.
    """
    codebytes = pyfunc.__code__.co_code
    if pyfunc.__closure__ is not None:
        cvars = tuple([x.cell_contents for x in pyfunc.__closure__])
        # Note: cloudpickle serializes a function differently depending
        #       on how the process is launched; e.g. multiprocessing.Process
        cvarbytes = dumps(cvars)
    else:
        cvarbytes = b""

    return (
        sig,
        codegen.magic_tuple(),
        backend,
        device_type,
        (
            hashlib.sha256(codebytes).hexdigest(),
            hashlib.sha256(cvarbytes).hexdigest(),
        ),
    )


class _CacheImpl(CacheImpl):
    """Implementation of `CacheImpl` to be used by subclasses of `_Cache`.

    This class is an implementation of `CacheImpl` to be used by subclasses of `_Cache`.
    To be assigned in `_impl_class`. Implements the more common and core mechanism for the
    caching.

    """

    def reduce(self, data):
        """Serialize an object before caching.

        Args:
            data (object): The object to be serialized before pickling.
        """
        # TODO: Implement, but looks like we might not need it at all.
        # Look at numba.core.caching for how to implement.
        pass

    def rebuild(self, target_context, reduced_data):
        """Deserialize after unpickling from the cache.

        Args:
            target_context (numba_dpex.core.target.DpexTargetContext):
                The target context for the kernel.
            reduced_data (object): The data to be deserialzed after unpickling.
        """
        # TODO: Implement, but looks like we might not need it at all.
        # Look at numba.core.caching for how to implement.
        pass

    def check_cachable(self, cres):
        """Check if a certain object is cacheable.

        Args:
            cres (object): The object to be cached. For example, if the object is
            `CompileResult`, then you might want to follow the similar checks as
            has been done in `numba.core.caching.CompileResultCacheImpl`.

        Returns:
            bool: Return `True` if cacheable, otherwise `False`.
        """
        # TODO: Although, for the time being, assuming all Kernels in numba_dpex
        # are always cachable. However, we might need to add some bells and
        # whistles in the future. Look at numba.core.caching for how to implement.
        return True


class AbstractCache(metaclass=ABCMeta):
    @abstractmethod
    def get(self):
        """_summary_"""

    @abstractmethod
    def put(self, key, value):
        """_summary_

        Args:
            key (_type_): _description_
            value (_type_): _description_

        Returns:
            _type_: _description_
        """


class NullCache(AbstractCache):
    def get(self, key):
        return None

    def put(self, key, value):
        pass


class Node:
    """A 'Node' class for LRUCache."""

    def __init__(self, key, value):
        """Constructor for the Node

        Args:
            key (object): The key to the value.
            value (object): The data to be saved.
        """
        self.key = key
        self.value = value
        self.next = None
        self.previous = None

    def __str__(self):
        """__str__ for Node

        Returns:
            str: str: A human readable representation of a Node.
        """
        return "(" + str(self.key) + ": " + str(self.value) + ")"

    def __repr__(self):
        """__repr__ for Node

        Returns:
            sstr: A human readable representation of a Node.
        """
        return self.__str__()


class LRUCache(AbstractCache):
    """LRUCache implementation for caching kernels, functions and modules

    The cache is basically a doubly-linked-list backed with a dictionary
    as a lookup table.
    """

    def __init__(self, capacity=10, pyfunc=None):
        """Constructor for LRUCache

        Args:
            capacity (int, optional): The max capacity of the cache. Defaults to 10.
            pyfunc (NoneType, optional): A python function to be cached. Defaults to None.
        """
        self._capacity = capacity
        self._lookup = {}
        self._evicted = {}
        self._dummy = Node(0, 0)
        self._head = self._dummy.next
        self._tail = self._dummy.next
        self._pyfunc = pyfunc
        print(type(self._pyfunc))
        self._cache_file = None
        # if pyfunc is specified, we will use files for evicted items
        if self._pyfunc is not None:
            # _CacheImpl object to be used
            self._impl_class = _CacheImpl
            self._impl = self._impl_class(self._pyfunc)
            self._cache_path = self._impl.locator.get_cache_path()
            # This may be a bit strict but avoids us maintaining a magic number
            source_stamp = self._impl.locator.get_source_stamp()
            filename_base = self._impl.filename_base
            self._cache_file = IndexDataCacheFile(
                cache_path=self._cache_path,
                filename_base=filename_base,
                source_stamp=source_stamp,
            )

    @property
    def head(self):
        """Get the head of the cache

        Returns:
            Node: The head of the cache.
        """
        return self._head

    @property
    def tail(self):
        """Get the tail of the cache

        Returns:
            Node: The tail of the cache.
        """
        return self._tail

    @property
    def evicted(self):
        """Get the list of evicted items from the cache

        Returns:
            dict: A table of evicted items from the cache.
        """
        return self._evicted

    def __str__(self):
        """__str__ function for the cache

        Returns:
            str: A human readable representation of the cache.
        """
        items = []
        current = self._head
        while current:
            items.append(str(current))
            current = current.next
        return "{" + ", ".join(items) + "}"

    def __repr__(self):
        """__repr__ function for the cache

        Returns:
            str: A human readable representation of the cache.
        """
        return self.__str__()

    def clean(self):
        """Clean the cache"""
        self._lookup = {}
        # self._evicted = {}
        self._dummy = Node(0, 0)
        self._head = self._dummy.next
        self._tail = self._dummy.next

    def _remove_head(self):
        """Remove the head of the cache"""
        if not self._head:
            return
        prev = self._head
        self._head = self._head.next
        if self._head:
            self._head.previous = None
        del prev

    def _append_tail(self, new_node):
        """Add the new node to the tail end"""
        if not self._tail:
            self._head = self._tail = new_node
        else:
            self._tail.next = new_node
            new_node.previous = self._tail
            self._tail = self._tail.next

    def _unlink_node(self, node):
        """Unlink current linked node"""
        if self._head is node:
            self._head = node.next
            if node.next:
                node.next.previous = None
            node.previous, node.next = None, None
            return

        # removing the node from somewhere in the middle; update pointers
        prev, nex = node.previous, node.next
        prev.next = nex
        nex.previous = prev
        node.previous, node.next = None, None

    def get(self, key):
        """Get the value associated with the key

        Args:
            key (object): A key for the lookup table.

        Returns:
            object: The value associated with the key.
        """
        print(
            "[cache] size: {0:d}, loading artifact, key: {1:s}".format(
                len(self._lookup), str(key)
            )
        )
        if config.DEBUG_CACHE:
            print("[cache] Contents: ", self.__str__())
        if key not in self._lookup:
            if key not in self._evicted:
                return None
            elif self._cache_file:
                if config.DEBUG_CACHE:
                    print(
                        "[cache]: unpickling an evicted item, indexed at {0:s}.".format(
                            self._cache_file._index_path
                        )
                    )
                value = self._cache_file.load(key)
            else:
                value = self._evicted[key]
            node = Node(key, value)
        else:
            node = self._lookup[key]

        if node is not self._tail:
            self._unlink_node(node)
            self._append_tail(node)

        return node.value

    def put(self, key, value):
        """Store the key-value pair into the cache

        Args:
            key (object): The key for the data.
            value (object): The data to be saved.
        """
        print(
            "[cache] size: {0:d}, saving artifact, key: {1:s}".format(
                len(self._lookup), str(key)
            )
        )
        if config.DEBUG_CACHE:
            print("[cache] Contents: ", self.__str__())
        if key in self._lookup:
            self._lookup[key].value = value
            self.get(key)
            return

        if key in self._evicted:
            self._evicted.pop(key)

        if len(self._lookup) == self._capacity:
            # remove head node and correspond key
            if config.DEBUG_CACHE:
                print(
                    "[cache] size: {0:d}, capacity exceeded, evicting".format(
                        len(self._lookup)
                    ),
                    self._head,
                )
            if self._cache_file:
                if config.DEBUG_CACHE:
                    print(
                        "[cache] pickling the evicted item, indexed at {0:s}.".format(
                            self._cache_file._index_path
                        )
                    )
                self._cache_file.save(self._head.key, self._head.value)
                self._evicted[
                    self._head.key
                ] = None  # as we are using cache files, we save memory
            else:
                self._evicted[self._head.key] = self._head.value
            self._lookup.pop(self._head.key)
            self._remove_head()

        # add new node and hash key
        new_node = Node(key, value)
        self._lookup[key] = new_node
        self._append_tail(new_node)
