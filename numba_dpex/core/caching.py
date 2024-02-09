# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import sys
from abc import ABCMeta, abstractmethod

from numba.core.caching import CacheImpl, IndexDataCacheFile

from numba_dpex.core import config


class _CacheImpl(CacheImpl):
    """Implementation of `CacheImpl` to be used by subclasses of `_Cache`.

    This class is an implementation of `CacheImpl` to be used by subclasses
    of `_Cache`. To be assigned in `_impl_class`. Implements the more common
    and core mechanism for the caching.

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
            target_context (numba_dpex.core.target.DpexKernelTargetContext):
                The target context for the kernel.
            reduced_data (object): The data to be deserialzed after unpickling.
        """
        # TODO: Implement, but looks like we might not need it at all.
        # Look at numba.core.caching for how to implement.
        pass

    def check_cachable(self, cres):
        """Check if a certain object is cacheable.

        Args:
            cres (object): The object to be cached. For example, if the object
            is `CompileResult`, then you might want to follow the similar
            checks as has been done in
            `numba.core.caching.CompileResultCacheImpl`.

        Returns:
            bool: Return `True` if cacheable, otherwise `False`.
        """
        # TODO: Although, for the time being, assuming all Kernels in numba_dpex
        # are always cachable. However, we might need to add some bells and
        # whistles in the future. Look at numba.core.caching for how to
        # implement.
        return True


class AbstractCache(metaclass=ABCMeta):
    """Abstract cache class to specify basic caching operations.

    This class will be used to create an non-functional dummy cache
    (i.e. NullCache) and other functional cache. The dummy cache
    will be used as a placeholder when caching is disabled.

    Args:
        metaclass (type, optional): Metaclass for the abstract class.
            Defaults to ABCMeta.
    """

    @abstractmethod
    def get(self):
        """An abstract method to retrieve item from the cache."""

    @abstractmethod
    def put(self, key, value):
        """An abstract method to save item into the cache.

        Args:
            key (object): The key for the data
                (i.e. compiled kernel/function etc.).
            value (object): The data (i.e. compiled kernel/function)
                to be saved.
        """


class NullCache(AbstractCache):
    """A dummy cache used if user decides to disable caching.

    If the caching is disabled this class will be used to
    perform all caching operations, all of which will be basically
    NOP. This idea is copied from numba.

    Args:
        AbstractCache (class): The abstract cache from which all
        other caching classes will be derived.
    """

    def get(self, key):
        """Function to get an item (i.e. compiled kernel/function)
        from the cache

        Args:
            key (object): The key to retrieve the
                data (i.e. compiled kernel/function)

        Returns:
            None: Returns None.
        """
        return None

    def put(self, key, value):
        """Function to save a compiled kernel/function
        into the cache.

        Args:
            key (object): The key to the data (i.e. compiled kernel/function).
            value (object): The data to be cached (i.e.
            compiled kernel/function).
        """
        pass


class Node:
    """A 'Node' class for LRUCache."""

    def __init__(self, key, value):
        """Constructor for the Node.

        Args:
            key (object): The key to the value.
            value (object): The data to be saved.
        """
        self.key = key
        self.value = value
        self.next = None
        self.previous = None

    def __str__(self):
        """__str__ for Node.

        Returns:
            str: A human readable representation of a Node.
        """
        return "(" + str(self.key) + ": " + str(self.value) + ")"

    def __repr__(self):
        """__repr__ for Node

        Returns:
            str: A human readable representation of a Node.
        """
        return self.__str__()


class LRUCache(AbstractCache):
    """LRUCache implementation for caching kernels,
    functions and modules.

    The cache is basically a doubly-linked-list backed
    with a dictionary as a lookup table.
    """

    def __init__(self, name="cache", capacity=10, pyfunc=None):
        """Constructor for LRUCache.

        Args:
            name (str, optional): The name of the cache, useful for
                debugging.
            capacity (int, optional): The max capacity of the cache.
                Defaults to 10.
            pyfunc (NoneType, optional): A python function to be cached.
                Defaults to None.
        """
        self._name = name
        self._capacity = capacity
        self._lookup = {}
        self._evicted = {}
        self._dummy = Node(0, 0)
        self._head = self._dummy.next
        self._tail = self._dummy.next
        self._pyfunc = pyfunc
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
        """Get the head of the cache.

        This is used for testing/debugging purposes.

        Returns:
            Node: The head of the cache.
        """
        return self._head

    @property
    def tail(self):
        """Get the tail of the cache.

        This is used for testing/debugging purposes.

        Returns:
            Node: The tail of the cache.
        """
        return self._tail

    @property
    def evicted(self):
        """Get the list of evicted items from the cache.

        This is used for testing/debugging purposes.

        Returns:
            dict: A table of evicted items from the cache.
        """
        return self._evicted

    def _get_memsize(self, obj, seen=None):
        """Recursively finds size of *almost any* object.

        This function might be useful in the future when
        size based (not count based) cache limit will be
        implemented.

        Args:
            obj (object): Any object.
            seen (set, optional): Set of seen object id().
                Defaults to None.

        Returns:
            int: Size of the object in bytes.
        """
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        # Important mark as seen *before* entering recursion to gracefully
        # handle self-referential objects
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum([self._get_memsize(v, seen) for v in obj.values()])
            size += sum([self._get_memsize(k, seen) for k in obj.keys()])
        elif hasattr(obj, "__dict__"):
            size += self._get_memsize(obj.__dict__, seen)
        elif hasattr(obj, "__iter__") and not isinstance(
            obj, (str, bytes, bytearray)
        ):
            size += sum([self._get_memsize(i, seen) for i in obj])
        return size

    def size(self):
        """Get the current size of the cache.

        Returns:
            int: The current number of items in the cache.
        """
        return len(self._lookup)

    def memsize(self):
        """Get the total memory size of the cache.

        This function might be useful in the future when
        size based (not count based) cache limit will be
        implemented.

        Returns:
            int: Get the total memory size of the cache in bytes.
        """
        size = 0
        current = self._head
        while current:
            size = size + self._get_memsize(current.value)
            current = current.next
        return size

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
        self._evicted = {}
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
        if node is None:
            return

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
        """Get the value associated with the key.

        Args:
            key (object): A key for the lookup table.

        Returns:
            object: The value associated with the key.
        """

        if key not in self._lookup:
            if key not in self._evicted:
                return None
            elif self._cache_file:
                value = self._cache_file.load(key)
                if config.DEBUG_CACHE:
                    print(
                        "[{0:s}]: unpickled an evicted artifact, "
                        "key: {1:s}.".format(self._name, str(key))
                    )
            else:
                value = self._evicted[key]
            self.put(key, value)
            return value
        else:
            if config.DEBUG_CACHE:
                print(
                    "[{0:s}] size: {1:d}, loading artifact, key: {2:s}".format(
                        self._name, len(self._lookup), str(key)
                    )
                )
            node = self._lookup[key]

        if node is not self._tail:
            self._unlink_node(node)
            self._append_tail(node)

        return node.value

    def put(self, key, value):
        """Store the key-value pair into the cache.

        Args:
            key (object): The key for the data.
            value (object): The data to be saved.
        """
        if key in self._lookup:
            if config.DEBUG_CACHE:
                print(
                    "[{0:s}] size: {1:d}, storing artifact, key: {2:s}".format(
                        self._name, len(self._lookup), str(key)
                    )
                )
            node = self._lookup[key]
            node.value = value

            if node is not self._tail:
                self._unlink_node(node)
                self._append_tail(node)

            return

        if key in self._evicted:
            self._evicted.pop(key)

        if len(self._lookup) >= self._capacity:
            # remove head node and correspond key
            if self._cache_file:
                if config.DEBUG_CACHE:
                    print(
                        "[{0:s}] size: {1:d}, pickling the LRU item, "
                        "key: {2:s}, indexed at {3:s}.".format(
                            self._name,
                            len(self._lookup),
                            str(self._head.key),
                            self._cache_file._index_path,
                        )
                    )
                self._cache_file.save(self._head.key, self._head.value)
                self._evicted[self._head.key] = (
                    None  # as we are using cache files, we save memory
                )
            else:
                self._evicted[self._head.key] = self._head.value
            self._lookup.pop(self._head.key)
            if config.DEBUG_CACHE:
                print(
                    "[{0:s}] size: {1:d}, capacity exceeded, evicted".format(
                        self._name, len(self._lookup)
                    ),
                    self._head.key,
                )
            self._remove_head()

        # add new node and hash key
        new_node = Node(key, value)
        self._lookup[key] = new_node
        self._append_tail(new_node)
        if config.DEBUG_CACHE:
            print(
                "[{0:s}] size: {1:d}, saved artifact, key: {2:s}".format(
                    self._name, len(self._lookup), str(key)
                )
            )
