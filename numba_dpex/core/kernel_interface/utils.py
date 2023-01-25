class Ranges:
    """A data structure to encapsulate kernel lauch parameters.

    This is just a wrapper class on top of tuples. The kernel
    launch parameter is consisted of two int's (or two tuples of int's).
    The first value is called `global_range` and the second value
    is called `local_range`.

    The `global_range` is analogous to DPC++'s "global size"
    and the `local_range` is analogous to DPC++'s "workgroup size",
    respectively.
    """

    def __init__(self, global_range, local_range=None):
        """Constructor for Ranges.

        Args:
            global_range (tuple or int): An int or a tuple of int's
                to specify DPC++'s global size.
            local_range (tuple, optional): An int or a tuple of int's
                to specify DPC++'s workgroup size. Defaults to None.
        """
        self._global_range = global_range
        self._local_range = local_range
        self._check_sanity()

    def _check_sanity(self):
        """Sanity checks for the global and local range tuples.

        Raises:
            ValueError: If the length of global_range is more than 3, if tuple.
            ValueError: If each of value global_range is not an int, if tuple.
            ValueError: If the global_range is not a tuple or an int.
            ValueError: If the length of local_range is more than 3, if tuple.
            ValueError: If the dimensions of local_range
                        and global_range are not same, if tuples.
            ValueError: If each of value local_range is not an int, if tuple.
            ValueError: If the range limits in the global_range is not
                        divisible by the range limit in the local_range
                        at the corresponding dimension.
            ValueError: If the local_range is not a tuple or an int.
        """
        if isinstance(self._global_range, tuple):
            if len(self._global_range) > 3:
                raise ValueError(
                    "The maximum allowed dimension for global_range is 3."
                )
            for i in range(len(self._global_range)):
                if not isinstance(self._global_range[i], int):
                    raise ValueError("The range limit values must be an int.")
        elif isinstance(self._global_range, int):
            self._global_range = tuple([self._global_range])
        else:
            raise ValueError("global_range must be a tuple or an int.")
        if self._local_range:
            if isinstance(self._local_range, tuple):
                if len(self._local_range) > 3:
                    raise ValueError(
                        "The maximum allowed dimension for local_range is 3."
                    )
                if len(self._global_range) != len(self._local_range):
                    raise ValueError(
                        "global_range and local_range must "
                        + "have the same dimensions."
                    )
                for i in range(len(self._local_range)):
                    if not isinstance(self._local_range[i], int):
                        raise ValueError(
                            "The range limit values must be an int."
                        )
                    if self._global_range[i] % self._local_range[i] != 0:
                        raise ValueError(
                            "Each limit in global_range must be divisible "
                            + "by each limit in local_range at "
                            + " the corresponding dimension."
                        )
            elif isinstance(self._local_range, int):
                self._local_range = tuple([self._local_range])
            else:
                raise ValueError("local_range must be a tuple or an int.")

    @property
    def global_range(self):
        """global_range accessor.

        Returns:
            tuple: global_range
        """
        return self._global_range

    @property
    def local_range(self):
        """local_range accessor.

        Returns:
            tuple: local_range
        """
        return self._local_range

    def __str__(self) -> str:
        """str() function for this class.

        Returns:
            str: str representation of a Ranges object.
        """
        return (
            "(" + str(self._global_range) + ", " + str(self._local_range) + ")"
        )

    def __repr__(self) -> str:
        """repr() function for this class.

        Returns:
            str: str representation of a Ranges object.
        """
        return self.__str__()


# tester
if __name__ == "__main__":
    ranges = Ranges(1)
    print(ranges)

    ranges = Ranges(1, 1)
    print(ranges)

    ranges = Ranges((2, 2, 2), (1, 1, 1))
    print(ranges)

    ranges = Ranges((2, 2, 2))
    print(ranges)

    try:
        ranges = Ranges((1, 1, 1, 1))
    except Exception as e:
        print(e)

    try:
        ranges = Ranges((2, 2, 2), (1, 1))
    except Exception as e:
        print(e)

    try:
        ranges = Ranges((3, 3, 3), (2, 2, 2))
    except Exception as e:
        print(e)
