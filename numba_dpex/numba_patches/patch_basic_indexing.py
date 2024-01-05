def patch():
    from numba.core import cgutils, types
    from numba.cpython import slicing
    from numba.np import arrayobj
    from numba.np.arrayobj import fix_integer_index
    from numba.np.numpy_support import is_nonelike

    from numba_dpex.core.types import DpnpNdArray

    # -------------------------------------------------------------------------
    # Basic indexing (with integers and slices only)

    def basic_indexing(
        context, builder, aryty, ary, index_types, indices, boundscheck=None
    ):
        """
        Perform basic indexing on the given array.
        A (data pointer, shapes, strides) tuple is returned describing
        the corresponding view.
        """
        print("============> doing this basic_indexing")

        zero = context.get_constant(types.intp, 0)
        one = context.get_constant(types.intp, 1)

        shapes = cgutils.unpack_tuple(builder, ary.shape, aryty.ndim)
        strides = cgutils.unpack_tuple(builder, ary.strides, aryty.ndim)

        if isinstance(aryty, DpnpNdArray):
            print(f"multiply each stride value with itemsize = {ary.itemsize}")

        output_indices = []
        output_shapes = []
        output_strides = []

        num_newaxes = len([idx for idx in index_types if is_nonelike(idx)])
        ax = 0
        for indexval, idxty in zip(indices, index_types):
            if idxty is types.ellipsis:
                # Fill up missing dimensions at the middle
                n_missing = aryty.ndim - len(indices) + 1 + num_newaxes
                for i in range(n_missing):
                    output_indices.append(zero)
                    output_shapes.append(shapes[ax])
                    output_strides.append(strides[ax])
                    ax += 1
                continue
            # Regular index value
            if isinstance(idxty, types.SliceType):
                slice = context.make_helper(builder, idxty, value=indexval)
                slicing.guard_invalid_slice(context, builder, idxty, slice)
                slicing.fix_slice(builder, slice, shapes[ax])
                output_indices.append(slice.start)
                sh = slicing.get_slice_length(builder, slice)
                st = slicing.fix_stride(builder, slice, strides[ax])
                output_shapes.append(sh)
                output_strides.append(st)
            elif isinstance(idxty, types.Integer):
                ind = fix_integer_index(
                    context, builder, idxty, indexval, shapes[ax]
                )
                if boundscheck:
                    cgutils.do_boundscheck(
                        context, builder, ind, shapes[ax], ax
                    )
                output_indices.append(ind)
            elif is_nonelike(idxty):
                output_shapes.append(one)
                output_strides.append(zero)
                ax -= 1
            else:
                raise NotImplementedError(
                    "unexpected index type: %s" % (idxty,)
                )
            ax += 1

        # Fill up missing dimensions at the end
        assert ax <= aryty.ndim
        while ax < aryty.ndim:
            output_shapes.append(shapes[ax])
            output_strides.append(strides[ax])
            ax += 1

        # No need to check wraparound, as negative indices were already
        # fixed in the loop above.
        dataptr = cgutils.get_item_pointer(
            context,
            builder,
            aryty,
            ary,
            output_indices,
            wraparound=False,
            boundscheck=False,
        )
        return (dataptr, output_shapes, output_strides)

    arrayobj.basic_indexing = basic_indexing
