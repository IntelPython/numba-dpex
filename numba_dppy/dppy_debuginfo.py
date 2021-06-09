from numba.core.debuginfo import DIBuilder


class DPPYDIBuilder(DIBuilder):
    def mark_subprogram(self, function, name, loc):
        if isinstance(name, tuple):
            name, linkage_name = name
        else:
            linkage_name = function.name

        di_subp = self._add_subprogram(
            name=name, linkagename=linkage_name, line=loc.line
        )
        function.set_metadata("dbg", di_subp)
        # disable inlining for this function for easier debugging
        function.attributes.add("noinline")
