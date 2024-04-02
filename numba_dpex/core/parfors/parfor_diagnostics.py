# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.parfors.parfor import ParforDiagnostics, _termwidth, print_wrapped


class ExtendedParforDiagnostics(ParforDiagnostics):
    def __init__(self):
        ParforDiagnostics.__init__(self)
        self.extra_info = {}

    def dump(self, level=1):
        if level == 0:
            level = 1
        super().dump(level)

        if self.extra_info:
            parfors_simple = self.get_parfors_simple(False)
            all_lines = self.get_all_lines(parfors_simple)
            print(" Auto-offloading ".center(_termwidth, "-"))
            self.print_auto_offloading(all_lines)
            if "kernel" in self.extra_info.keys():
                print_wrapped("Device - '%s'" % self.extra_info["kernel"])
            print(_termwidth * "-")

    def print_auto_offloading(self, lines):
        # Code partially borrowed from https://github.com/IntelPython/numba/blob/97fe221b3704bd17567b57ea47f4fc6604476cf9/numba/parfors/parfor.py#L982
        sword = "+--"
        fac = len(sword)
        fadj, froots = self.compute_graph_info(self.fusion_info)
        nadj, _nroots = self.compute_graph_info(self.nested_fusion_info)

        if len(fadj) > len(nadj):
            lim = len(fadj)
            tmp = nadj
        else:
            lim = len(nadj)
            tmp = fadj
        for x in range(len(tmp), lim):
            tmp.append([])

        summary = dict()

        def print_nest(fadj_, nadj_, theroot, reported, region_id):
            def print_g(fadj_, nadj_, nroot, depth):
                for k in nadj_[nroot]:
                    msg = fac * depth * " " + "%s%s %s" % (sword, k, "(serial")
                    if nadj_[k] == []:
                        fused = []
                        if fadj_[k] != [] and k not in reported:
                            fused = sorted(self.reachable_nodes(fadj_, k))
                            msg += ", fused with loop(s): "
                            msg += ", ".join([str(x) for x in fused])
                        msg += ")"
                        reported.append(k)
                        print_wrapped(msg)
                        summary[region_id]["fused"] += len(fused)
                    else:
                        print_wrapped(msg + ")")
                        print_g(fadj_, nadj_, k, depth + 1)
                    summary[region_id]["serialized"] += 1

            if nadj_[theroot] != []:
                print_wrapped("Parallel region %s:" % region_id)
                print_wrapped("%s%s %s" % (sword, theroot, "(parallel)"))
                summary[region_id] = {
                    "root": theroot,
                    "fused": 0,
                    "serialized": 0,
                }
                print_g(fadj_, nadj_, theroot, 1)
                print("\n")
                region_id = region_id + 1
            return region_id

        def print_fuse(ty, pf_id, adj, depth, region_id):
            print_wrapped("Parallel region %s:" % region_id)
            msg = fac * depth * " " + "%s%s %s" % (sword, pf_id, "(parallel")
            fused = []
            if adj[pf_id] != []:
                fused = sorted(self.reachable_nodes(adj, pf_id))
                msg += ", fused with loop(s): "
                msg += ", ".join([str(x) for x in fused])

            summary[region_id] = {
                "root": pf_id,
                "fused": len(fused),
                "serialized": 0,
            }
            msg += ")"
            print_wrapped(msg)
            extra_info = self.extra_info.get(str(region_id))
            if extra_info:
                print_wrapped("Device - '%s'" % extra_info)
            region_id = region_id + 1
            return region_id

        # Walk the parfors by src line and print optimised structure
        region_id = 0
        reported = []
        for line, info in sorted(lines.items()):
            opt_ty, pf_id, adj = info
            if opt_ty == "fuse":
                if pf_id not in reported:
                    region_id = print_fuse("f", pf_id, adj, 0, region_id)
            elif opt_ty == "nest":
                region_id = print_nest(fadj, nadj, pf_id, reported, region_id)
            else:
                assert 0

        # print the summary of the fuse/serialize rewrite
        if summary:
            for k, v in sorted(summary.items()):
                msg = (
                    "\n \nParallel region %s (loop #%s) had %s " "loop(s) fused"
                )
                root = v["root"]
                fused = v["fused"]
                serialized = v["serialized"]
                if serialized != 0:
                    msg += (
                        " and %s loop(s) "
                        "serialized as part of the larger "
                        "parallel loop (#%s)."
                    )
                    print_wrapped(msg % (k, root, fused, serialized, root))
                else:
                    msg += "."
                    print_wrapped(msg % (k, root, fused))
        else:
            print_wrapped("Parallel structure is already optimal.")
