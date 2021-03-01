CONFIG = {
    "framework": "other",
    "arg": "PYCMD -m numba.runtests -b -v -m",
    "env": None,
    "expect_except": None,
    "need_sources": False,
    "location": None,
    "use_re": False,
    "set_cl_path": False,
    "errors": {
        "2.7": {
            "Win": {
                "test_compile_helperlib (numba.tests.test_pycc.TestCC)",
                "test_compile_for_cpu_host (numba.tests.test_pycc.TestCC)",
                "test_compile (numba.tests.test_pycc.TestCC)",
                "test_compile_for_cpu (numba.tests.test_pycc.TestCC)",
                "test_compile_nrt (numba.tests.test_pycc.TestCC)",
                "test_divide_array_op (numba.tests.test_ufuncs.TestArrayOperators)",
                "test_inspect_cfg (numba.tests.test_dispatcher.TestDispatcherMethods)",
            },
            "Lin": {
                "test_all (numba.tests.test_runtests.TestCase)",
                "test_cuda (numba.tests.test_runtests.TestCase)",
                "test_default (numba.tests.test_runtests.TestCase)",
                "test_module (numba.tests.test_runtests.TestCase)",
                "test_subpackage (numba.tests.test_runtests.TestCase)",
                "test_optional_unpack (numba.tests.test_optional.TestOptional)",  # non-reproducible on Ubuntu Ubuntu 14.04.4 LTS
                "test_profiler_np_dot (numba.tests.test_profiler.TestProfiler)",
                "test_compile_helperlib (numba.tests.test_pycc.TestCC)",
                "test_unserialize_other_process_bitcode (numba.tests.test_codegen.JITCPUCodegenTestCase)",
                "test_other_process (numba.tests.test_serialize.TestDispatcherPickling)",
                "test_compile_nrt (numba.tests.test_pycc.TestCC)",
                "test_np_implicit_initialization (numba.tests.test_random.TestProcesses)",
                "test_py_implicit_initialization (numba.tests.test_random.TestProcesses)",
                "test_unserialize_other_process_object_code (numba.tests.test_codegen.JITCPUCodegenTestCase)",
            },
            "Mac": set(),
        },
        "3.5": {
            "Win": {
                "test_compile_helperlib (numba.tests.test_pycc.TestCC)",
                "test_compile_nrt (numba.tests.test_pycc.TestCC)",
                "test_inspect_cfg (numba.tests.test_dispatcher.TestDispatcherMethods)",
            },
            "Lin": {
                "test_all (numba.tests.test_runtests.TestCase)",
                "test_cuda (numba.tests.test_runtests.TestCase)",
                "test_default (numba.tests.test_runtests.TestCase)",
                "test_module (numba.tests.test_runtests.TestCase)",
                "test_subpackage (numba.tests.test_runtests.TestCase)",
            },
            "Mac": set(),
        },
        "3.6": {
            "Win": {
                "test_compile_helperlib (numba.tests.test_pycc.TestCC)",
                "test_compile_nrt (numba.tests.test_pycc.TestCC)",
                "test_inspect_cfg (numba.tests.test_dispatcher.TestDispatcherMethods)",
            },
            "Lin": {
                "test_compile_helperlib (numba.tests.test_pycc.TestCC)",
                "test_compile_nrt (numba.tests.test_pycc.TestCC)",
            },
            "Mac": set(),
        },
        "3.7": {
            "Win": {
                "test_compile_helperlib (numba.tests.test_pycc.TestCC)",  # SAT-3716
                "test_ediff1d_edge_cases (numba.tests.test_np_functions.TestNPFunctions)",  # SAT-3709
            },
            "Lin": {
                "test_isinf_m_? (numba.tests.test_ufuncs.TestLoopTypesDatetimeNoPython)",  # SAT-3088
                "test_isinf_M_? (numba.tests.test_ufuncs.TestLoopTypesDatetimeNoPython)",  # SAT-3088
                "test_isnan_M_? (numba.tests.test_ufuncs.TestLoopTypesDatetimeNoPython)",  # SAT-3088
                "test_isnan_m_? (numba.tests.test_ufuncs.TestLoopTypesDatetimeNoPython)",  # SAT-3088
                "test_ediff1d_edge_cases (numba.tests.test_np_functions.TestNPFunctions)",  # SAT-3709
            },
            "Mac": {
                "test_ediff1d_edge_cases (numba.tests.test_np_functions.TestNPFunctions)",  # SAT-3709
            },
        },
    },
    "failures": {
        "2.7": {
            "Win": {
                "test_divide_array_op (numba.tests.test_ufuncs.TestArrayOperators)",
                "test_round_array (numba.tests.test_array_methods.TestArrayMethods)",
                "test_reciprocal_F_F (numba.tests.test_ufuncs.TestLoopTypesComplexNoPython)",
                "test_reciprocal_f_f (numba.tests.test_ufuncs.TestLoopTypesReciprocalNoPython)",
                "test_reciprocal_f_f (numba.tests.test_ufuncs.TestLoopTypesFloatNoPython)",
                "test_reciprocal_F_F (numba.tests.test_ufuncs.TestLoopTypesReciprocalNoPython)",
                "test_ipython (numba.tests.test_dispatcher.TestCache)",  # SAT-1070
                "test_divmod_floats_npm (numba.tests.test_builtins.TestBuiltins)",
                "test_divmod_floats (numba.tests.test_builtins.TestBuiltins)",
                "test_setup_py_setuptools (numba.tests.test_pycc.TestDistutilsSupport)",
                "test_array_prod_global_float32_1d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-1167
                "test_array_prod_float32_1d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-1167
                "test_array_nanprod_float32_1d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-1167
                "test_array_nanprod_float32_2d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-1167
                "test_array_prod_global_float32_2d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-1167
                "test_array_prod_float32_2d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-1167
                "test_linalg_cond (numba.tests.test_linalg.TestLinalgCond)",  # new in 2.7 on all platforms with NumPy 1.15
            },
            "Lin": {
                "test_ipython (numba.tests.test_dispatcher.TestCache)",  # SAT-1070
                "test_complex_unify_issue599_multihash (numba.tests.test_typeinfer.TestUnifyUseCases)",
                "test_cache_reuse (numba.tests.test_dispatcher.TestCache)",
                "test_caching_overload_method (numba.tests.test_extending.TestOverloadMethodCaching)",
                "test_numpy_random_startup (numba.tests.test_random.TestRandom)",
                "test_caching (numba.tests.test_cfunc.TestCFuncCache)",
                "test_setup_py_setuptools (numba.tests.test_pycc.TestDistutilsSupport)",
                "test_random_random_startup (numba.tests.test_random.TestRandom)",
                "test_first_load_cached_gufunc (numba.tests.npyufunc.test_caching.TestCacheSpecificIssue)",
                "test_first_load_cached_ufunc (numba.tests.npyufunc.test_caching.TestCacheSpecificIssue)",
                "test_numpy_gauss_startup (numba.tests.test_random.TestRandom)",
                "test_setup_py_distutils (numba.tests.test_pycc.TestDistutilsSupport)",
                "test_caching (numba.tests.test_dispatcher.TestCache)",
                "test_random_gauss_startup (numba.tests.test_random.TestRandom)",
                "test_linalg_cond (numba.tests.test_linalg.TestLinalgCond)",  # new in 2.7 on all platforms with NumPy 1.15
            },
            "Mac": {
                "test_ipython (numba.tests.test_dispatcher.TestCache)",  # SAT-1070
                "test_ufunc (numba.tests.test_smart_array.TestJIT)",  # SAT-1095
                "test_index_ufunc (numba.tests.test_extending.TestPandasLike)",  # SAT-1095
                "test_array_prod_global_float32_1d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-1167
                "test_array_prod_float32_1d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-1167
                "test_array_nanprod_float32_1d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-1167
                "test_array_nanprod_float32_2d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-1167
                "test_array_prod_global_float32_2d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-1167
                "test_array_prod_float32_2d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-1167
                "test_linalg_cond (numba.tests.test_linalg.TestLinalgCond)",  # new in 2.7 on all platforms with NumPy 1.15
            },
        },
        "3.5": {
            "Win": {
                "test_ipython (numba.tests.test_dispatcher.TestCache)",  # SAT-1070
                "test_expm1_npm (numba.tests.test_mathlib.TestMathLib)",
            },
            "Lin": {"test_ipython (numba.tests.test_dispatcher.TestCache)"},  # SAT-1070
            "Mac": {
                "test_ipython (numba.tests.test_dispatcher.TestCache)",  # SAT-1070
                "test_ufunc (numba.tests.test_smart_array.TestJIT)",  # SAT-1095
                "test_index_ufunc (numba.tests.test_extending.TestPandasLike)",  # SAT-1095
            },
        },
        "3.6": {
            "Win": {
                "test_ipython (numba.tests.test_dispatcher.TestCache)",  # SAT-1070
                "test_expm1_npm (numba.tests.test_mathlib.TestMathLib)",
            },
            "Lin": {
                "test_ipython (numba.tests.test_dispatcher.TestCache)",
                "test_caching (numba.tests.test_cfunc.TestCFuncCache)",
                "test_caching (numba.tests.test_dispatcher.TestCache)",
            },
            "Mac": {
                "test_ipython (numba.tests.test_dispatcher.TestCache)",  # SAT-1070
                "test_ufunc (numba.tests.test_smart_array.TestJIT)",  # SAT-1095
                "test_index_ufunc (numba.tests.test_extending.TestPandasLike)",  # SAT-1095
            },
        },
        "3.7": {
            "Win": {
                "test_expm1_npm (numba.tests.test_mathlib.TestMathLib)",  # SAT-3715
                "test_expm1 (numba.tests.test_mathlib.TestMathLib)",  # SAT-3715
            },
            "Lin": {
                "test_maximum_MM_M (numba.tests.test_ufuncs.TestLoopTypesDatetimeNoPython)",  # SAT-3088
                "test_minimum_mm_m (numba.tests.test_ufuncs.TestLoopTypesDatetimeNoPython)",  # SAT-3088
                "test_maximum_mm_m (numba.tests.test_ufuncs.TestLoopTypesDatetimeNoPython)",  # SAT-3088
                "test_minimum_MM_M (numba.tests.test_ufuncs.TestLoopTypesDatetimeNoPython)",  # SAT-3088
                "test_argmax_npdatetime (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3088
                "test_argmin_npdatetime (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3088
                "test_max_npdatetime (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3088
                "test_min_npdatetime (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3088
                "test_argwhere_array_like (numba.tests.test_array_manipulation.TestArrayManipulation)",  # SAT-3088
                "test_argwhere_array_like (numba.tests.test_array_manipulation.TestArrayManipulation)",  # SAT-3088
                "test_array_nanprod_float32_3d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3138
                "test_array_prod_float32_3d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3138
                "test_array_prod_global_float32_3d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3138
                "test_ipython (numba.tests.test_dispatcher.TestCache)",  # SAT-1070
            },
            "Mac": {
                "test_ipython (numba.tests.test_dispatcher.TestCache)",  # SAT-1070
                "test_lift_objectmode_issue_4223 (numba.tests.test_looplifting.TestLoopLiftingInAction)",  # SAT-3712
                "test_index_ufunc (numba.tests.test_extending.TestPandasLike)",  # SAT-1095
                "test_has_no_error (numba.tests.test_sysinfo.TestSysInfo)",  # SAT-3713
                "test_array_prod_float32_1d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3714
                "test_array_prod_float32_2d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3714
                "test_array_prod_global_float32_2d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3714
                "test_array_nanprod_float32_2d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3714
                "test_array_prod_global_float32_1d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3714
                "test_array_nanprod_float32_1d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3714
            },
        },
    },
    "validation": {
        "errors": {
            "2.7": {
                "Win": {
                    "test_sin_function_ool (numba.tests.test_cffi.TestCFFI)",  # SAT-1663
                    "test_from_buffer_float64 (numba.tests.test_cffi.TestCFFI)",  # SAT-1663
                    "test_two_funcs_ool (numba.tests.test_cffi.TestCFFI)",  # SAT-1663
                    "test_type_map (numba.tests.test_cffi.TestCFFI)",  # SAT-1663
                    "test_from_buffer_error (numba.tests.test_cffi.TestCFFI)",  # SAT-1663
                    "test_indirect_multiple_use (numba.tests.test_cffi.TestCFFI)",  # SAT-1663
                    "test_sin_function (numba.tests.test_cffi.TestCFFI)",  # SAT-1663
                    "test_from_buffer_float32 (numba.tests.test_cffi.TestCFFI)",  # SAT-1663
                    "test_user_defined_symbols (numba.tests.test_cffi.TestCFFI)",  # SAT-1663
                    "test_bool_function_ool (numba.tests.test_cffi.TestCFFI)",  # SAT-1663
                    "test_function_pointer (numba.tests.test_cffi.TestCFFI)",  # SAT-1663
                    "test_sin_function_npm (numba.tests.test_cffi.TestCFFI)",  # SAT-1663
                    "test_sin_function_npm_ool (numba.tests.test_cffi.TestCFFI)",  # SAT-1663
                    "test_two_funcs (numba.tests.test_cffi.TestCFFI)",  # SAT-1663
                    "test_from_buffer_struct (numba.tests.test_cffi.TestCFFI)",  # SAT-1663
                    "test_from_buffer_numpy_multi_array (numba.tests.test_cffi.TestCFFI)",  # SAT-1663 & SAT-1778
                },
                "Lin": set(),
                "Mac": {
                    "test_subpackage (numba.tests.test_runtests.TestCase)",  # SAT-1175
                    "test_cuda (numba.tests.test_runtests.TestCase)",  # SAT-1175
                    "test_all (numba.tests.test_runtests.TestCase)",  # SAT-1175
                    "test_default (numba.tests.test_runtests.TestCase)",  # SAT-1175
                    "test_module (numba.tests.test_runtests.TestCase)",  # SAT-1175
                },
            },
            "3.7": {
                "Win": {
                    "test_compile_helperlib (numba.tests.test_pycc.TestCC)",  # SAT-3716
                    "test_ediff1d_edge_cases (numba.tests.test_np_functions.TestNPFunctions)",  # SAT-3709
                },
                "Lin": {
                    "test_isinf_m_? (numba.tests.test_ufuncs.TestLoopTypesDatetimeNoPython)",  # SAT-3088
                    "test_isinf_M_? (numba.tests.test_ufuncs.TestLoopTypesDatetimeNoPython)",  # SAT-3088
                    "test_isnan_M_? (numba.tests.test_ufuncs.TestLoopTypesDatetimeNoPython)",  # SAT-3088
                    "test_isnan_m_? (numba.tests.test_ufuncs.TestLoopTypesDatetimeNoPython)",  # SAT-3088
                    "test_ediff1d_edge_cases (numba.tests.test_np_functions.TestNPFunctions)",  # SAT-3709
                },
                "Mac": {
                    "test_ediff1d_edge_cases (numba.tests.test_np_functions.TestNPFunctions)",  # SAT-3709
                },
            },
        },
        "failures": {
            "2.7": {
                "Win": set(),
                "Lin": set(),
                "Mac": set(),
            },
            "3.7": {
                "Win": {
                    "test_expm1_npm (numba.tests.test_mathlib.TestMathLib)",  # SAT-3715
                    "test_expm1 (numba.tests.test_mathlib.TestMathLib)",  # SAT-3715
                },
                "Lin": {
                    "test_maximum_MM_M (numba.tests.test_ufuncs.TestLoopTypesDatetimeNoPython)",  # SAT-3088
                    "test_minimum_mm_m (numba.tests.test_ufuncs.TestLoopTypesDatetimeNoPython)",  # SAT-3088
                    "test_maximum_mm_m (numba.tests.test_ufuncs.TestLoopTypesDatetimeNoPython)",  # SAT-3088
                    "test_minimum_MM_M (numba.tests.test_ufuncs.TestLoopTypesDatetimeNoPython)",  # SAT-3088
                    "test_argmax_npdatetime (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3088
                    "test_argmin_npdatetime (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3088
                    "test_max_npdatetime (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3088
                    "test_min_npdatetime (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3088
                    "test_argwhere_array_like (numba.tests.test_array_manipulation.TestArrayManipulation)",  # SAT-3088
                    "test_argwhere_array_like (numba.tests.test_array_manipulation.TestArrayManipulation)",  # SAT-3088
                    "test_array_nanprod_float32_3d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3138
                    "test_array_prod_float32_3d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3138
                    "test_array_prod_global_float32_3d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3138
                    "test_ipython (numba.tests.test_dispatcher.TestCache)",  # SAT-1070
                },
                "Mac": {
                    "test_ipython (numba.tests.test_dispatcher.TestCache)",  # SAT-1070
                    "test_lift_objectmode_issue_4223 (numba.tests.test_looplifting.TestLoopLiftingInAction)",  # SAT-3712
                    "test_index_ufunc (numba.tests.test_extending.TestPandasLike)",  # SAT-1095
                    "test_has_no_error (numba.tests.test_sysinfo.TestSysInfo)",  # SAT-3713
                    "test_array_prod_float32_1d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3714
                    "test_array_prod_float32_2d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3714
                    "test_array_prod_global_float32_2d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3714
                    "test_array_nanprod_float32_2d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3714
                    "test_array_prod_global_float32_1d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3714
                    "test_array_nanprod_float32_1d (numba.tests.test_array_reductions.TestArrayReductions)",  # SAT-3714
                },
            },
        },
    },
}


def fetch_config(pyver):
    if pyver != "3.7":
        CONFIG["arg"] += " --tags important"
    return CONFIG
