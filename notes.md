1. Specialized function gets compiled when the file is loaded
2. Dig into:
    cres.typing_context.insert_user_function(devfn, _function_template)
    libs = [cres.library]
    cres.target_context.insert_user_function(devfn, cres.fndesc, libs)
3. why the numba code is downcasting? Look into the lowering logic
    %".87" = fptosi double %".86" to i32
    store i32 0, i32* %".88"
    %".91" = call spir_func i32 @"dpex_py_devfn__5F__5F_main_5F__5F__2E_a_5F_device_5F_function_2E_int32"(i32* %".88", i32 %".87")
4. This is what should happen: if a func is specialized, dpex should throw an error rather than doing an automatic typecasting.
5. look into numba's lowering and intercept the typecasting -- the goal is to stop lowering when type doesn't match in specialization case
6. Most likely 867 will be resolved if these problems are addressed.
