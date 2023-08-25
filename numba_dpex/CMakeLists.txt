if(IS_INSTALL)
    install(DIRECTORY core
        DESTINATION numba_dpex
        FILES_MATCHING PATTERN "*.py")
endif()

add_subdirectory(core/runtime)

if(IS_INSTALL)
    install(DIRECTORY dpctl_iface
        DESTINATION numba_dpex
        FILES_MATCHING PATTERN "*.py")
endif()

if(IS_INSTALL)
    install(DIRECTORY dpnp_iface
        DESTINATION numba_dpex
        FILES_MATCHING PATTERN "*.py")
endif()

if(IS_INSTALL)
    install(DIRECTORY numba_patches
        DESTINATION numba_dpex
        FILES_MATCHING PATTERN "*.py")
endif()

if(IS_INSTALL)
    install(DIRECTORY ocl
        DESTINATION numba_dpex
        FILES_MATCHING PATTERN "*.py")
endif()

if(IS_INSTALL)
    install(DIRECTORY tests
        DESTINATION numba_dpex
        FILES_MATCHING PATTERN "*.py")
endif()

if(IS_INSTALL)
    install(DIRECTORY utils
        DESTINATION numba_dpex
        FILES_MATCHING PATTERN "*.py")
endif()
