# Copyright (c) 2020 ARM Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sets cpu core options
function(mbed_set_cpu_core_options target mbed_toolchain)
    if(${mbed_toolchain} STREQUAL "GCC_ARM")
        list(APPEND common_toolchain_options
            "-mthumb"
            "-mcpu=cortex-m4"
        )

        target_compile_options(${target}
            INTERFACE
                ${common_toolchain_options}
        )

        target_link_options(${target}
            INTERFACE
                ${common_toolchain_options}
        )
    elseif(${mbed_toolchain} STREQUAL "ARM")
        list(APPEND compile_options
            "-mcpu=cortex-m4"
            "-mfpu=none"
        )

        target_compile_options(${target}
            INTERFACE
                $<$<COMPILE_LANGUAGE:C>:${compile_options}>
                $<$<COMPILE_LANGUAGE:CXX>:${compile_options}>
                $<$<COMPILE_LANGUAGE:ASM>:-mcpu=Cortex-M4.no_fp>
        )

        target_link_options(${target}
            INTERFACE
                "--cpu=Cortex-M4.no_fp"
        )
    endif()
endfunction()

function(mbed_set_cpu_core_definitions target)
    target_compile_definitions(${target}
        INTERFACE
            __CORTEX_M4
            ARM_MATH_CM4
            __CMSIS_RTOS
            __MBED_CMSIS_RTOS_CM
    )
endfunction()
