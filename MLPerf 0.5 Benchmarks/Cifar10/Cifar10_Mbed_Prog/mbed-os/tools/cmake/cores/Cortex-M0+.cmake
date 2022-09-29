# Copyright (c) 2020 ARM Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sets cpu core options
function(mbed_set_cpu_core_options target mbed_toolchain)
    if(${mbed_toolchain} STREQUAL "GCC_ARM")
        list(APPEND common_toolchain_options
            "-mthumb"
            "-mcpu=cortex-m0plus"
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
            "-mcpu=cortex-m0plus"
        )

        target_compile_options(${target}
            INTERFACE
                $<$<COMPILE_LANGUAGE:C>:${compile_options}>
                $<$<COMPILE_LANGUAGE:CXX>:${compile_options}>
                $<$<COMPILE_LANGUAGE:ASM>:-mcpu=Cortex-M0plus>
        )

        target_link_options(${target}
            INTERFACE
                "--cpu=Cortex-M0plus"
        )
    endif()
endfunction()

function(mbed_set_cpu_core_definitions target)
    target_compile_definitions(${target}
        INTERFACE
            __CORTEX_M0PLUS
            ARM_MATH_CM0PLUS
            __CMSIS_RTOS
            __MBED_CMSIS_RTOS_CM
    )
endfunction()
