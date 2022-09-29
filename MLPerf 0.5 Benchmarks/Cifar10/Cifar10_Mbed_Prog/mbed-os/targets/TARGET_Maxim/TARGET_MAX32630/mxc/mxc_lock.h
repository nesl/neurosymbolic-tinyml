/**
 * @file     
 * @brief    Exclusive access lock utility functions.
*/
/* ****************************************************************************
 * Copyright (C) 2016 Maxim Integrated Products, Inc., All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
 * OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * Except as contained in this notice, the name of Maxim Integrated
 * Products, Inc. shall not be used except as stated in the Maxim Integrated
 * Products, Inc. Branding Policy.
 *
 * The mere transfer of this software does not imply any licenses
 * of trade secrets, proprietary technology, copyrights, patents,
 * trademarks, maskwork rights, or any other form of intellectual
 * property whatsoever. Maxim Integrated Products, Inc. retains all
 * ownership rights.
 *
 * $Date: 2016-10-10 19:16:20 -0500 (Mon, 10 Oct 2016) $
 * $Revision: 24663 $
 *
 *************************************************************************** */

/* Define to prevent redundant inclusion */
#ifndef _MXC_LOCK_H_
#define _MXC_LOCK_H_

/***** Includes *****/
#include "mxc_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup    sysconfig
 * @defgroup   mxc_lock_utilities Lock functions for Exclusive Access
 * @brief      Lock functions to obtain and release a variable for exclusive
 *             access. These functions are marked interrupt safe if they are
 *             interrupt safe. 
 * @{
 */ 

/* **** Definitions **** */

/* **** Globals **** */

/* **** Function Prototypes **** */

/**
 * @brief      Attempts to acquire the lock.
 * @details    This in an interrupt safe function that can be used as a mutex.
 *             The lock variable must remain in scope until the lock is
 *             released. Will not block if another thread has already acquired
 *             the lock.
 * @param      lock   Pointer to variable that is used for the lock.
 * @param      value  Value to be place in the lock. Can not be 0.
 *
 * @return     #E_NO_ERROR if everything successful, #E_BUSY if lock is taken.
 */
__STATIC_INLINE int mxc_get_lock(uint32_t *lock, uint32_t value)
{
    do {

        // Return if the lock is taken by a different thread
        if(__LDREXW((volatile uint32_t *)lock) != 0) {
            return E_BUSY;
        }

        // Attempt to take the lock
    } while(__STREXW(value, (volatile uint32_t *)lock) != 0);

    // Do not start any other memory access until memory barrier is complete
    __DMB();

    return E_NO_ERROR;
}

/**
 * @brief         Free the given lock.
 * @param[in,out] lock  Pointer to the variable used for the lock. When the lock
 *                      is free, the value pointed to by @p lock is set to zero.
 */
__STATIC_INLINE void mxc_free_lock(uint32_t *lock)
{
    // Ensure memory operations complete before releasing lock
    __DMB();
    *lock = 0;
}

/**@} end of group mxc_lock_utilities */

#ifdef __cplusplus
}
#endif

#endif /* _MXC_LOCK_H_ */
