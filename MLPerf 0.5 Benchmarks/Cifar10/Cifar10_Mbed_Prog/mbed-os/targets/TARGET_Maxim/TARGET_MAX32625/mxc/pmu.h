/*******************************************************************************
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
 * $Date: 2016-06-21 16:14:41 -0500 (Tue, 21 Jun 2016) $
 * $Revision: 23446 $
 *
 ******************************************************************************/

/**
 * @file  pmu.h
 * @addtogroup pmu Peripheral Management Unit
 * @{
 * @brief This is the API for the peripheral management unit.
 */

#ifndef _PMU_H_
#define _PMU_H_

#include "pmu_regs.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Defines Clock scale used for Timeout
typedef enum {
    PMU_PS_SEL_DISABLE  = MXC_V_PMU_CFG_PS_SEL_DISABLE,  // Timeout disabled
    PMU_PS_SEL_DIV_2_8  = MXC_V_PMU_CFG_PS_SEL_DIV_2_8,  // Timeout clk = PMU clock / 2^8 = 256
    PMU_PS_SEL_DIV_2_16 = MXC_V_PMU_CFG_PS_SEL_DIV_2_16, // Timeout clk = PMU clock / 2^16 = 65536
    PMU_PS_SEL_DIV_2_24 = MXC_V_PMU_CFG_PS_SEL_DIV_2_24  // Timeout clk =  PMU clock / 2^24 = 16777216
}pmu_ps_sel_t;

/// @brief Defines the number of clk ticks for timeout duration
typedef enum {
    PMU_TO_SEL_TICKS_4   = MXC_V_PMU_CFG_TO_SEL_TICKS_4,      //timeout =  4 * Timeout clk period
    PMU_TO_SEL_TICKS_8   = MXC_V_PMU_CFG_TO_SEL_TICKS_8,      //timeout =  8 * Timeout clk period
    PMU_TO_SEL_TICKS_16  = MXC_V_PMU_CFG_TO_SEL_TICKS_16,     //timeout =  16 * Timeout clk period
    PMU_TO_SEL_TICKS_32  = MXC_V_PMU_CFG_TO_SEL_TICKS_32,     //timeout =  32 * Timeout clk period
    PMU_TO_SEL_TICKS_64  = MXC_V_PMU_CFG_TO_SEL_TICKS_64,     //timeout =  64 * Timeout clk period
    PMU_TO_SEL_TICKS_128 = MXC_V_PMU_CFG_TO_SEL_TICKS_128,    //timeout =  128 * Timeout clk period
    PMU_TO_SEL_TICKS_256 = MXC_V_PMU_CFG_TO_SEL_TICKS_256,    //timeout =  256 * Timeout clk period
    PMU_TO_SEL_TICKS_512 = MXC_V_PMU_CFG_TO_SEL_TICKS_512     //timeout =  512 * Timeout clk period
}pmu_to_sel_t;


/* The macros like the one below are designed to help build static PMU programs
 *  as arrays of 32bit words.
 */
#define PMU_IS(interrupt, stop) ((!!interrupt) << PMU_INT_POS) | ((!!stop) << PMU_STOP_POS)

typedef struct pmu_move_des_t {
    uint32_t op_code      : 3; /* 0x0 */
    uint32_t interrupt    : 1;
    uint32_t stop         : 1;
    uint32_t read_size    : 2;
    uint32_t read_inc     : 1;
    uint32_t write_size   : 2;
    uint32_t write_inc    : 1;
    uint32_t cont         : 1;
    uint32_t length       : 20;

    uint32_t write_address;
    uint32_t read_address;
} pmu_move_des_t;
#define PMU_MOVE(i, s, rs, ri, ws, wi, c, length, wa, ra) \
        (PMU_MOVE_OP | PMU_IS(i,s) | ((rs & 3) << PMU_MOVE_READS_POS) | ((!!ri) << PMU_MOVE_READI_POS) | \
        ((ws & 3) << PMU_MOVE_WRITES_POS) | ((!!wi) << PMU_MOVE_WRITEI_POS) | ((!!c) << PMU_MOVE_CONT_POS) | ((length & 0xFFFFF) << PMU_MOVE_LEN_POS)), wa, ra

/* new_value = value | (old_value & ~ mask)  */
typedef struct pmu_write_des_t {
    uint32_t op_code      : 3; /* 0x1 */
    uint32_t interrupt    : 1;
    uint32_t stop         : 1;
    uint32_t              : 3;
    uint32_t write_method : 4;
    uint32_t              : 20;

    uint32_t write_address;
    uint32_t value;
    uint32_t mask;
} pmu_write_des_t;
#define PMU_WRITE(i, s, wm, a, v, m) (PMU_WRITE_OP | PMU_IS(i,s) | ((wm & 0xF) << PMU_WRITE_METHOD_POS)), a, v, m

typedef struct pmu_wait_des_t {
    uint32_t op_code      : 3; /* 0x2 */
    uint32_t interrupt    : 1;
    uint32_t stop         : 1;
    uint32_t wait         : 1;
    uint32_t sel          : 1;
    uint32_t              : 25;

    uint32_t mask1;
    uint32_t mask2;
    uint32_t wait_count;
} pmu_wait_des_t;
#define PMU_WAIT(i, s, sel, m1, m2, cnt) (PMU_WAIT_OP | PMU_IS(i,s) | ((cnt>0)?(1<<PMU_WAIT_WAIT_POS):0) | ((!!sel) << PMU_WAIT_SEL_POS)), \
        m1, m2, cnt

typedef struct pmu_jump_des_t {
    uint32_t op_code      : 3; /* 0x3 */
    uint32_t interrupt    : 1;
    uint32_t stop         : 1;
    uint32_t              : 27;

    uint32_t address;
} pmu_jump_des_t;
#define PMU_JUMP(i, s, a) (PMU_JUMP_OP | PMU_IS(i,s)), a

typedef struct pmu_loop_des_t {
    uint32_t op_code      : 3; /* 0x4 */
    uint32_t interrupt    : 1;
    uint32_t stop         : 1;
    uint32_t sel_counter  : 1;
    uint32_t              : 26;

    uint32_t address;
} pmu_loop_des_t;
#define PMU_LOOP(i, s, c, a) (PMU_LOOP_OP | PMU_IS(i,s) | ((!!c) << PMU_LOOP_SEL_COUNTER_POS)), a

typedef struct pmu_poll_des_t {
    uint32_t op_code      : 3; /* 0x5 */
    uint32_t interrupt    : 1;
    uint32_t stop         : 1;
    uint32_t              : 2;
    uint32_t and          : 1;
    uint32_t              : 24;

    uint32_t poll_addr;
    uint32_t data;
    uint32_t mask;
    uint32_t poll_interval;
} pmu_poll_des_t;
#define PMU_POLL(i, s, a, adr, d, m, per) (PMU_POLL_OP | PMU_IS(i,s) | ((!!a) << PMU_POLL_AND_POS)), adr, d, m, per

typedef struct pmu_branch_des_t {
    uint32_t op_code      : 3; /* 0x6 */
    uint32_t interrupt    : 1;
    uint32_t stop         : 1;
    uint32_t              : 2;
    uint32_t and          : 1;
    uint32_t type         : 3;
    uint32_t              : 21;

    uint32_t poll_addr;
    uint32_t data;
    uint32_t mask;
    uint32_t address;
} pmu_branch_des_t;
#define PMU_BRANCH(i, s, a, t, adr, d, m, badr) \
       (PMU_BRANCH_OP | PMU_IS(i,s) | ((!!a) << PMU_BRANCH_AND_POS)| ((t & 7) << PMU_BRANCH_TYPE_POS)), adr, d, m, badr

typedef struct pmu_transfer_des_t {
    uint32_t op_code      : 3; /* 0x7 */
    uint32_t interrupt    : 1;
    uint32_t stop         : 1;
    uint32_t read_size    : 2;
    uint32_t read_inc     : 1;
    uint32_t write_size   : 2;
    uint32_t write_inc    : 1;
    uint32_t              : 1;
    uint32_t tx_length    : 20;

    uint32_t write_address;
    uint32_t read_address;

    uint32_t int_mask     : 25; /* valid int_mask is from 0 - 24 */
    uint32_t              : 1;
    uint32_t burst_size   : 6;
} pmu_transfer_des_t;
#define PMU_TRANSFER(i, s, rs, ri, ws, wi, l, wa, ra, imsk, b) \
        (PMU_TRANSFER_OP | PMU_IS(i,s) | ((rs & 3) << PMU_TX_READS_POS) | ((!!ri) << PMU_TX_READI_POS) | \
        ((ws & 3) << PMU_TX_WRITES_POS) | ((!!wi) << PMU_TX_WRITEI_POS) | ((l & 0xFFFFF) << PMU_TX_LEN_POS)), wa, ra, \
        ((imsk) | ((b & 0x3F) << PMU_TX_BS_POS))

/**
 * @brief Type alias \c pmu_callback with function signature: \code void (*pmu_callback)(int pmu_status) \endcode
 * @details The callback function will be called for every opcode that has
 *          the interrupt bit set. If NULL, the channel interrupt will not
 *          be enabled. The callback function argument is a status bit
 *          indicating the status of the PMU program.
 * @param pmu_status The callback function argument is a status bit indicating
 *                   the status of the PMU program.
 */
typedef void (*pmu_callback)(int pmu_status);

/**
 * @brief Start a PMU program on a channel
 * @param channel            Channel to start
 * @param program_address    Pointer to the first opcode of the PMU program
 * @param callback  Callback function of the signature \c pmu_callback
 * @param arg       Pointer to be passed to the interrupt callback function.
 * @returns E_NO_ERROR if everything is successful, error if unsuccessful.
 */
int PMU_Start(unsigned int channel, const void *program_address, pmu_callback callback);

/**
 * @brief Set a loop counter value on a channel
 * @param channel      Channel number to set the value on
 * @param counter_num  Counter number for the channel (0 or 1)
 * @param value        Loop count value
 * @returns E_NO_ERROR if everything is successful, error if unsuccessful.
 */
int PMU_SetCounter(unsigned int channel, unsigned int counter_num, uint16_t value);

/**
 * @brief Stop a running channel. This will clear the enable bit on the channel
 *        and stop the running PMU program at the current opcode. The callback
 *        function is not called.
 * @param channel   Channel to stop
 */
void PMU_Stop(unsigned int channel);

/**
 * @brief Function to handle PMU interrupts. This function can be called from
 *        the PMU interrupt service routine, or periodically from the
 *        application if interrupts are not enabled.
 */
void PMU_Handler(void);

/**
 * @brief   Set the AHB bus operation timeout on a channel
 * @param   channel             Selected PMU channel
 * @param   timeoutClkScale     Clk scale use for timeout clk
 * @param   timeoutTicks        Number of ticks for timeout duration
 * @returns E_NO_ERROR if everything is successful, error if unsuccessful.
 */
int PMU_SetTimeout(unsigned int channel, pmu_ps_sel_t timeoutClkScale, pmu_to_sel_t timeoutTicks);

/**
 * @brief   Gets the PMU channel's flags
 * @param   channel      Selected PMU channel
 * @return  0 = flags not set, non-zero = flags
 */
uint32_t PMU_GetFlags(unsigned int channel);

/**
 * @brief   Clear the PMU channel's flags based on the mask
 * @param   channel      Selected PMU channel
 * @param   mask         bits of the flags to clear
 */
void PMU_ClearFlags(unsigned int channel, unsigned int mask);

/**
 * @brief   Determines if the PMU channel is running
 * @param   channel      Selected PMU channel
 * @return  0 = channel is off, non-zero = channel is running
 */
uint32_t PMU_IsActive(unsigned int channel);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif /* _PMU_H_ */
