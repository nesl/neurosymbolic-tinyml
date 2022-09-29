/**
 * @file
 * @brief      This file contains the function implementations for the I2CM 
 *             (Inter-Integrated Circuit Master) peripheral module.
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
 * $Date: 2016-09-09 11:40:02 -0500 (Fri, 09 Sep 2016) $
 * $Revision: 24336 $
 *
 *************************************************************************** */

/* **** Includes **** */
#include <string.h>
#include "mxc_assert.h"
#include "mxc_lock.h"
#include "mxc_errors.h"
#include "mxc_sys.h"
#include "i2cm.h"


/**
 * @ingroup i2cm
 * @{
 */
    
///@cond
// No Doxygen documentation for the items between here and endcond. 
/* **** Definitions **** */
#ifndef MXC_I2CM_TX_TIMEOUT
#define MXC_I2CM_TX_TIMEOUT     0x5000      /**< Master Transmit Timeout in number of repetitive attempts to receive an ACK/NACK or for a transmission to occur */
#endif

#ifndef MXC_I2CM_RX_TIMEOUT
#define MXC_I2CM_RX_TIMEOUT     0x5000      /**< Master Receive Timeout in number of attempts to check FIFO for received data from a slave */
#endif

#define I2CM_READ_BIT           0x0001      /**< Bit location to specify a read for the I2C protocol */
///@cond
#define I2CM_FIFO_DEPTH_3Q      ((3 * MXC_I2CM_FIFO_DEPTH) / 4)
#define I2CM_FIFO_DEPTH_2Q      (MXC_I2CM_FIFO_DEPTH / 2)

//
/* **** Globals **** */

/* Clock divider lookup table */
static const uint32_t clk_div_table[3][8] = {
    /* I2CM_SPEED_100KHZ */
    {
        // 12000000
        ((6 << MXC_F_I2CM_FS_CLK_DIV_FS_FILTER_CLK_DIV_POS) |
            (17 << MXC_F_I2CM_FS_CLK_DIV_FS_SCL_HI_CNT_POS) |
            (72 << MXC_F_I2CM_FS_CLK_DIV_FS_SCL_LO_CNT_POS)),
        // 24000000
        ((12 << MXC_F_I2CM_FS_CLK_DIV_FS_FILTER_CLK_DIV_POS) |
            (38 << MXC_F_I2CM_FS_CLK_DIV_FS_SCL_HI_CNT_POS) |
            (144 << MXC_F_I2CM_FS_CLK_DIV_FS_SCL_LO_CNT_POS)),
        // 36000000 NOT SUPPORTED
        0,
        // 48000000
        ((24 << MXC_F_I2CM_FS_CLK_DIV_FS_FILTER_CLK_DIV_POS) |
            (80 << MXC_F_I2CM_FS_CLK_DIV_FS_SCL_HI_CNT_POS) |
            (288 << MXC_F_I2CM_FS_CLK_DIV_FS_SCL_LO_CNT_POS)),
        // 60000000 NOT SUPPORTED
        0,
        // 72000000 NOT SUPPORTED
        0,
        // 84000000 NOT SUPPORTED
        0,
        // 96000000
        ((48 << MXC_F_I2CM_FS_CLK_DIV_FS_FILTER_CLK_DIV_POS) |
            (164 << MXC_F_I2CM_FS_CLK_DIV_FS_SCL_HI_CNT_POS) |
            (576 << MXC_F_I2CM_FS_CLK_DIV_FS_SCL_LO_CNT_POS)),
    },
    /* I2CM_SPEED_400KHZ */
    {
        // 12000000
        ((2 << MXC_F_I2CM_FS_CLK_DIV_FS_FILTER_CLK_DIV_POS) |
            (1 << MXC_F_I2CM_FS_CLK_DIV_FS_SCL_HI_CNT_POS) |
            (18 << MXC_F_I2CM_FS_CLK_DIV_FS_SCL_LO_CNT_POS)),
        // 24000000
        ((3 << MXC_F_I2CM_FS_CLK_DIV_FS_FILTER_CLK_DIV_POS) |
            (5 << MXC_F_I2CM_FS_CLK_DIV_FS_SCL_HI_CNT_POS) |
            (36 << MXC_F_I2CM_FS_CLK_DIV_FS_SCL_LO_CNT_POS)),
        // 36000000 NOT SUPPORTED
        0,
        // 48000000
        ((6 << MXC_F_I2CM_FS_CLK_DIV_FS_FILTER_CLK_DIV_POS) |
            (15 << MXC_F_I2CM_FS_CLK_DIV_FS_SCL_HI_CNT_POS) |
            (72 << MXC_F_I2CM_FS_CLK_DIV_FS_SCL_LO_CNT_POS)),
        // 60000000 NOT SUPPORTED
        0,
        // 72000000 NOT SUPPORTED
        0,
        // 84000000 NOT SUPPORTED
        0,
        // 96000000
        ((12 << MXC_F_I2CM_FS_CLK_DIV_FS_FILTER_CLK_DIV_POS) |
            (33 << MXC_F_I2CM_FS_CLK_DIV_FS_SCL_HI_CNT_POS) |
            (144 << MXC_F_I2CM_FS_CLK_DIV_FS_SCL_LO_CNT_POS)),
    },
};

// Saves the state of the non-blocking requests
typedef enum {
    I2CM_STATE_READING = 0,
    I2CM_STATE_WRITING = 1
} i2cm_state_t;

typedef struct {
    i2cm_req_t *req;
    i2cm_state_t state;
} i2cm_req_state_t;
static i2cm_req_state_t states[MXC_CFG_I2CM_INSTANCES];

/* **** Local Function Prototypes **** */

static void I2CM_FreeCallback(int i2cm_num, int error);

static int I2CM_Rx(mxc_i2cm_regs_t *i2cm, mxc_i2cm_fifo_regs_t *fifo, uint8_t addr,
    uint8_t *data, uint32_t len);

static int I2CM_CmdHandler(mxc_i2cm_regs_t *i2cm, mxc_i2cm_fifo_regs_t *fifo, i2cm_req_t *req);
static int I2CM_ReadHandler(mxc_i2cm_regs_t *i2cm, i2cm_req_t *req, int i2cm_num);
static int I2CM_WriteHandler(mxc_i2cm_regs_t *i2cm, i2cm_req_t *req, int i2cm_num);
///@endcond 
//
/* ************************************************************************* */
int I2CM_Init(mxc_i2cm_regs_t *i2cm, const sys_cfg_i2cm_t *sys_cfg, i2cm_speed_t speed)
{
    int err, clki;

    // Check the base pointer
    MXC_ASSERT(MXC_I2CM_GET_IDX(i2cm) >= 0);

    // Set system level configurations
    if ((err = SYS_I2CM_Init(i2cm, sys_cfg)) != E_NO_ERROR) {
        return err;
    }

    // Compute clock array index
    clki = ((SYS_I2CM_GetFreq(i2cm) / 12000000) - 1);

    // Get clock divider settings from lookup table
    if ((speed == I2CM_SPEED_100KHZ) && (clk_div_table[I2CM_SPEED_100KHZ][clki] > 0)) {
        i2cm->fs_clk_div = clk_div_table[I2CM_SPEED_100KHZ][clki];

    } else if ((speed == I2CM_SPEED_400KHZ) && (clk_div_table[I2CM_SPEED_400KHZ][clki] > 0)) {
        i2cm->fs_clk_div = clk_div_table[I2CM_SPEED_400KHZ][clki];

    } else {
        // Requested speed is not achievable with the current clock setup
        return E_NOT_SUPPORTED;
    }

    // Reset module
    i2cm->ctrl = MXC_F_I2CM_CTRL_MSTR_RESET_EN;
    i2cm->ctrl = 0;

    // Set timeout to 255 ms and turn on the auto-stop option
    i2cm->timeout = (MXC_F_I2CM_TIMEOUT_TX_TIMEOUT | MXC_F_I2CM_TIMEOUT_AUTO_STOP_EN);

    // Enable tx_fifo and rx_fifo
    i2cm->ctrl |= (MXC_F_I2CM_CTRL_TX_FIFO_EN | MXC_F_I2CM_CTRL_RX_FIFO_EN);

    return E_NO_ERROR;
}

/* ************************************************************************* */
int I2CM_Shutdown(mxc_i2cm_regs_t *i2cm)
{
    int i2cm_num, err;

    // Check the base pointer
    i2cm_num = MXC_I2CM_GET_IDX(i2cm);
    MXC_ASSERT(i2cm_num >= 0);

    // Disable and clear interrupts
    i2cm->inten = 0;
    i2cm->intfl = i2cm->intfl;

    // Call all of the pending callbacks for this I2CM
    if(states[i2cm_num].req != NULL) {
        I2CM_Recover(i2cm);
        I2CM_FreeCallback(i2cm_num, E_SHUTDOWN);
    }

    // Clears system level configurations
    if ((err = SYS_I2CM_Shutdown(i2cm)) != E_NO_ERROR) {
        return err;
    }

    return E_NO_ERROR;
}


/* ************************************************************************* */
int I2CM_Read(mxc_i2cm_regs_t *i2cm, uint8_t addr, const uint8_t *cmd_data,
    uint32_t cmd_len, uint8_t* data, uint32_t len)
{
    int i2cm_num;
    int error = E_NO_ERROR;
    int retval = E_NO_ERROR;
    mxc_i2cm_fifo_regs_t *fifo;

    if(data == NULL) {
        return E_NULL_PTR;
    }

    // Make sure the I2CM has been initialized
    if(i2cm->ctrl == 0) {
        return E_UNINITIALIZED;
    }

    if(!(len > 0)) {
        return E_NO_ERROR;
    }

    // Lock this I2CM
    i2cm_num = MXC_I2CM_GET_IDX(i2cm);
    while(mxc_get_lock((uint32_t*)&states[i2cm_num].req,1) != E_NO_ERROR) {}

    // Get the FIFO pointer for this I2CM
    fifo = MXC_I2CM_GET_FIFO(i2cm_num);

    // Disable and clear the interrupts
	i2cm->inten = 0;
    i2cm->intfl = i2cm->intfl;

    // Transmit the command if there is command data and length
    if((cmd_data != NULL) && (cmd_len > 0)) {
        retval = I2CM_Tx(i2cm, fifo, addr, cmd_data, cmd_len, 0);
    }

    // Read data from the slave if we don't have any errors
    if(retval == E_NO_ERROR) {
        retval = I2CM_Rx(i2cm, fifo, addr, data, len);
    }

    // Wait for the transaction to complete
    if((error = I2CM_TxInProgress(i2cm)) != E_NO_ERROR) {
        retval = error;
    }

    // Unlock this I2CM
    mxc_free_lock((uint32_t*)&states[i2cm_num].req);

    if(retval != E_NO_ERROR) {
        return retval;
    }

    return len;
}

/* ************************************************************************* */
int I2CM_Write(mxc_i2cm_regs_t *i2cm, uint8_t addr, const uint8_t *cmd_data,
    uint32_t cmd_len, uint8_t* data, uint32_t len)
{
    int i2cm_num;
    int error = E_NO_ERROR;
    int retval = E_NO_ERROR;
    mxc_i2cm_fifo_regs_t *fifo;

    if(data == NULL) {
        return E_NULL_PTR;
    }

    // Make sure the I2CM has been initialized
    if(i2cm->ctrl == 0) {
        return E_UNINITIALIZED;
    }

    if(!(len > 0)) {
        return E_NO_ERROR;
    }

    // Lock this I2CM
    i2cm_num = MXC_I2CM_GET_IDX(i2cm);
    while(mxc_get_lock((uint32_t*)&states[i2cm_num].req,1) != E_NO_ERROR) {}

    // Get the FIFO pointer for this I2CM
    fifo = MXC_I2CM_GET_FIFO(i2cm_num);

    // Disable and clear the interrupts
	i2cm->inten = 0;
    i2cm->intfl = i2cm->intfl;

    // Transmit the command if there is command data and length, don't send stop bit
    if((cmd_data != NULL) && (cmd_len > 0)) {
        retval = I2CM_Tx(i2cm, fifo, addr, cmd_data, cmd_len, 0);
    }

    // Write data to the slave, send the stop bit
    if(retval == E_NO_ERROR) {
        retval = I2CM_Tx(i2cm, fifo, addr, data, len, 1);
    }

    // Wait for the transaction to complete
    if((error = I2CM_TxInProgress(i2cm)) != E_NO_ERROR) {
        retval = error;
    }

    // Unlock this I2CM
    mxc_free_lock((uint32_t*)&states[i2cm_num].req);

    if(retval != E_NO_ERROR) {
        return retval;
    }

    return len;
}

/* ************************************************************************* */
int I2CM_ReadAsync(mxc_i2cm_regs_t *i2cm, i2cm_req_t *req)
{
    int i2cm_num, error;

    if(req->data == NULL) {
        return E_NULL_PTR;
    }

    // Make sure the I2CM has been initialized
    if(i2cm->ctrl == 0) {
        return E_UNINITIALIZED;
    }

    if(!(req->data_len > 0)) {
        return E_NO_ERROR;
    }

    i2cm_num = MXC_I2CM_GET_IDX(i2cm);

    // Attempt to register this request
    if(mxc_get_lock((uint32_t*)&states[i2cm_num].req, (uint32_t)req) != E_NO_ERROR) {
        return E_BUSY;
    }

    states[i2cm_num].state = I2CM_STATE_READING;

    // Clear the number of bytes counter
    req->cmd_num = 0;
    req->data_num = 0;

    // Disable and clear the interrupts
    i2cm->inten = 0;
    i2cm->intfl = i2cm->intfl;

    // Start the read
    if((error = I2CM_ReadHandler(i2cm, req, i2cm_num)) != E_NO_ERROR) {
        I2CM_Recover(i2cm);
        I2CM_FreeCallback(i2cm_num, error);
        return error;
    }

    return E_NO_ERROR;
}

/* ************************************************************************* */
int I2CM_WriteAsync(mxc_i2cm_regs_t *i2cm, i2cm_req_t *req)
{
    int i2cm_num, error;

    if(req->data == NULL) {
        return E_NULL_PTR;
    }

    // Make sure the I2CM has been initialized
    if(i2cm->ctrl == 0) {
        return E_UNINITIALIZED;
    }

    if(!(req->data_len > 0)) {
        return E_NO_ERROR;
    }

    i2cm_num = MXC_I2CM_GET_IDX(i2cm);

    // Attempt to register this request
    if(mxc_get_lock((uint32_t*)&states[i2cm_num].req, (uint32_t)req) != E_NO_ERROR) {
        return E_BUSY;
    }

    states[i2cm_num].state = I2CM_STATE_WRITING;

    // Clear the number of bytes counter
    req->cmd_num = 0;
    req->data_num = 0;

    // Disable and clear the interrupts
    i2cm->inten = 0;
    i2cm->intfl = i2cm->intfl;

    // Start the Write
    if((error = I2CM_WriteHandler(i2cm, req, i2cm_num)) != E_NO_ERROR) {
        I2CM_Recover(i2cm);
        I2CM_FreeCallback(i2cm_num, error);
        return error;
    }

    return E_NO_ERROR;
}

/* ************************************************************************* */
int I2CM_AbortAsync(i2cm_req_t *req)
{
    int i2cm_num;
    mxc_i2cm_regs_t *i2cm;

    // Find the request, set to NULL
    for(i2cm_num = 0; i2cm_num < MXC_CFG_I2CM_INSTANCES; i2cm_num++)
    {
        if(req == states[i2cm_num].req) {

            i2cm = MXC_I2CM_GET_I2CM(i2cm_num);
            I2CM_Recover(i2cm);
            I2CM_FreeCallback(i2cm_num, E_ABORT);

            return E_NO_ERROR;
        }
    }

    return E_BAD_PARAM;
}

/* ************************************************************************* */
void I2CM_Handler(mxc_i2cm_regs_t *i2cm)
{
    uint32_t intfl;
    int i2cm_num, error;

    // Save and clear the interrupts
    intfl = i2cm->intfl;
    i2cm->intfl = intfl;

    // Mask the disabled interrupts
    intfl &= i2cm->inten;

    i2cm_num = MXC_I2CM_GET_IDX(i2cm);

    // Check for errors
    if ((intfl & MXC_F_I2CM_INTFL_TX_NACKED) || (intfl & MXC_F_I2CM_INTFL_TX_LOST_ARBITR)) {
        I2CM_Recover(i2cm);
        I2CM_FreeCallback(i2cm_num, E_COMM_ERR);
        return;
    }

    if(intfl & MXC_F_I2CM_INTFL_TX_TIMEOUT) {
        I2CM_Recover(i2cm);
        I2CM_FreeCallback(i2cm_num, E_TIME_OUT);
        return;
    }

    // Read or write
    if(states[i2cm_num].state == I2CM_STATE_READING) {
        if((error = I2CM_ReadHandler(i2cm, states[i2cm_num].req, i2cm_num)) != E_NO_ERROR) {
            I2CM_Recover(i2cm);
            I2CM_FreeCallback(i2cm_num, error);
            return;
        }

    } else if(states[i2cm_num].state == I2CM_STATE_WRITING) {
        if((error = I2CM_WriteHandler(i2cm, states[i2cm_num].req, i2cm_num)) != E_NO_ERROR) {
            I2CM_Recover(i2cm);
            I2CM_FreeCallback(i2cm_num, error);
            return;
        }
    }

    // Done with the transaction
    if(intfl & MXC_F_I2CM_INTFL_TX_DONE) {
        I2CM_Recover(i2cm);
        I2CM_FreeCallback(i2cm_num, E_NO_ERROR);
    }

}

/* ************************************************************************* */
int I2CM_Busy(mxc_i2cm_regs_t *i2cm)
{
    // Check to see if there are any ongoing transactions
    if((states[MXC_I2CM_GET_IDX(i2cm)].req == NULL) &&
        !(i2cm->trans & MXC_F_I2CM_TRANS_TX_IN_PROGRESS)) {

        return E_NO_ERROR;
    }

    return E_BUSY;
}

/* ************************************************************************* */
int I2CM_PrepForSleep(mxc_i2cm_regs_t *i2cm)
{
    if(I2CM_Busy(i2cm) != E_NO_ERROR) {
        return E_BUSY;
    }

    // Disable interrupts
    i2cm->inten = 0;
    return E_NO_ERROR;
}

/* ************************************************************************* */
int I2CM_BusCheck(mxc_i2cm_regs_t *i2cm)
{
    // If SCL is low, we don't have the bus
    if(!(i2cm->bb & MXC_F_I2CM_BB_BB_SCL_IN_VAL)) {
        return E_BUSY;
    }

    // If SDA is low, we don't have the bus
    if(!(i2cm->bb & MXC_F_I2CM_BB_BB_SDA_IN_VAL)) {
        return E_BUSY;
    }

    return E_NO_ERROR;
}

/* ************************************************************************* */
static void I2CM_FreeCallback(int i2cm_num, int error)
{
    // Save the request
    i2cm_req_t *temp_req = states[i2cm_num].req;

    // Unlock this UART to write
    mxc_free_lock((uint32_t*)&states[i2cm_num].req);

    // Callback if not NULL
    if(temp_req->callback != NULL) {
        temp_req->callback(temp_req, error);
    }
}

/* ************************************************************************* */
void I2CM_Recover(mxc_i2cm_regs_t *i2cm)
{
    // Disable and clear interrupts
    i2cm->inten = 0;
    i2cm->intfl = i2cm->intfl;
    i2cm->ctrl = MXC_F_I2CM_CTRL_MSTR_RESET_EN;
    i2cm->ctrl = MXC_F_I2CM_CTRL_TX_FIFO_EN | MXC_F_I2CM_CTRL_RX_FIFO_EN;
}

/* ************************************************************************* */
int I2CM_WriteTxFifo(mxc_i2cm_regs_t *i2cm, mxc_i2cm_fifo_regs_t *fifo, const uint16_t data)
{
    int32_t timeout = MXC_I2CM_TX_TIMEOUT;

    // Read the TX FIFO to determine if it's full
    do {

        // Wait for the TX FIFO to have room and check for errors
        if (i2cm->intfl & (MXC_F_I2CM_INTFL_TX_NACKED |
            MXC_F_I2CM_INTFL_TX_LOST_ARBITR)) {

            return E_COMM_ERR;
        }

        if((i2cm->intfl & MXC_F_I2CM_INTFL_TX_TIMEOUT) || !timeout--) {
            return E_TIME_OUT;
        }

    } while (fifo->tx);

    fifo->tx = data;

    return E_NO_ERROR;
}

/* ************************************************************************* */
int I2CM_TxInProgress(mxc_i2cm_regs_t *i2cm)
{
    int32_t timeout = MXC_I2CM_TX_TIMEOUT;

    while ((i2cm->trans & MXC_F_I2CM_TRANS_TX_IN_PROGRESS) && --timeout);

    if (i2cm->intfl & (MXC_F_I2CM_INTFL_TX_NACKED |
        MXC_F_I2CM_INTFL_TX_LOST_ARBITR)) {

        I2CM_Recover(i2cm);
        return E_COMM_ERR;
    }

    if((i2cm->intfl & MXC_F_I2CM_INTFL_TX_TIMEOUT) && !timeout--) {
        I2CM_Recover(i2cm);
        return E_TIME_OUT;
    }

    return E_NO_ERROR;
}

/* ************************************************************************* */
int I2CM_Tx(mxc_i2cm_regs_t *i2cm, mxc_i2cm_fifo_regs_t *fifo, uint8_t addr,
    const uint8_t *data, uint32_t len, uint8_t stop)
{
    uint32_t i;
    int error;

    // Write the address to the TXFIFO
    if((error = I2CM_WriteTxFifo(i2cm, fifo, (MXC_S_I2CM_TRANS_TAG_START | (addr << 1)))) != E_NO_ERROR) {
        return error;
    }

    // Start the transaction if it is not currently ongoing
    if (!(i2cm->trans & MXC_F_I2CM_TRANS_TX_IN_PROGRESS)) {
        i2cm->trans |= MXC_F_I2CM_TRANS_TX_START;
    }

    // Fill the FIFO
    for (i = 0; i < len; i++) {
        if((error = I2CM_WriteTxFifo(i2cm, fifo, (MXC_S_I2CM_TRANS_TAG_TXDATA_ACK | data[i]))) != E_NO_ERROR) {
            return error;
        }
    }

    // Send the stop condition
    if(stop) {
        if ((error = I2CM_WriteTxFifo(i2cm, fifo, MXC_S_I2CM_TRANS_TAG_STOP)) != E_NO_ERROR) {
            return error;
        }
    }

    return E_NO_ERROR;
}

/* ************************************************************************* */
static int I2CM_Rx(mxc_i2cm_regs_t *i2cm, mxc_i2cm_fifo_regs_t *fifo, uint8_t addr,
    uint8_t *data, uint32_t len)
{
    uint32_t i = len;
    int32_t timeout;
    uint16_t temp;
    int error;

    // Write the address to the TXFIFO
    if((error = I2CM_WriteTxFifo(i2cm, fifo, (MXC_S_I2CM_TRANS_TAG_START |
        (addr << 1) | I2CM_READ_BIT))) != E_NO_ERROR) {

        return error;
    }

    // Write to the TXFIFO the number of bytes we want to read
    while(i > 256) {
        if((error = I2CM_WriteTxFifo(i2cm, fifo, (MXC_S_I2CM_TRANS_TAG_RXDATA_COUNT | 255))) != E_NO_ERROR) {
            return error;
        }

        i -= 256;
    }

    if(i > 1) {
        if((error = I2CM_WriteTxFifo(i2cm, fifo, (MXC_S_I2CM_TRANS_TAG_RXDATA_COUNT | (i-2)))) != E_NO_ERROR) {
            return error;
        }
    }

    // Start the transaction if it is not currently ongoing
    if (!(i2cm->trans & MXC_F_I2CM_TRANS_TX_IN_PROGRESS)) {
        i2cm->trans |= MXC_F_I2CM_TRANS_TX_START;
    }


    // NACK the last read byte
    if((error = I2CM_WriteTxFifo(i2cm, fifo, (MXC_S_I2CM_TRANS_TAG_RXDATA_NACK))) != E_NO_ERROR) {
        return error;
    }

    // Send the stop condition
    if ((error = I2CM_WriteTxFifo(i2cm, fifo, MXC_S_I2CM_TRANS_TAG_STOP)) != E_NO_ERROR) {
        return error;
    }

    // Get the data from the RX FIFO
    i = 0;
    while (i < len) {

        // Wait for there to be data in the RX FIFO
        timeout = MXC_I2CM_RX_TIMEOUT;
        while (!(i2cm->intfl & MXC_F_I2CM_INTFL_RX_FIFO_NOT_EMPTY) &&
            ((i2cm->bb & MXC_F_I2CM_BB_RX_FIFO_CNT) == 0)) {

            if((timeout-- < 0) || (i2cm->trans & MXC_F_I2CM_TRANS_TX_TIMEOUT)) {
                return E_TIME_OUT;
            }

            if (i2cm->trans & (MXC_F_I2CM_TRANS_TX_LOST_ARBITR | MXC_F_I2CM_TRANS_TX_NACKED)) {
                return E_COMM_ERR;
            }
        }
        i2cm->intfl = MXC_F_I2CM_INTFL_RX_FIFO_NOT_EMPTY;

        // Save the data from the RX FIFO
        temp = fifo->rx;
        if (temp & MXC_S_I2CM_RSTLS_TAG_EMPTY) {
            continue;
        }
        data[i++] = (uint8_t)temp;
    }

    return E_NO_ERROR;
}

/* ************************************************************************* */
static int I2CM_CmdHandler(mxc_i2cm_regs_t *i2cm, mxc_i2cm_fifo_regs_t *fifo, i2cm_req_t *req)
{
    int error;

    // Start of the command
    if(req->cmd_num == 0) {

        // Write the address to the TXFIFO
        if((error = I2CM_WriteTxFifo(i2cm, fifo, (MXC_S_I2CM_TRANS_TAG_START | (req->addr << 1)))) != E_NO_ERROR) {
            return error;
        }

        // Start the transaction if it is not currently ongoing
        if (!(i2cm->trans & MXC_F_I2CM_TRANS_TX_IN_PROGRESS)) {
            i2cm->trans |= MXC_F_I2CM_TRANS_TX_START;
        }
    }

    // Write to the FIFO until it is full or we run out of command bytes
    while((req->cmd_num < req->cmd_len) && (!fifo->tx)) {
        fifo->tx = MXC_S_I2CM_TRANS_TAG_TXDATA_ACK | req->cmd_data[req->cmd_num++];
    }

    return E_NO_ERROR;
}

/* ************************************************************************* */
static int I2CM_ReadHandler(mxc_i2cm_regs_t *i2cm, i2cm_req_t *req, int i2cm_num)
{
    int error, cmd_remain, data_remain;
    uint16_t data;
    uint32_t temp_len, inten;
    mxc_i2cm_fifo_regs_t *fifo;

    // Get the FIFO pointer for this I2CM
    fifo = MXC_I2CM_GET_FIFO(i2cm_num);

    cmd_remain = req->cmd_len - req->cmd_num;
    data_remain = req->data_len - req->data_num;

    // Process the command portion
    if((cmd_remain) && (req->cmd_data != NULL)) {
        if((error = I2CM_CmdHandler(i2cm, fifo, req)) != E_NO_ERROR) {
            return error;
        }

        cmd_remain = req->cmd_len - req->cmd_num;
    }

    // Process the data portion
    if((cmd_remain == 0) && (data_remain)) {

        // Save the data from the RXFIFO
        data = fifo->rx;
        while((req->data_num < req->data_len) && !(data & MXC_S_I2CM_RSTLS_TAG_EMPTY)) {
            req->data[req->data_num++] = data;
            data = fifo->rx;
        }

        // Start of the data portion
        if(req->data_num == 0) {

            temp_len = req->data_len;

            // Write the address to the TXFIFO
            if((error = I2CM_WriteTxFifo(i2cm, fifo, (MXC_S_I2CM_TRANS_TAG_START |
                (req->addr << 1) | I2CM_READ_BIT))) != E_NO_ERROR) {

                return error;
            }

            // Write to the TXFIFO the number of bytes we want to read
            while(temp_len > 256) {
                if((error = I2CM_WriteTxFifo(i2cm, fifo, (MXC_S_I2CM_TRANS_TAG_RXDATA_COUNT | 255))) != E_NO_ERROR) {
                    return error;
                }

                temp_len -= 256;
            }

            if(temp_len > 1) {
                if((error = I2CM_WriteTxFifo(i2cm, fifo, (MXC_S_I2CM_TRANS_TAG_RXDATA_COUNT | (temp_len-2)))) != E_NO_ERROR) {
                    return error;
                }
            }

            // Start the transaction if it is not currently ongoing
            if (!(i2cm->trans & MXC_F_I2CM_TRANS_TX_IN_PROGRESS)) {
                i2cm->trans |= MXC_F_I2CM_TRANS_TX_START;
            }

            // NACK the last read byte
            if((error = I2CM_WriteTxFifo(i2cm, fifo, (MXC_S_I2CM_TRANS_TAG_RXDATA_NACK))) != E_NO_ERROR) {
                return error;
            }

            // Send the stop condition
            if ((error = I2CM_WriteTxFifo(i2cm, fifo, MXC_S_I2CM_TRANS_TAG_STOP)) != E_NO_ERROR) {
                return error;
            }
        }
    }

    // Enable the required interrupts
    inten = MXC_F_I2CM_INTEN_TX_DONE | MXC_F_I2CM_INTEN_TX_NACKED |
        MXC_F_I2CM_INTEN_TX_LOST_ARBITR | MXC_F_I2CM_INTEN_TX_TIMEOUT;

    if (cmd_remain) {
        inten |= (MXC_F_I2CM_INTEN_TX_FIFO_EMPTY | MXC_F_I2CM_INTEN_TX_FIFO_3Q_EMPTY);
    }

    data_remain = req->data_len - req->data_num;
    if (data_remain > I2CM_FIFO_DEPTH_3Q) {
        inten |= MXC_F_I2CM_INTEN_RX_FIFO_3Q_FULL;

    } else if (data_remain > I2CM_FIFO_DEPTH_2Q) {
        inten |= MXC_F_I2CM_INTEN_RX_FIFO_2Q_FULL;

    } else if (data_remain > 0) {
        inten |= MXC_F_I2CM_INTEN_RX_FIFO_NOT_EMPTY;
    }

    i2cm->inten = inten;

    return E_NO_ERROR;
}

/* ************************************************************************* */
static int I2CM_WriteHandler(mxc_i2cm_regs_t *i2cm, i2cm_req_t *req, int i2cm_num)
{
    int error, cmd_remain, data_remain;
    uint32_t inten;
    mxc_i2cm_fifo_regs_t *fifo;

    // Get the FIFO pointer for this I2CM
    fifo = MXC_I2CM_GET_FIFO(i2cm_num);

    cmd_remain = req->cmd_len - req->cmd_num;
    data_remain = req->data_len - req->data_num;

    // Process the command portion
    if((cmd_remain) && (req->cmd_data != NULL)) {
        if((error = I2CM_CmdHandler(i2cm, fifo, req)) != E_NO_ERROR) {
            return error;
        }

        cmd_remain = req->cmd_len - req->cmd_num;
    }

    // Process the data portion
    if((cmd_remain == 0) && (data_remain)) {

        // Start of the data portion
        if(req->data_num == 0) {

            // Write the address to the TXFIFO
            if((error = I2CM_WriteTxFifo(i2cm, fifo, (MXC_S_I2CM_TRANS_TAG_START |
                (req->addr << 1)))) != E_NO_ERROR) {

                return error;
            }

            // Start the transaction if it is not currently ongoing
            if (!(i2cm->trans & MXC_F_I2CM_TRANS_TX_IN_PROGRESS)) {
                i2cm->trans |= MXC_F_I2CM_TRANS_TX_START;
            }
        }

        // Write bytes to the FIFO until it's full or we run out of bytes
        while(req->data_num < req->data_len) {
            fifo->tx = MXC_S_I2CM_TRANS_TAG_TXDATA_ACK | req->data[req->data_num++];
        }

        // Send the stop condition
        if ((error = I2CM_WriteTxFifo(i2cm, fifo, MXC_S_I2CM_TRANS_TAG_STOP)) != E_NO_ERROR) {
            return error;
        }
    }

    // Enable the required interrupts
    data_remain = req->data_len - req->data_num;
    inten = MXC_F_I2CM_INTEN_TX_DONE | MXC_F_I2CM_INTEN_TX_NACKED |
        MXC_F_I2CM_INTEN_TX_LOST_ARBITR | MXC_F_I2CM_INTEN_TX_TIMEOUT;

    if(data_remain || cmd_remain) {
        inten |= (MXC_F_I2CM_INTEN_TX_FIFO_EMPTY | MXC_F_I2CM_INTEN_TX_FIFO_3Q_EMPTY);
    }
    i2cm->inten = inten;

    return E_NO_ERROR;
}
/**@} end of group i2cm */
