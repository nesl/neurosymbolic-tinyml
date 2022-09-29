/**************************************************************************//**
 * @file     bpwm.c
 * @version  V0.10
 * @brief    M251 series BPWM driver source file
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (c) 2019 Nuvoton Technology Corp. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *   1. Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *   2. Redistributions in binary form must reproduce the above copyright notice,
 *      this list of conditions and the following disclaimer in the documentation
 *      and/or other materials provided with the distribution.
 *   3. Neither the name of Nuvoton Technology Corp. nor the names of its contributors
 *      may be used to endorse or promote products derived from this software
 *      without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/
#include "NuMicro.h"

/** @addtogroup Standard_Driver Standard Driver
  @{
*/

/** @addtogroup BPWM_Driver BPWM Driver
  @{
*/


/** @addtogroup BPWM_EXPORTED_FUNCTIONS BPWM Exported Functions
  @{
*/

/**
 * @brief Configure BPWM capture and get the nearest unit time.
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. Valid values are between 0~5
 * @param[in] u32UnitTimeNsec The unit time of counter
 * @param[in] u32CaptureEdge The condition to latch the counter. This parameter is not used
 * @return The nearest unit time in nano second.
 * @details This function is used to Configure BPWM capture and get the nearest unit time.
 */
uint32_t BPWM_ConfigCaptureChannel(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32UnitTimeNsec, uint32_t u32CaptureEdge)
{
    uint32_t u32Src;
    uint32_t u32BPWMClockSrc;
    uint32_t u32NearestUnitTimeNsec;
    uint16_t u16Prescale = 1UL, u16CNR = 0xFFFFUL;
    uint8_t u8BreakLoop = 0UL;

    if (bpwm == BPWM0)
    {
        u32Src = CLK->CLKSEL2 & CLK_CLKSEL2_BPWM0SEL_Msk;
    }
    else     /* (bpwm == BPWM1) */
    {
        u32Src = CLK->CLKSEL2 & CLK_CLKSEL2_BPWM1SEL_Msk;
    }

    if (u32Src == 0UL)
    {
        //clock source is from PLL clock
        u32BPWMClockSrc = CLK_GetPLLClockFreq();
    }
    else
    {
        //clock source is from PCLK
        SystemCoreClockUpdate();

        if (bpwm == BPWM0)
        {
            u32BPWMClockSrc = CLK_GetPCLK0Freq();
        }
        else    /* (bpwm == BPWM1) */
        {
            u32BPWMClockSrc = CLK_GetPCLK1Freq();
        }
    }

    u32BPWMClockSrc /= 1000UL;

    for (u16Prescale = 1UL; u16Prescale <= 0x1000UL; u16Prescale++)
    {
        u32NearestUnitTimeNsec = (1000000UL * u16Prescale) / u32BPWMClockSrc;

        if (u32NearestUnitTimeNsec < u32UnitTimeNsec)
        {
            if (u16Prescale == 0x1000UL)
            {
                /* limit to the maximum unit time(nano second) */
                u8BreakLoop = 1UL;
            }

            if (!((1000000UL * (u16Prescale + 1UL) > (u32NearestUnitTimeNsec * u32BPWMClockSrc))))
            {
                u8BreakLoop = 1UL;
            }
        }
        else
        {
            u8BreakLoop = 1UL;
        }

        if (u8BreakLoop)
        {
            break;
        }
    }

    // convert to real register value
    u16Prescale = u16Prescale - 1UL;
    // all channels share a prescaler
    BPWM_SET_PRESCALER(bpwm, u32ChannelNum, (uint32_t)u16Prescale);

    // set BPWM to down count type(edge aligned)
    (bpwm)->CTL1 = BPWM_DOWN_COUNTER;

    BPWM_SET_CNR(bpwm, u32ChannelNum, u16CNR);

    return (u32NearestUnitTimeNsec);
}

/**
 * @brief This function Configure BPWM generator and get the nearest frequency in edge aligned(down countertype) auto-reload mode
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. Valid values are between 0~5
 * @param[in] u32Frequency Target generator frequency
 * @param[in] u32DutyCycle Target generator duty cycle percentage. Valid range are between 0 ~ 100. 10 means 10%, 20 means 20%...
 * @return Nearest frequency clock in nano second
 * @note Since all channels shares a prescaler. Call this API to configure BPWM frequency may affect
 *       existing frequency of other channel.
 */
uint32_t BPWM_ConfigOutputChannel(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32Frequency, uint32_t u32DutyCycle)
{
    uint32_t u32Src;
    uint32_t u32BPWMClockSrc;
    uint32_t i;
    uint16_t u16Prescale = 1UL, u16CNR = 0xFFFFUL;

    if (bpwm == BPWM0)
    {
        u32Src = CLK->CLKSEL2 & CLK_CLKSEL2_BPWM0SEL_Msk;
    }
    else     /* (bpwm == BPWM1) */
    {
        u32Src = CLK->CLKSEL2 & CLK_CLKSEL2_BPWM1SEL_Msk;
    }

    if (u32Src == 0UL)
    {
        //clock source is from PLL clock
        u32BPWMClockSrc = CLK_GetPLLClockFreq();
    }
    else
    {
        //clock source is from PCLK
        SystemCoreClockUpdate();

        if (bpwm == BPWM0)
        {
            u32BPWMClockSrc = CLK_GetPCLK0Freq();
        }
        else     /* (bpwm == BPWM1) */
        {
            u32BPWMClockSrc = CLK_GetPCLK1Freq();
        }
    }

    for (u16Prescale = 1UL; u16Prescale < 0xFFFUL; u16Prescale++)   //prescale could be 0~0xFFF
    {
        i = (u32BPWMClockSrc / u32Frequency) / u16Prescale;

        // If target value is larger than CNR, need to use a larger prescaler
        if (i <= (0x10000UL))
        {
            u16CNR = (uint16_t)i;
            break;
        }
    }

    // Store return value here 'cos we're gonna change u16Prescale & u16CNR to the real value to fill into register
    i = u32BPWMClockSrc / ((uint32_t)u16Prescale * (uint32_t)u16CNR);

    // convert to real register value
    u16Prescale = u16Prescale - 1UL;
    // all channels share a prescaler
    BPWM_SET_PRESCALER(bpwm, u32ChannelNum, (uint32_t)u16Prescale);
    // set BPWM to down count type
    (bpwm)->CTL1 = BPWM_DOWN_COUNTER;

    u16CNR = u16CNR - 1UL;
    BPWM_SET_CNR(bpwm, u32ChannelNum, u16CNR);

    if (u32DutyCycle)
    {
        if (u32DutyCycle >= 100UL)
            BPWM_SET_CMR(bpwm, u32ChannelNum, u16CNR);
        else
            BPWM_SET_CMR(bpwm, u32ChannelNum, u32DutyCycle * (u16CNR + 1UL) / 100UL);

        (bpwm)->WGCTL0 &= ~((BPWM_WGCTL0_PRDPCTL0_Msk | BPWM_WGCTL0_ZPCTL0_Msk) << (u32ChannelNum << 1UL));
        (bpwm)->WGCTL0 |= (BPWM_OUTPUT_LOW << ((u32ChannelNum << 1UL) + BPWM_WGCTL0_PRDPCTL0_Pos));
        (bpwm)->WGCTL1 &= ~((BPWM_WGCTL1_CMPDCTL0_Msk | BPWM_WGCTL1_CMPUCTL0_Msk) << (u32ChannelNum << 1UL));
        (bpwm)->WGCTL1 |= (BPWM_OUTPUT_HIGH << ((u32ChannelNum << 1UL) + BPWM_WGCTL1_CMPDCTL0_Pos));
    }
    else
    {
        BPWM_SET_CMR(bpwm, u32ChannelNum, 0UL);
        (bpwm)->WGCTL0 &= ~((BPWM_WGCTL0_PRDPCTL0_Msk | BPWM_WGCTL0_ZPCTL0_Msk) << (u32ChannelNum << 1UL));
        (bpwm)->WGCTL0 |= (BPWM_OUTPUT_LOW << ((u32ChannelNum << 1UL) + BPWM_WGCTL0_ZPCTL0_Pos));
        (bpwm)->WGCTL1 &= ~((BPWM_WGCTL1_CMPDCTL0_Msk | BPWM_WGCTL1_CMPUCTL0_Msk) << (u32ChannelNum << 1UL));
        (bpwm)->WGCTL1 |= (BPWM_OUTPUT_HIGH << ((u32ChannelNum << 1UL) + BPWM_WGCTL1_CMPDCTL0_Pos));
    }

    return (i);
}

/**
 * @brief Start BPWM module
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelMask Combination of enabled channels. This parameter is not used.
 * @return None
 * @details This function is used to start BPWM module.
 * @note All channels share one counter.
 */
void BPWM_Start(BPWM_T *bpwm, uint32_t u32ChannelMask)
{
    (bpwm)->CNTEN = BPWM_CNTEN_CNTEN0_Msk;
}

/**
 * @brief Stop BPWM module
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelMask Combination of enabled channels. This parameter is not used.
 * @return None
 * @details This function is used to stop BPWM module.
 * @note All channels share one period.
 */
void BPWM_Stop(BPWM_T *bpwm, uint32_t u32ChannelMask)
{
    (bpwm)->PERIOD = 0UL;
}

/**
 * @brief Stop BPWM generation immediately by clear channel enable bit
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelMask Combination of enabled channels. This parameter is not used.
 * @return None
 * @details This function is used to stop BPWM generation immediately by clear channel enable bit.
 * @note All channels share one counter.
 */
void BPWM_ForceStop(BPWM_T *bpwm, uint32_t u32ChannelMask)
{
    (bpwm)->CNTEN &= ~BPWM_CNTEN_CNTEN0_Msk;
}

/**
 * @brief Enable selected channel to trigger ADC
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. Valid values are between 0~5
 * @param[in] u32Condition The condition to trigger ADC. Combination of following conditions:
 *                  - \ref BPWM_TRIGGER_ADC_EVEN_ZERO_POINT
 *                  - \ref BPWM_TRIGGER_ADC_EVEN_PERIOD_POINT
 *                  - \ref BPWM_TRIGGER_ADC_EVEN_ZERO_OR_PERIOD_POINT
 *                  - \ref BPWM_TRIGGER_ADC_EVEN_CMP_UP_COUNT_POINT
 *                  - \ref BPWM_TRIGGER_ADC_EVEN_CMP_DOWN_COUNT_POINT
 *                  - \ref BPWM_TRIGGER_ADC_ODD_CMP_UP_COUNT_POINT
 *                  - \ref BPWM_TRIGGER_ADC_ODD_CMP_DOWN_COUNT_POINT
 * @return None
 * @details This function is used to enable selected channel to trigger ADC
 */
void BPWM_EnableADCTrigger(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32Condition)
{
    if (u32ChannelNum < 4UL)
    {
        (bpwm)->EADCTS0 &= ~((BPWM_EADCTS0_TRGSEL0_Msk) << (u32ChannelNum << 3UL));
        (bpwm)->EADCTS0 |= ((BPWM_EADCTS0_TRGEN0_Msk | u32Condition) << (u32ChannelNum << 3UL));
    }
    else
    {
        (bpwm)->EADCTS1 &= ~((BPWM_EADCTS1_TRGSEL4_Msk) << ((u32ChannelNum - 4UL) << 3UL));
        (bpwm)->EADCTS1 |= ((BPWM_EADCTS1_TRGEN4_Msk | u32Condition) << ((u32ChannelNum - 4UL) << 3UL));
    }
}

/**
 * @brief Disable selected channel to trigger ADC
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. Valid values are between 0~5
 * @return None
 * @details This function is used to disable selected channel to trigger ADC
 */
void BPWM_DisableADCTrigger(BPWM_T *bpwm, uint32_t u32ChannelNum)
{
    if (u32ChannelNum < 4UL)
    {
        (bpwm)->EADCTS0 &= ~(BPWM_EADCTS0_TRGEN0_Msk << (u32ChannelNum << 3UL));
    }
    else
    {
        (bpwm)->EADCTS1 &= ~(BPWM_EADCTS1_TRGEN4_Msk << ((u32ChannelNum - 4UL) << 3UL));
    }
}

/**
 * @brief Clear selected channel trigger ADC flag
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. Valid values are between 0~5
 * @param[in] u32Condition This parameter is not used
 * @return None
 * @details This function is used to clear selected channel trigger ADC flag
 */
void BPWM_ClearADCTriggerFlag(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32Condition)
{
    (bpwm)->STATUS = (BPWM_STATUS_EADCTRG0_Msk << u32ChannelNum);
}

/**
 * @brief Get selected channel trigger ADC flag
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. Valid values are between 0~5
 * @retval 0 The specified channel trigger ADC to start of conversion flag is not set
 * @retval 1 The specified channel trigger ADC to start of conversion flag is set
 * @details This function is used to get BPWM trigger ADC to start of conversion flag for specified channel
 */
uint32_t BPWM_GetADCTriggerFlag(BPWM_T *bpwm, uint32_t u32ChannelNum)
{
    return (((bpwm)->STATUS & (BPWM_STATUS_EADCTRG0_Msk << u32ChannelNum)) ? 1UL : 0UL);
}

/**
 * @brief Enable capture of selected channel(s)
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelMask Combination of enabled channels. Each bit corresponds to a channel.
 *                           Bit 0 is channel 0, bit 1 is channel 1...
 * @return None
 * @details This function is used to enable capture of selected channel(s)
 */
void BPWM_EnableCapture(BPWM_T *bpwm, uint32_t u32ChannelMask)
{
    (bpwm)->CAPINEN |= u32ChannelMask;
    (bpwm)->CAPCTL |= u32ChannelMask;
}

/**
 * @brief Disable capture of selected channel(s)
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelMask Combination of enabled channels. Each bit corresponds to a channel.
 *                           Bit 0 is channel 0, bit 1 is channel 1...
 * @return None
 * @details This function is used to disable capture of selected channel(s)
 */
void BPWM_DisableCapture(BPWM_T *bpwm, uint32_t u32ChannelMask)
{
    (bpwm)->CAPINEN &= ~u32ChannelMask;
    (bpwm)->CAPCTL &= ~u32ChannelMask;
}

/**
 * @brief Enables BPWM output generation of selected channel(s)
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelMask Combination of enabled channels. Each bit corresponds to a channel.
 *                           Set bit 0 to 1 enables channel 0 output, set bit 1 to 1 enables channel 1 output...
 * @return None
 * @details This function is used to enables BPWM output generation of selected channel(s)
 */
void BPWM_EnableOutput(BPWM_T *bpwm, uint32_t u32ChannelMask)
{
    (bpwm)->POEN |= u32ChannelMask;
}

/**
 * @brief Disables BPWM output generation of selected channel(s)
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelMask Combination of enabled channels. Each bit corresponds to a channel
 *                           Set bit 0 to 1 disables channel 0 output, set bit 1 to 1 disables channel 1 output...
 * @return None
 * @details This function is used to disables BPWM output generation of selected channel(s)
 */
void BPWM_DisableOutput(BPWM_T *bpwm, uint32_t u32ChannelMask)
{
    (bpwm)->POEN &= ~u32ChannelMask;
}

/**
 * @brief Enable capture interrupt of selected channel.
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. Valid values are between 0~5
 * @param[in] u32Edge Rising or falling edge to latch counter.
 *              - \ref BPWM_CAPTURE_INT_RISING_LATCH
 *              - \ref BPWM_CAPTURE_INT_FALLING_LATCH
 * @return None
 * @details This function is used to enable capture interrupt of selected channel.
 */
void BPWM_EnableCaptureInt(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32Edge)
{
    (bpwm)->CAPIEN |= (u32Edge << u32ChannelNum);
}

/**
 * @brief Disable capture interrupt of selected channel.
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. Valid values are between 0~5
 * @param[in] u32Edge Rising or falling edge to latch counter.
 *              - \ref BPWM_CAPTURE_INT_RISING_LATCH
 *              - \ref BPWM_CAPTURE_INT_FALLING_LATCH
 * @return None
 * @details This function is used to disable capture interrupt of selected channel.
 */
void BPWM_DisableCaptureInt(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32Edge)
{
    (bpwm)->CAPIEN &= ~(u32Edge << u32ChannelNum);
}

/**
 * @brief Clear capture interrupt of selected channel.
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. Valid values are between 0~5
 * @param[in] u32Edge Rising or falling edge to latch counter.
 *              - \ref BPWM_CAPTURE_INT_RISING_LATCH
 *              - \ref BPWM_CAPTURE_INT_FALLING_LATCH
 * @return None
 * @details This function is used to clear capture interrupt of selected channel.
 */
void BPWM_ClearCaptureIntFlag(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32Edge)
{
    (bpwm)->CAPIF = (u32Edge << u32ChannelNum);
}

/**
 * @brief Get capture interrupt of selected channel.
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. Valid values are between 0~5
 * @retval 0 No capture interrupt
 * @retval 1 Rising edge latch interrupt
 * @retval 2 Falling edge latch interrupt
 * @retval 3 Rising and falling latch interrupt
 * @details This function is used to get capture interrupt of selected channel.
 */
uint32_t BPWM_GetCaptureIntFlag(BPWM_T *bpwm, uint32_t u32ChannelNum)
{
    return (((((bpwm)->CAPIF & (BPWM_CAPIF_CAPFIF0_Msk << u32ChannelNum)) ? 1UL : 0UL) << 1UL) | \
            (((bpwm)->CAPIF & (BPWM_CAPIF_CAPRIF0_Msk << u32ChannelNum)) ? 1UL : 0UL));
}
/**
 * @brief Enable duty interrupt of selected channel
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. Valid values are between 0~5
 * @param[in] u32IntDutyType Duty interrupt type, could be either
 *              - \ref BPWM_DUTY_INT_DOWN_COUNT_MATCH_CMP
 *              - \ref BPWM_DUTY_INT_UP_COUNT_MATCH_CMP
 * @return None
 * @details This function is used to enable duty interrupt of selected channel.
 */
void BPWM_EnableDutyInt(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32IntDutyType)
{
    (bpwm)->INTEN |= (u32IntDutyType << u32ChannelNum);
}

/**
 * @brief Disable duty interrupt of selected channel
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. Valid values are between 0~5
 * @return None
 * @details This function is used to disable duty interrupt of selected channel
 */
void BPWM_DisableDutyInt(BPWM_T *bpwm, uint32_t u32ChannelNum)
{
    (bpwm)->INTEN &= ~((uint32_t)(BPWM_DUTY_INT_DOWN_COUNT_MATCH_CMP | BPWM_DUTY_INT_UP_COUNT_MATCH_CMP) << u32ChannelNum);
}

/**
 * @brief Clear duty interrupt flag of selected channel
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. Valid values are between 0~5
 * @return None
 * @details This function is used to clear duty interrupt flag of selected channel
 */
void BPWM_ClearDutyIntFlag(BPWM_T *bpwm, uint32_t u32ChannelNum)
{
    (bpwm)->INTSTS = (BPWM_INTSTS_CMPUIF0_Msk | BPWM_INTSTS_CMPDIF0_Msk) << u32ChannelNum;
}

/**
 * @brief Get duty interrupt flag of selected channel
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. Valid values are between 0~5
 * @return Duty interrupt flag of specified channel
 * @retval 0 Duty interrupt did not occur
 * @retval 1 Duty interrupt occurred
 * @details This function is used to get duty interrupt flag of selected channel
 */
uint32_t BPWM_GetDutyIntFlag(BPWM_T *bpwm, uint32_t u32ChannelNum)
{
    return ((((bpwm)->INTSTS & ((BPWM_INTSTS_CMPDIF0_Msk | BPWM_INTSTS_CMPUIF0_Msk) << u32ChannelNum))) ? 1UL : 0UL);
}

/**
 * @brief Enable period interrupt of selected channel
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. This parameter is not used.
 * @param[in] u32IntPeriodType Period interrupt type. This parameter is not used.
 * @return None
 * @details This function is used to enable period interrupt of selected channel.
 * @note All channels share channel 0's setting.
 */
void BPWM_EnablePeriodInt(BPWM_T *bpwm, uint32_t u32ChannelNum,  uint32_t u32IntPeriodType)
{
    (bpwm)->INTEN |= BPWM_INTEN_PIEN0_Msk;
}

/**
 * @brief Disable period interrupt of selected channel
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. This parameter is not used.
 * @return None
 * @details This function is used to disable period interrupt of selected channel.
 * @note All channels share channel 0's setting.
 */
void BPWM_DisablePeriodInt(BPWM_T *bpwm, uint32_t u32ChannelNum)
{
    (bpwm)->INTEN &= ~BPWM_INTEN_PIEN0_Msk;
}

/**
 * @brief Clear period interrupt of selected channel
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. This parameter is not used.
 * @return None
 * @details This function is used to clear period interrupt of selected channel
 * @note All channels share channel 0's setting.
 */
void BPWM_ClearPeriodIntFlag(BPWM_T *bpwm, uint32_t u32ChannelNum)
{
    (bpwm)->INTSTS = BPWM_INTSTS_PIF0_Msk;
}

/**
 * @brief Get period interrupt of selected channel
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. This parameter is not used.
 * @return Period interrupt flag of specified channel
 * @retval 0 Period interrupt did not occur
 * @retval 1 Period interrupt occurred
 * @details This function is used to get period interrupt of selected channel
 * @note All channels share channel 0's setting.
 */
uint32_t BPWM_GetPeriodIntFlag(BPWM_T *bpwm, uint32_t u32ChannelNum)
{
    return (((bpwm)->INTSTS & BPWM_INTSTS_PIF0_Msk) ? 1UL : 0UL);
}

/**
 * @brief Enable zero interrupt of selected channel
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. This parameter is not used.
 * @return None
 * @details This function is used to enable zero interrupt of selected channel.
 * @note All channels share channel 0's setting.
 */
void BPWM_EnableZeroInt(BPWM_T *bpwm, uint32_t u32ChannelNum)
{
    (bpwm)->INTEN |= BPWM_INTEN_ZIEN0_Msk;
}

/**
 * @brief Disable zero interrupt of selected channel
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. This parameter is not used.
 * @return None
 * @details This function is used to disable zero interrupt of selected channel.
 * @note All channels share channel 0's setting.
 */
void BPWM_DisableZeroInt(BPWM_T *bpwm, uint32_t u32ChannelNum)
{
    (bpwm)->INTEN &= ~BPWM_INTEN_ZIEN0_Msk;
}

/**
 * @brief Clear zero interrupt of selected channel
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. This parameter is not used.
 * @return None
 * @details This function is used to clear zero interrupt of selected channel.
 * @note All channels share channel 0's setting.
 */
void BPWM_ClearZeroIntFlag(BPWM_T *bpwm, uint32_t u32ChannelNum)
{
    (bpwm)->INTSTS = BPWM_INTSTS_ZIF0_Msk;
}

/**
 * @brief Get zero interrupt of selected channel
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. This parameter is not used.
 * @return zero interrupt flag of specified channel
 * @retval 0 zero interrupt did not occur
 * @retval 1 zero interrupt occurred
 * @details This function is used to get zero interrupt of selected channel.
 * @note All channels share channel 0's setting.
 */
uint32_t BPWM_GetZeroIntFlag(BPWM_T *bpwm, uint32_t u32ChannelNum)
{
    return (((bpwm)->INTSTS & BPWM_INTSTS_ZIF0_Msk) ? 1UL : 0UL);
}

/**
 * @brief Enable load mode of selected channel
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. Valid values are between 0~5
 * @param[in] u32LoadMode BPWM counter loading mode.
 *              - \ref BPWM_LOAD_MODE_IMMEDIATE
 *              - \ref BPWM_LOAD_MODE_CENTER
 * @return None
 * @details This function is used to enable load mode of selected channel.
 */
void BPWM_EnableLoadMode(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32LoadMode)
{
    (bpwm)->CTL0 |= (u32LoadMode << u32ChannelNum);
}

/**
 * @brief Disable load mode of selected channel
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. Valid values are between 0~5
 * @param[in] u32LoadMode BPWM counter loading mode.
 *              - \ref BPWM_LOAD_MODE_IMMEDIATE
 *              - \ref BPWM_LOAD_MODE_CENTER
 * @return None
 * @details This function is used to disable load mode of selected channel.
 */
void BPWM_DisableLoadMode(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32LoadMode)
{
    (bpwm)->CTL0 &= ~(u32LoadMode << u32ChannelNum);
}

/**
 * @brief Set BPWM clock source
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. This parameter is not used.
 * @param[in] u32ClkSrcSel BPWM external clock source.
 *              - \ref BPWM_CLKSRC_BPWM_CLK
 *              - \ref BPWM_CLKSRC_TIMER0
 *              - \ref BPWM_CLKSRC_TIMER1
 *              - \ref BPWM_CLKSRC_TIMER2
 *              - \ref BPWM_CLKSRC_TIMER3
 * @return None
 * @details This function is used to set BPWM clock source.
 * @note All channels share channel 0's setting.
 */
void BPWM_SetClockSource(BPWM_T *bpwm, uint32_t u32ChannelNum, uint32_t u32ClkSrcSel)
{
    (bpwm)->CLKSRC = (u32ClkSrcSel);
}

/**
 * @brief Get the time-base counter reached its maximum value flag of selected channel
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. This parameter is not used.
 * @return Count to max interrupt flag of specified channel
 * @retval 0 Count to max interrupt did not occur
 * @retval 1 Count to max interrupt occurred
 * @details This function is used to get the time-base counter reached its maximum value flag of selected channel.
 * @note All channels share channel 0's setting.
 */
uint32_t BPWM_GetWrapAroundFlag(BPWM_T *bpwm, uint32_t u32ChannelNum)
{
    return (((bpwm)->STATUS & BPWM_STATUS_CNTMAX0_Msk) ? 1UL : 0UL);
}

/**
 * @brief Clear the time-base counter reached its maximum value flag of selected channel
 * @param[in] bpwm The pointer of the specified BPWM module
 *                - BPWM0 : BPWM Group 0
 *                - BPWM1 : BPWM Group 1
 * @param[in] u32ChannelNum BPWM channel number. This parameter is not used.
 * @return None
 * @details This function is used to clear the time-base counter reached its maximum value flag of selected channel.
 * @note All channels share channel 0's setting.
 */
void BPWM_ClearWrapAroundFlag(BPWM_T *bpwm, uint32_t u32ChannelNum)
{
    (bpwm)->STATUS = BPWM_STATUS_CNTMAX0_Msk;
}


/*@}*/ /* end of group BPWM_EXPORTED_FUNCTIONS */

/*@}*/ /* end of group BPWM_Driver */

/*@}*/ /* end of group Standard_Driver */

/*** (C) COPYRIGHT 2018 Nuvoton Technology Corp. ***/
