/**************************************************************************//**
 * @file     opa.h
 * @version  V1.00
 * @brief    M251 series OPA driver header file
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
#ifndef __OPA_H__
#define __OPA_H__

#ifdef __cplusplus
extern "C"
{
#endif


/** @addtogroup Standard_Driver Standard Driver
  @{
*/

/** @addtogroup OPA_Driver OPA Driver
  @{
*/

/** @addtogroup OPA_EXPORTED_CONSTANTS OPA Exported Constants
  @{
*/
#define OPA_CALIBRATION_CLK_1K                        (0UL)                     /*!< OPA calibration clock select 1 KHz  \hideinitializer */
#define OPA_CALIBRATION_RV_1_2_AVDD                   (0UL)                     /*!< OPA calibration reference voltage select 1/2 AVDD  \hideinitializer */
#define OPA_CALIBRATION_RV_H_L_VCM                    (1UL)                     /*!< OPA calibration reference voltage select from high vcm to low vcm \hideinitializer */

/*@}*/ /* end of group OPA_EXPORTED_CONSTANTS */

/** @addtogroup OPA_EXPORTED_FUNCTIONS OPA Exported Functions
  @{
*/

/*---------------------------------------------------------------------------------------------------------*/
/* Define OPA functions prototype                                                                         */
/*---------------------------------------------------------------------------------------------------------*/

__STATIC_INLINE int32_t OPA_Calibration(OPA_T *opa, uint32_t u32OpaNum, uint32_t u32ClockSel, uint32_t u32RefVol);

/**
  * @brief This macro is used to power on the OPA circuit
  * @param[in] opa The pointer of the specified OPA module
  * @param[in] u32OpaNum The OPA number. 0 for OPA0.
  * @return None
  * @details This macro will set OPENx (x=0) bit of OPA_CTL register to power on the OPA circuit.
  * \hideinitializer
  */
#define OPA_POWER_ON(opa, u32OpaNum) ((opa)->CTL |= (1UL<<(OPA_CTL_OPEN0_Pos+(u32OpaNum))))

/**
  * @brief This macro is used to power down the OPA circuit
  * @param[in] opa The pointer of the specified OPA module
  * @param[in] u32OpaNum The OPA number. 0 for OPA0.
  * @return None
  * @details This macro will clear OPENx (x=0) bit of OPA_CTL register to power down the OPA circuit.
  * \hideinitializer
  */
#define OPA_POWER_DOWN(opa, u32OpaNum) ((opa)->CTL &= ~(1UL<<(OPA_CTL_OPEN0_Pos+(u32OpaNum))))

/**
  * @brief This macro is used to enable the OPA Schmitt trigger buffer
  * @param[in] opa The pointer of the specified OPA module
  * @param[in] u32OpaNum The OPA number. 0 for OPA0.
  * @return None
  * @details This macro will set OPDOENx (x=0) bit of OPA_CTL register to enable the OPA Schmitt trigger buffer.
  * \hideinitializer
  */
#define OPA_ENABLE_SCH_TRIGGER(opa, u32OpaNum) ((opa)->CTL |= (1UL<<(OPA_CTL_OPDOEN0_Pos+(u32OpaNum))))

/**
  * @brief This macro is used to disable the OPA Schmitt trigger buffer
  * @param[in] opa The pointer of the specified OPA module
  * @param[in] u32OpaNum The OPA number. 0 for OPA0.
  * @return None
  * @details This macro will clear OPDOENx (x=0) bit of OPA_CTL register to disable the OPA Schmitt trigger buffer.
  * \hideinitializer
  */
#define OPA_DISABLE_SCH_TRIGGER(opa, u32OpaNum) ((opa)->CTL &= ~(1UL<<(OPA_CTL_OPDOEN0_Pos+(u32OpaNum))))

/**
  * @brief This macro is used to enable OPA Schmitt trigger digital output interrupt
  * @param[in] opa The pointer of the specified OPA module
  * @param[in] u32OpaNum The OPA number. 0 for OPA0.
  * @return None
  * @details This macro will set OPDOIENx (x=0) bit of OPA_CTL register to enable the OPA Schmitt trigger digital output interrupt.
  * \hideinitializer
  */
#define OPA_ENABLE_INT(opa, u32OpaNum) ((opa)->CTL |= (1UL<<(OPA_CTL_OPDOIEN0_Pos+(u32OpaNum))))

/**
  * @brief This macro is used to disable OPA Schmitt trigger digital output interrupt
  * @param[in] opa The pointer of the specified OPA module
  * @param[in] u32OpaNum The OPA number. 0 for OPA0.
  * @return None
  * @details This macro will clear OPDOIENx (x=0) bit of OPA_CTL register to disable the OPA Schmitt trigger digital output interrupt.
  * \hideinitializer
  */
#define OPA_DISABLE_INT(opa, u32OpaNum) ((opa)->CTL &= ~(1UL<<(OPA_CTL_OPDOIEN0_Pos+(u32OpaNum))))

/**
  * @brief This macro is used to get OPA digital output state
  * @param[in] opa The pointer of the specified OPA module
  * @param[in] u32OpaNum The OPA number. 0 for OPA0.
  * @return  OPA digital output state
  * @details This macro will return the OPA digital output value.
  * \hideinitializer
  */
#define OPA_GET_DIGITAL_OUTPUT(opa, u32OpaNum) (((opa)->STATUS & (OPA_STATUS_OPDO0_Msk<<(u32OpaNum)))?1UL:0UL)

/**
  * @brief This macro is used to get OPA interrupt flag
  * @param[in] opa The pointer of the specified OPA module
  * @param[in] u32OpaNum The OPA number. 0 for OPA0.
  * @retval     0 OPA interrupt does not occur.
  * @retval     1 OPA interrupt occurs.
  * @details This macro will return the OPA interrupt flag.
  * \hideinitializer
  */
#define OPA_GET_INT_FLAG(opa, u32OpaNum) (((opa)->STATUS & (OPA_STATUS_OPDOIF0_Msk<<(u32OpaNum)))?1UL:0UL)

/**
  * @brief This macro is used to clear OPA interrupt flag
  * @param[in] opa The pointer of the specified OPA module
  * @param[in] u32OpaNum The OPA number. 0 for OPA0.
  * @return   None
  * @details This macro will write 1 to OPDOIFx (x=0) bit of OPA_STATUS register to clear interrupt flag.
  * \hideinitializer
  */
#define OPA_CLR_INT_FLAG(opa, u32OpaNum) ((opa)->STATUS = (OPA_STATUS_OPDOIF0_Msk<<(u32OpaNum)))


/**
  * @brief This function is used to configure and start OPA calibration
  * @param[in] opa The pointer of the specified OPA module
  * @param[in] u32OpaNum The OPA number. 0 for OPA0.
  * @param[in] u32ClockSel Select OPA calibration clock
  *                 - \ref OPA_CALIBRATION_CLK_1K
  * @param[in] u32RefVol Select OPA reference voltage
  *                 - \ref OPA_CALIBRATION_RV_1_2_AVDD
  *                 - \ref OPA_CALIBRATION_RV_H_L_VCM
  * @retval      0 PMOS and NMOS calibration successfully.
  * @retval     -1 only PMOS calibration failed.
  * @retval     -2 only NMOS calibration failed.
  * @retval     -3 PMOS and NMOS calibration failed.
  */
__STATIC_INLINE int32_t OPA_Calibration(OPA_T *opa, uint32_t u32OpaNum, uint32_t u32ClockSel, uint32_t u32RefVol)
{
    uint32_t u32CALResult;
    int32_t i32Ret = 0L;

    (opa)->CALCTL = (((opa)->CALCTL) & ~(0x30ul << (u32OpaNum << 1))) | (((u32ClockSel) << 4) << (u32OpaNum << 1));
    (opa)->CALCTL = (((opa)->CALCTL) & ~(OPA_CALCTL_CALRVS0_Msk << (u32OpaNum))) | (((u32RefVol) << OPA_CALCTL_CALRVS0_Pos) << (u32OpaNum));
    (opa)->CALCTL |= (OPA_CALCTL_CALTRG0_Msk << (u32OpaNum));

    while ((opa)->CALCTL & (OPA_CALCTL_CALTRG0_Msk << (u32OpaNum))) {}

    u32CALResult = ((opa)->CALST >> ((u32OpaNum) * 4U)) & (OPA_CALST_CALNS0_Msk | OPA_CALST_CALPS0_Msk);

    if (u32CALResult == 0ul)
    {
        i32Ret = 0L;
    }
    else if (u32CALResult == OPA_CALST_CALNS0_Msk)
    {
        i32Ret = -2L;
    }
    else if (u32CALResult == OPA_CALST_CALPS0_Msk)
    {
        i32Ret = -1L;
    }
    else if (u32CALResult == (uint32_t)(OPA_CALST_CALNS0_Msk | OPA_CALST_CALPS0_Msk))
    {
        i32Ret = -3L;
    }
    else
    {

    }

    return i32Ret;
}



/*@}*/ /* end of group OPA_EXPORTED_FUNCTIONS */

/*@}*/ /* end of group OPA_Driver */

/*@}*/ /* end of group Standard_Driver */

#ifdef __cplusplus
}
#endif

#endif /* __OPA_H__ */

/*** (C) COPYRIGHT 2018 Nuvoton Technology Corp. ***/
