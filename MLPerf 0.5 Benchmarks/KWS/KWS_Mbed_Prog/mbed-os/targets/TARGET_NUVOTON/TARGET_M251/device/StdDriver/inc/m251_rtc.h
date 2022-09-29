/**************************************************************************//**
 * @file     rtc.h
 * @version  V1.00
 * @brief    M251 series RTC driver header file
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
#ifndef __RTC_H__
#define __RTC_H__

#ifdef __cplusplus
extern "C"
{
#endif


/** @addtogroup Standard_Driver Standard Driver
  @{
*/

/** @addtogroup RTC_Driver RTC Driver
  @{
*/

/** @addtogroup RTC_EXPORTED_CONSTANTS RTC Exported Constants
  @{
*/
/*---------------------------------------------------------------------------------------------------------*/
/*  RTC Initial Keyword Constant Definitions                                                               */
/*---------------------------------------------------------------------------------------------------------*/
#define RTC_INIT_KEY            0xA5EB1357UL    /*!< RTC Initiation Key to make RTC leaving reset state \hideinitializer */

/*---------------------------------------------------------------------------------------------------------*/
/*  RTC Frequency Compensation Definitions                                                                 */
/*---------------------------------------------------------------------------------------------------------*/
#define RTC_INTEGER_32752       (0x0ul << RTC_FREQADJ_INTEGER_Pos )    /*!< RTC Frequency is 32752HZ \hideinitializer */
#define RTC_INTEGER_32753       (0x1ul << RTC_FREQADJ_INTEGER_Pos )    /*!< RTC Frequency is 32753HZ \hideinitializer */
#define RTC_INTEGER_32754       (0x2ul << RTC_FREQADJ_INTEGER_Pos )    /*!< RTC Frequency is 32754HZ \hideinitializer */
#define RTC_INTEGER_32755       (0x3ul << RTC_FREQADJ_INTEGER_Pos )    /*!< RTC Frequency is 32755HZ \hideinitializer */
#define RTC_INTEGER_32756       (0x4ul << RTC_FREQADJ_INTEGER_Pos )    /*!< RTC Frequency is 32756HZ \hideinitializer */
#define RTC_INTEGER_32757       (0x5ul << RTC_FREQADJ_INTEGER_Pos )    /*!< RTC Frequency is 32757HZ \hideinitializer */
#define RTC_INTEGER_32758       (0x6ul << RTC_FREQADJ_INTEGER_Pos )    /*!< RTC Frequency is 32758HZ \hideinitializer */
#define RTC_INTEGER_32759       (0x7ul << RTC_FREQADJ_INTEGER_Pos )    /*!< RTC Frequency is 32759HZ \hideinitializer */
#define RTC_INTEGER_32760       (0x8ul << RTC_FREQADJ_INTEGER_Pos )    /*!< RTC Frequency is 32760HZ \hideinitializer */
#define RTC_INTEGER_32761       (0x9ul << RTC_FREQADJ_INTEGER_Pos )    /*!< RTC Frequency is 32761HZ \hideinitializer */
#define RTC_INTEGER_32762       (0xaul << RTC_FREQADJ_INTEGER_Pos )    /*!< RTC Frequency is 32762HZ \hideinitializer */
#define RTC_INTEGER_32763       (0xbul << RTC_FREQADJ_INTEGER_Pos )    /*!< RTC Frequency is 32763HZ \hideinitializer */
#define RTC_INTEGER_32764       (0xcul << RTC_FREQADJ_INTEGER_Pos )    /*!< RTC Frequency is 32764HZ \hideinitializer */
#define RTC_INTEGER_32765       (0xdul << RTC_FREQADJ_INTEGER_Pos )    /*!< RTC Frequency is 32765HZ \hideinitializer */
#define RTC_INTEGER_32766       (0xeul << RTC_FREQADJ_INTEGER_Pos )    /*!< RTC Frequency is 32766HZ \hideinitializer */
#define RTC_INTEGER_32767       (0xful << RTC_FREQADJ_INTEGER_Pos )    /*!< RTC Frequency is 32767HZ \hideinitializer */
#define RTC_INTEGER_32768       (0x10ul << RTC_FREQADJ_INTEGER_Pos )   /*!< RTC Frequency is 32768HZ \hideinitializer */
#define RTC_INTEGER_32769       (0x11ul << RTC_FREQADJ_INTEGER_Pos )   /*!< RTC Frequency is 32769HZ \hideinitializer */
#define RTC_INTEGER_32770       (0x12ul << RTC_FREQADJ_INTEGER_Pos )   /*!< RTC Frequency is 32770HZ \hideinitializer */
#define RTC_INTEGER_32771       (0x13ul << RTC_FREQADJ_INTEGER_Pos )   /*!< RTC Frequency is 32771HZ \hideinitializer */
#define RTC_INTEGER_32772       (0x14ul << RTC_FREQADJ_INTEGER_Pos )   /*!< RTC Frequency is 32772HZ \hideinitializer */
#define RTC_INTEGER_32773       (0x15ul << RTC_FREQADJ_INTEGER_Pos )   /*!< RTC Frequency is 32773HZ \hideinitializer */
#define RTC_INTEGER_32774       (0x16ul << RTC_FREQADJ_INTEGER_Pos )   /*!< RTC Frequency is 32774HZ \hideinitializer */
#define RTC_INTEGER_32775       (0x17ul << RTC_FREQADJ_INTEGER_Pos )   /*!< RTC Frequency is 32775HZ \hideinitializer */
#define RTC_INTEGER_32776       (0x18ul << RTC_FREQADJ_INTEGER_Pos )   /*!< RTC Frequency is 32776HZ \hideinitializer */
#define RTC_INTEGER_32777       (0x19ul << RTC_FREQADJ_INTEGER_Pos )   /*!< RTC Frequency is 32777HZ \hideinitializer */
#define RTC_INTEGER_32778       (0x1aul << RTC_FREQADJ_INTEGER_Pos )   /*!< RTC Frequency is 32778HZ \hideinitializer */
#define RTC_INTEGER_32779       (0x1bul << RTC_FREQADJ_INTEGER_Pos )   /*!< RTC Frequency is 32779HZ \hideinitializer */
#define RTC_INTEGER_32780       (0x1cul << RTC_FREQADJ_INTEGER_Pos )   /*!< RTC Frequency is 32780HZ \hideinitializer */
#define RTC_INTEGER_32781       (0x1dul << RTC_FREQADJ_INTEGER_Pos )   /*!< RTC Frequency is 32781HZ \hideinitializer */
#define RTC_INTEGER_32782       (0x1eul << RTC_FREQADJ_INTEGER_Pos )   /*!< RTC Frequency is 32782HZ \hideinitializer */
#define RTC_INTEGER_32783       (0x1ful << RTC_FREQADJ_INTEGER_Pos )   /*!< RTC Frequency is 32783HZ \hideinitializer */

/*---------------------------------------------------------------------------------------------------------*/
/*  RTC Time Attribute Constant Definitions                                                                */
/*---------------------------------------------------------------------------------------------------------*/
#define RTC_CLOCK_12            0UL               /*!< RTC as 12-hour time scale with AM and PM indication \hideinitializer */
#define RTC_CLOCK_24            1UL               /*!< RTC as 24-hour time scale \hideinitializer */
#define RTC_AM                  1UL               /*!< RTC as AM indication \hideinitializer */
#define RTC_PM                  2UL               /*!< RTC as PM indication \hideinitializer */

/*---------------------------------------------------------------------------------------------------------*/
/*  RTC Tick Period Constant Definitions                                                                   */
/*---------------------------------------------------------------------------------------------------------*/
#define RTC_TICK_1_SEC          0x0UL           /*!< RTC time tick period is 1 second \hideinitializer */
#define RTC_TICK_1_2_SEC        0x1UL           /*!< RTC time tick period is 1/2 second \hideinitializer */
#define RTC_TICK_1_4_SEC        0x2UL           /*!< RTC time tick period is 1/4 second \hideinitializer */
#define RTC_TICK_1_8_SEC        0x3UL           /*!< RTC time tick period is 1/8 second \hideinitializer */
#define RTC_TICK_1_16_SEC       0x4UL           /*!< RTC time tick period is 1/16 second \hideinitializer */
#define RTC_TICK_1_32_SEC       0x5UL           /*!< RTC time tick period is 1/32 second \hideinitializer */
#define RTC_TICK_1_64_SEC       0x6UL           /*!< RTC time tick period is 1/64 second \hideinitializer */
#define RTC_TICK_1_128_SEC      0x7UL           /*!< RTC time tick period is 1/128 second \hideinitializer */

/*---------------------------------------------------------------------------------------------------------*/
/*  RTC Day of Week Constant Definitions                                                                   */
/*---------------------------------------------------------------------------------------------------------*/
#define RTC_SUNDAY              0x0UL           /*!< Day of the Week is Sunday \hideinitializer */
#define RTC_MONDAY              0x1UL           /*!< Day of the Week is Monday \hideinitializer */
#define RTC_TUESDAY             0x2UL           /*!< Day of the Week is Tuesday \hideinitializer */
#define RTC_WEDNESDAY           0x3UL           /*!< Day of the Week is Wednesday \hideinitializer */
#define RTC_THURSDAY            0x4UL           /*!< Day of the Week is Thursday \hideinitializer */
#define RTC_FRIDAY              0x5UL           /*!< Day of the Week is Friday \hideinitializer */
#define RTC_SATURDAY            0x6UL           /*!< Day of the Week is Saturday \hideinitializer */

/*---------------------------------------------------------------------------------------------------------*/
/*  RTC Miscellaneous Constant Definitions                                                                         */
/*---------------------------------------------------------------------------------------------------------*/
#define RTC_WAIT_COUNT          0xFFFFFFFFUL      /*!< Initial Time-out Value \hideinitializer */
#define RTC_YEAR2000            2000UL            /*!< RTC Reference for compute year data \hideinitializer */
#define RTC_FCR_REFERENCE       32761UL           /*!< RTC Reference for frequency compensation \hideinitializer */


#define RTC_TAMPER0_SELECT (0x1ul << 0)     /*!< Select Tamper 0 \hideinitializer */
#define MAX_TAMPER_PIN_NUM 1ul              /*!< Tamper Pin number \hideinitializer */

#define RTC_TAMPER_HIGH_LEVEL_DETECT  0ul    /*!< Tamper pin detect voltage level is High  \hideinitializer */
#define RTC_TAMPER_LOW_LEVEL_DETECT   1ul    /*!< Tamper pin detect voltage level is low \hideinitializer */

#define RTC_TAMPER_DEBOUNCE_ENABLE   1ul    /*!< Enable RTC tamper pin de-bounce function \hideinitializer */
#define RTC_TAMPER_DEBOUNCE_DISABLE  0ul    /*!< Disable RTC tamper pin de-bounce function \hideinitializer */


/*@}*/ /* end of group RTC_EXPORTED_CONSTANTS */


/** @addtogroup RTC_EXPORTED_STRUCTS RTC Exported Structs
  @{
*/
/**
  * @details    RTC define Time Data Struct
  */
typedef struct
{
    uint32_t u32Year;           /*!< Year value */
    uint32_t u32Month;          /*!< Month value */
    uint32_t u32Day;            /*!< Day value */
    uint32_t u32DayOfWeek;      /*!< Day of week value */
    uint32_t u32Hour;           /*!< Hour value */
    uint32_t u32Minute;         /*!< Minute value */
    uint32_t u32Second;         /*!< Second value */
    uint32_t u32TimeScale;      /*!< 12-Hour, 24-Hour */
    uint32_t u32AmPm;           /*!< Only Time Scale select 12-hr used */
} S_RTC_TIME_DATA_T;

/*@}*/ /* end of group RTC_EXPORTED_STRUCTS */


/** @addtogroup RTC_EXPORTED_FUNCTIONS RTC Exported Functions
  @{
*/

/**
  * @brief      Indicate is Leap Year or not
  *
  * @param      None
  *
  * @retval     0   This year is not a leap year
  * @retval     1   This year is a leap year
  *
  * @details    According to current date, return this year is leap year or not.
  * \hideinitializer
  */
#define RTC_IS_LEAP_YEAR()              (RTC->LEAPYEAR & RTC_LEAPYEAR_LEAPYEAR_Msk ? 1:0)

/**
  * @brief      Clear RTC Alarm Interrupt Flag
  *
  * @param      None
  *
  * @return     None
  *
  * @details    This macro is used to clear RTC alarm interrupt flag.
  * \hideinitializer
  */
#define RTC_CLEAR_ALARM_INT_FLAG()      (RTC->INTSTS = RTC_INTSTS_ALMIF_Msk)

/**
  * @brief      Clear RTC Tick Interrupt Flag
  *
  * @param      None
  *
  * @return     None
  *
  * @details    This macro is used to clear RTC tick interrupt flag.
  * \hideinitializer
  */
#define RTC_CLEAR_TICK_INT_FLAG()       (RTC->INTSTS = RTC_INTSTS_TICKIF_Msk)

/**
  * @brief      Clear RTC Tamper Interrupt Flag
  *
  * @param      u32TamperFlag   Tamper interrupt flag. It consists of:    \n
  *                             - \ref RTC_INTSTS_TAMP0IF_Msk
  *
  * @return     None
  *
  * @details    This macro is used to clear RTC snooper pin interrupt flag.
  * \hideinitializer
  */
#define RTC_CLEAR_TAMPER_INT_FLAG(u32TamperFlag)    (RTC->INTSTS = (u32TamperFlag))

/**
  * @brief      Get RTC Alarm Interrupt Flag
  *
  * @param      None
  *
  * @retval     0   RTC alarm interrupt did not occur
  * @retval     1   RTC alarm interrupt occurred
  *
  * @details    This macro indicates RTC alarm interrupt occurred or not.
  * \hideinitializer
  */
#define RTC_GET_ALARM_INT_FLAG()        ((RTC->INTSTS & RTC_INTSTS_ALMIF_Msk)? 1:0)

/**
  * @brief      Get RTC Time Tick Interrupt Flag
  *
  * @param      None
  *
  * @retval     0   RTC time tick interrupt did not occur
  * @retval     1   RTC time tick interrupt occurred
  *
  * @details    This macro indicates RTC time tick interrupt occurred or not.
  * \hideinitializer
  */
#define RTC_GET_TICK_INT_FLAG()         ((RTC->INTSTS & RTC_INTSTS_TICKIF_Msk)? 1:0)

/**
  * @brief      Get RTC Tamper Interrupt Flag
  *
  * @param      None
  *
  * @retval     0   RTC snooper pin interrupt did not occur
  * @retval     1   RTC snooper pin interrupt occurred
  *
  * @details    This macro indicates RTC snooper pin interrupt occurred or not.
  * \hideinitializer
  */
#define RTC_GET_TAMPER_INT_FLAG()      ((RTC->INTSTS & (0x0100))? 1:0)

/**
  * @brief      Get RTC TAMPER Interrupt Status
  *
  * @param      None
  *
  * @retval     RTC_INTSTS_TAMP0IF_Msk    Tamper 0 interrupt flag is generated
  *
  * @details    This macro indicates RTC snooper pin interrupt occurred or not.
  * \hideinitializer
  */
#define RTC_GET_TAMPER_INT_STATUS()      ((RTC->INTSTS & (0x0100)))

/**
 * @brief      Enable RTC Tick Wake-up Function
 *
 * @param      None
 *
 * @return     None
 *
 * @details    This macro is used to enable RTC tick interrupt wake-up function.
 * \hideinitializer
 */
#define RTC_ENABLE_TICK_WAKEUP()         ((RTC->INTEN |= RTC_INTEN_TICKIEN_Msk))

/**
  * @brief      Disable RTC Tick Wake-up Function
  *
  * @param      None
  *
  * @return     None
  *
  * @details    This macro is used to disable RTC tick interrupt wake-up function.
  * \hideinitializer
  */
#define RTC_DISABLE_TICK_WAKEUP()        ((RTC->INTEN &= ~RTC_INTEN_TICKIEN_Msk));

/**
 * @brief      Enable RTC Alarm Wake-up Function
 *
 * @param      None
 *
 * @return     None
 *
 * @details    This macro is used to enable RTC Alarm interrupt wake-up function.
 * \hideinitializer
 */
#define RTC_ENABLE_ALARM_WAKEUP()         ((RTC->INTEN |= RTC_INTEN_ALMIEN_Msk))

/**
  * @brief      Disable RTC Alarm Wake-up Function
  *
  * @param      None
  *
  * @return     None
  *
  * @details    This macro is used to disable RTC Alarm interrupt wake-up function.
  * \hideinitializer
  */
#define RTC_DISABLE_ALARM_WAKEUP()        ((RTC->INTEN &= ~RTC_INTEN_ALMIEN_Msk));

/**
  * @brief      Read Spare Register
  *
  * @param[in]  u32RegNum   The spare register number, 0~4.
  *
  * @return     Spare register content
  *
  * @details    Read the specify spare register content.
  * @note       The returned value is valid only when SPRRWEN(SPRCTL[2] RTC Spare Function control register) bit is set. \n
  *             And its controlled by RTC Spare Function control register(RTC_SPRCTL).
  * \hideinitializer
  */
#define RTC_READ_SPARE_REGISTER(u32RegNum)                  (RTC->SPR[(u32RegNum)])

/**
  * @brief      Write Spare Register
  *
  * @param[in]  u32RegNum    The spare register number, 0~4.
  * @param[in]  u32RegValue  The spare register value.
  *
  * @return     None
  *
  * @details    Write specify data to spare register.
  * @note       This macro is effect only when SPRRWEN(SPRCTL[2] RTC Spare Function control register) bit is set. \n
  *             And its controlled by RTC Spare Function control register(RTC_SPRCTL).
  * \hideinitializer
  */
#define RTC_WRITE_SPARE_REGISTER(u32RegNum, u32RegValue)    (RTC->SPR[(u32RegNum)] = (u32RegValue))

/* Declare these inline functions here to avoid MISRA C 2004 rule 8.1 error */
static __INLINE void RTC_WaitAccessEnable(void);

/**
  * @brief      Wait RTC Access Enable
  *
  * @param      None
  *
  * @return     None
  *
  * @details    This function is used to enable the maximum RTC read/write accessible time.
  */
static __INLINE void RTC_WaitAccessEnable(void)
{
    /* Dummy for M251/M252 */
}

void RTC_Open(S_RTC_TIME_DATA_T *psPt);
void RTC_Close(void);
void RTC_32KCalibration(int32_t i32FrequencyX10000);
void RTC_GetDateAndTime(S_RTC_TIME_DATA_T *psPt);
void RTC_GetAlarmDateAndTime(S_RTC_TIME_DATA_T *psPt);
void RTC_SetDateAndTime(S_RTC_TIME_DATA_T *psPt);
void RTC_SetAlarmDateAndTime(S_RTC_TIME_DATA_T *psPt);
void RTC_SetDate(uint32_t u32Year, uint32_t u32Month, uint32_t u32Day, uint32_t u32DayOfWeek);
void RTC_SetTime(uint32_t u32Hour, uint32_t u32Minute, uint32_t u32Second, uint32_t u32TimeMode, uint32_t u32AmPm);
void RTC_SetAlarmDate(uint32_t u32Year, uint32_t u32Month, uint32_t u32Day);
void RTC_SetAlarmTime(uint32_t u32Hour, uint32_t u32Minute, uint32_t u32Second, uint32_t u32TimeMode, uint32_t u32AmPm);
void RTC_SetAlarmDateMask(uint8_t u8IsTenYMsk, uint8_t u8IsYMsk, uint8_t u8IsTenMMsk, uint8_t u8IsMMsk, uint8_t u8IsTenDMsk, uint8_t u8IsDMsk);
void RTC_SetAlarmTimeMask(uint8_t u8IsTenHMsk, uint8_t u8IsHMsk, uint8_t u8IsTenMMsk, uint8_t u8IsMMsk, uint8_t u8IsTenSMsk, uint8_t u8IsSMsk);
uint32_t RTC_GetDayOfWeek(void);
void RTC_SetTickPeriod(uint32_t u32TickSelection);
void RTC_EnableInt(uint32_t u32IntFlagMask);
void RTC_DisableInt(uint32_t u32IntFlagMask);
void RTC_EnableSpareAccess(void);
void RTC_DisableSpareRegister(void);
void RTC_StaticTamperEnable(uint32_t u32TamperSelect, uint32_t u32DetecLevel, uint32_t u32DebounceEn);
void RTC_StaticTamperDisable(uint32_t u32TamperSelect);


/*@}*/ /* end of group RTC_EXPORTED_FUNCTIONS */

/*@}*/ /* end of group RTC_Driver */

/*@}*/ /* end of group Standard_Driver */

#ifdef __cplusplus
}
#endif

#endif /* __RTC_H__ */

/*** (C) COPYRIGHT 2018 Nuvoton Technology Corp. ***/
