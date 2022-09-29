/******************************************************************************
 * @file     usbd.h
 * @brief    M451 series USB driver header file
 * @version  2.0.0
 * @date     10, January, 2014
 *
 * @note
 * Copyright (C) 2014~2015 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef __USBD_H__
#define __USBD_H__

#ifdef __cplusplus
extern "C"
{
#endif

/** @addtogroup Standard_Driver Standard Driver
  @{
*/

/** @addtogroup USBD_Driver USBD Driver
  @{
*/

/** @addtogroup USBD_EXPORTED_STRUCTS USBD Exported Structs
  @{
*/
typedef struct s_usbd_info
{
    const uint8_t *gu8DevDesc;            /*!< Pointer for USB Device Descriptor          */
    const uint8_t *gu8ConfigDesc;         /*!< Pointer for USB Configuration Descriptor   */
    const uint8_t **gu8StringDesc;        /*!< Pointer for USB String Descriptor pointers */
    const uint8_t **gu8HidReportDesc;     /*!< Pointer for USB HID Report Descriptor      */
    const uint32_t *gu32HidReportSize;    /*!< Pointer for HID Report descriptor Size */
    const uint32_t *gu32ConfigHidDescIdx; /*!< Pointer for HID Descriptor start index */	

} S_USBD_INFO_T;

extern const S_USBD_INFO_T gsInfo;

/*@}*/ /* end of group USBD_EXPORTED_STRUCTS */




/** @addtogroup USBD_EXPORTED_CONSTANTS USBD Exported Constants
  @{
*/
#define USBD_BUF_BASE   (USBD_BASE+0x100)
#define USBD_MAX_EP     8

#define EP0     0       /*!< Endpoint 0 */
#define EP1     1       /*!< Endpoint 1 */
#define EP2     2       /*!< Endpoint 2 */
#define EP3     3       /*!< Endpoint 3 */
#define EP4     4       /*!< Endpoint 4 */
#define EP5     5       /*!< Endpoint 5 */
#define EP6     6       /*!< Endpoint 6 */
#define EP7     7       /*!< Endpoint 7 */


/*!<USB Request Type */
#define REQ_STANDARD        0x00
#define REQ_CLASS           0x20
#define REQ_VENDOR          0x40

/*!<USB Standard Request */
#define USBD_GET_STATUS          0x00
#define USBD_CLEAR_FEATURE       0x01
#define USBD_SET_FEATURE         0x03
#define USBD_SET_ADDRESS         0x05
#define USBD_GET_DESCRIPTOR      0x06
#define USBD_SET_DESCRIPTOR      0x07
#define USBD_GET_CONFIGURATION   0x08
#define USBD_SET_CONFIGURATION   0x09
#define USBD_GET_INTERFACE       0x0A
#define USBD_SET_INTERFACE       0x0B
#define USBD_SYNC_FRAME          0x0C

/*!<USB Descriptor Type */
#define DESC_DEVICE         0x01
#define DESC_CONFIG         0x02
#define DESC_STRING         0x03
#define DESC_INTERFACE      0x04
#define DESC_ENDPOINT       0x05
#define DESC_QUALIFIER      0x06
#define DESC_OTHERSPEED     0x07

/*!<USB HID Descriptor Type */
#define DESC_HID            0x21
#define DESC_HID_RPT        0x22

/*!<USB Descriptor Length */
#define LEN_DEVICE          18
#define LEN_CONFIG          9
#define LEN_INTERFACE       9
#define LEN_ENDPOINT        7
#define LEN_HID             9
#define LEN_CCID            0x36

/*!<USB Endpoint Type */
#define EP_ISO              0x01
#define EP_BULK             0x02
#define EP_INT              0x03

#define EP_INPUT            0x80
#define EP_OUTPUT           0x00

/*!<USB Feature Selector */
#define FEATURE_DEVICE_REMOTE_WAKEUP    0x01
#define FEATURE_ENDPOINT_HALT           0x00

/******************************************************************************/
/*                USB Specific Macros                                         */
/******************************************************************************/

#define USBD_WAKEUP_EN          USBD_INTEN_WKEN_Msk         /*!< USB Wake-up Enable */
#define USBD_DRVSE0             USBD_SE0_SE0_Msk            /*!< Drive SE0 */

#define USBD_DPPU_EN            USBD_ATTR_DPPUEN_Msk        /*!< USB D+ Pull-up Enable */
#define USBD_PWRDN              USBD_ATTR_PWRDN_Msk         /*!< PHY Turn-On */
#define USBD_PHY_EN             USBD_ATTR_PHYEN_Msk         /*!< PHY Enable */
#define USBD_USB_EN             USBD_ATTR_USBEN_Msk         /*!< USB Enable */

#define USBD_INT_BUS            USBD_INTEN_BUSIEN_Msk       /*!< USB Bus Event Interrupt */
#define USBD_INT_USB            USBD_INTEN_USBIEN_Msk       /*!< USB Event Interrupt */
#define USBD_INT_FLDET          USBD_INTEN_VBDETIEN_Msk     /*!< USB VBUS Detection Interrupt */
#define USBD_INT_WAKEUP         (USBD_INTEN_NEVWKIEN_Msk | USBD_INTEN_WKEN_Msk)     /*!< USB No-Event-Wake-Up Interrupt */

#define USBD_INTSTS_WAKEUP      USBD_INTSTS_NEVWKIF_Msk     /*!< USB No-Event-Wake-Up Interrupt Status */
#define USBD_INTSTS_FLDET       USBD_INTSTS_VBDETIF_Msk     /*!< USB Float Detect Interrupt Status */
#define USBD_INTSTS_BUS         USBD_INTSTS_BUSIF_Msk       /*!< USB Bus Event Interrupt Status */
#define USBD_INTSTS_USB         USBD_INTSTS_USBIF_Msk       /*!< USB Event Interrupt Status */
#define USBD_INTSTS_SETUP       USBD_INTSTS_SETUP_Msk       /*!< USB Setup Event */
#define USBD_INTSTS_EP0         USBD_INTSTS_EPEVT0_Msk      /*!< USB Endpoint 0 Event */
#define USBD_INTSTS_EP1         USBD_INTSTS_EPEVT1_Msk      /*!< USB Endpoint 1 Event */
#define USBD_INTSTS_EP2         USBD_INTSTS_EPEVT2_Msk      /*!< USB Endpoint 2 Event */
#define USBD_INTSTS_EP3         USBD_INTSTS_EPEVT3_Msk      /*!< USB Endpoint 3 Event */
#define USBD_INTSTS_EP4         USBD_INTSTS_EPEVT4_Msk      /*!< USB Endpoint 4 Event */
#define USBD_INTSTS_EP5         USBD_INTSTS_EPEVT5_Msk      /*!< USB Endpoint 5 Event */
#define USBD_INTSTS_EP6         USBD_INTSTS_EPEVT6_Msk      /*!< USB Endpoint 6 Event */
#define USBD_INTSTS_EP7         USBD_INTSTS_EPEVT7_Msk      /*!< USB Endpoint 7 Event */

#define USBD_STATE_USBRST       USBD_ATTR_USBRST_Msk        /*!< USB Bus Reset */
#define USBD_STATE_SUSPEND      USBD_ATTR_SUSPEND_Msk       /*!< USB Bus Suspend */
#define USBD_STATE_RESUME       USBD_ATTR_RESUME_Msk        /*!< USB Bus Resume */
#define USBD_STATE_TIMEOUT      USBD_ATTR_TOUT_Msk          /*!< USB Bus Timeout */

#define USBD_CFGP_SSTALL        USBD_CFGP_SSTALL_Msk        /*!< Set Stall */
#define USBD_CFG_CSTALL         USBD_CFG_CSTALL_Msk         /*!< Clear Stall */

#define USBD_CFG_EPMODE_DISABLE (0ul << USBD_CFG_STATE_Pos)/*!< Endpoint Disable */
#define USBD_CFG_EPMODE_OUT     (1ul << USBD_CFG_STATE_Pos)/*!< Out Endpoint */
#define USBD_CFG_EPMODE_IN      (2ul << USBD_CFG_STATE_Pos)/*!< In Endpoint */
#define USBD_CFG_TYPE_ISO       (1ul << USBD_CFG_ISOCH_Pos) /*!< Isochronous */



/*@}*/ /* end of group USBD_EXPORTED_CONSTANTS */


/** @addtogroup USBD_EXPORTED_FUNCTIONS USBD Exported Functions
  @{
*/

/**
  * @brief      Compare two input numbers and return maximum one.
  *
  * @param[in]  a   First number to be compared.
  * @param[in]  b   Second number to be compared.
  *
  * @return     Maximum value between a and b.
  *
  * @details    If a > b, then return a. Otherwise, return b.
  */
#define Maximum(a,b)        ((a)>(b) ? (a) : (b))


/**
  * @brief      Compare two input numbers and return minimum one
  *
  * @param[in]  a   First number to be compared
  * @param[in]  b   Second number to be compared
  *
  * @return     Minimum value between a and b
  *
  * @details    If a < b, then return a. Otherwise, return b.
  */
#define Minimum(a,b)        ((a)<(b) ? (a) : (b))


/**
  * @brief    Enable USB
  *
  * @param    None
  *
  * @return   None
  *
  * @details  To set USB ATTR control register to enable USB and PHY.
  *
  */
#define USBD_ENABLE_USB()           ((uint32_t)(USBD->ATTR |= (USBD_USB_EN|USBD_PHY_EN)))

/**
  * @brief    Disable USB
  *
  * @param    None
  *
  * @return   None
  *
  * @details  To set USB ATTR control register to disable USB.
  *
  */
#define USBD_DISABLE_USB()          ((uint32_t)(USBD->ATTR &= ~USBD_USB_EN))

/**
  * @brief    Enable USB PHY
  *
  * @param    None
  *
  * @return   None
  *
  * @details  To set USB ATTR control register to enable USB PHY.
  *
  */
#define USBD_ENABLE_PHY()           ((uint32_t)(USBD->ATTR |= USBD_PHY_EN))

/**
  * @brief    Disable USB PHY
  *
  * @param    None
  *
  * @return   None
  *
  * @details  To set USB ATTR control register to disable USB PHY.
  *
  */
#define USBD_DISABLE_PHY()          ((uint32_t)(USBD->ATTR &= ~USBD_PHY_EN))

/**
  * @brief    Enable SE0. Force USB PHY transceiver to drive SE0.
  *
  * @param    None
  *
  * @return   None
  *
  * @details  Set DRVSE0 bit of USB_DRVSE0 register to enable software-disconnect function. Force USB PHY transceiver to drive SE0 to bus.
  *
  */
#define USBD_SET_SE0()              ((uint32_t)(USBD->SE0 |= USBD_DRVSE0))

/**
  * @brief    Disable SE0
  *
  * @param    None
  *
  * @return   None
  *
  * @details  Clear DRVSE0 bit of USB_DRVSE0 register to disable software-disconnect function.
  *
  */
#define USBD_CLR_SE0()              ((uint32_t)(USBD->SE0 &= ~USBD_DRVSE0))

/**
  * @brief       Set USB device address
  *
  * @param[in]   addr The USB device address.
  *
  * @return      None
  *
  * @details     Write USB device address to USB_FADDR register.
  *
  */
#define USBD_SET_ADDR(addr)         (USBD->FADDR = (addr))

/**
  * @brief    Get USB device address
  *
  * @param    None
  *
  * @return   USB device address
  *
  * @details  Read USB_FADDR register to get USB device address.
  *
  */
#define USBD_GET_ADDR()             ((uint32_t)(USBD->FADDR))

/**
  * @brief      Enable USB interrupt function
  *
  * @param[in]  intr The combination of the specified interrupt enable bits.
  *             Each bit corresponds to a interrupt enable bit.
  *             This parameter decides which interrupts will be enabled.
  *             (USBD_INT_WAKEUP, USBD_INT_FLDET, USBD_INT_USB, USBD_INT_BUS)
  *
  * @return     None
  *
  * @details    Enable USB related interrupt functions specified by intr parameter.
  *
  */
#define USBD_ENABLE_INT(intr)       (USBD->INTEN |= (intr))

/**
  * @brief    Get interrupt status
  *
  * @param    None
  *
  * @return   The value of USB_INTSTS register
  *
  * @details  Return all interrupt flags of USB_INTSTS register.
  *
  */
#define USBD_GET_INT_FLAG()         ((uint32_t)(USBD->INTSTS))

/**
  * @brief      Clear USB interrupt flag
  *
  * @param[in]  flag The combination of the specified interrupt flags.
  *             Each bit corresponds to a interrupt source.
  *             This parameter decides which interrupt flags will be cleared.
  *             (USBD_INTSTS_WAKEUP, USBD_INTSTS_FLDET, USBD_INTSTS_BUS, USBD_INTSTS_USB)
  *
  * @return     None
  *
  * @details    Clear USB related interrupt flags specified by flag parameter.
  *
  */
#define USBD_CLR_INT_FLAG(flag)     (USBD->INTSTS = (flag))

/**
  * @brief    Get endpoint status
  *
  * @param    None
  *
  * @return   The value of USB_EPSTS register.
  *
  * @details  Return all endpoint status.
  *
  */
#define USBD_GET_EP_FLAG()          ((uint32_t)(USBD->EPSTS))

/**
  * @brief    Get USB bus state
  *
  * @param    None
  *
  * @return   The value of USB_ATTR[3:0].
  *           Bit 0 indicates USB bus reset status.
  *           Bit 1 indicates USB bus suspend status.
  *           Bit 2 indicates USB bus resume status.
  *           Bit 3 indicates USB bus time-out status.
  *
  * @details  Return USB_ATTR[3:0] for USB bus events.
  *
  */
#define USBD_GET_BUS_STATE()        ((uint32_t)(USBD->ATTR & 0xf))

/**
  * @brief    Check cable connection state
  *
  * @param    None
  *
  * @retval   0 USB cable is not attached.
  * @retval   1 USB cable is attached.
  *
  * @details  Check the connection state by FLDET bit of USB_FLDET register.
  *
  */
#define USBD_IS_ATTACHED()          ((uint32_t)(USBD->VBUSDET & USBD_VBUSDET_VBUSDET_Msk))

/**
  * @brief      Stop USB transaction of the specified endpoint ID
  *
  * @param[in]  ep The USB endpoint ID. M451 Series supports 8 hardware endpoint ID. This parameter could be 0 ~ 7.
  *
  * @return     None
  *
  * @details    Write 1 to CLRRDY bit of USB_CFGPx register to stop USB transaction of the specified endpoint ID.
  *
  */
#define USBD_STOP_TRANSACTION(ep)   (*((__IO uint32_t *) ((uint32_t)&USBD->EP[0].CFGP + (uint32_t)((ep) << 4))) |= USBD_CFGP_CLRRDY_Msk)

/**
  * @brief      Set USB DATA1 PID for the specified endpoint ID
  *
  * @param[in]  ep The USB endpoint ID. M451 Series supports 8 hardware endpoint ID. This parameter could be 0 ~ 7.
  *
  * @return     None
  *
  * @details    Set DSQ_SYNC bit of USB_CFGx register to specify the DATA1 PID for the following IN token transaction.
  *             Base on this setting, hardware will toggle PID between DATA0 and DATA1 automatically for IN token transactions.
  *
  */
#define USBD_SET_DATA1(ep)          (*((__IO uint32_t *) ((uint32_t)&USBD->EP[0].CFG + (uint32_t)((ep) << 4))) |= USBD_CFG_DSQSYNC_Msk)

/**
  * @brief      Set USB DATA0 PID for the specified endpoint ID
  *
  * @param[in]  ep The USB endpoint ID. M451 Series supports 8 hardware endpoint ID. This parameter could be 0 ~ 7.
  *
  * @return     None
  *
  * @details    Clear DSQ_SYNC bit of USB_CFGx register to specify the DATA0 PID for the following IN token transaction.
  *             Base on this setting, hardware will toggle PID between DATA0 and DATA1 automatically for IN token transactions.
  *
  */
#define USBD_SET_DATA0(ep)          (*((__IO uint32_t *) ((uint32_t)&USBD->EP[0].CFG + (uint32_t)((ep) << 4))) &= (~USBD_CFG_DSQSYNC_Msk))

/**
  * @brief      Set USB payload size (IN data)
  *
  * @param[in]  ep The USB endpoint ID. M451 Series supports 8 hardware endpoint ID. This parameter could be 0 ~ 7.
  *
  * @param[in]  size The transfer length.
  *
  * @return     None
  *
  * @details    This macro will write the transfer length to USB_MXPLDx register for IN data transaction.
  *
  */
#define USBD_SET_PAYLOAD_LEN(ep, size)  (*((__IO uint32_t *) ((uint32_t)&USBD->EP[0].MXPLD + (uint32_t)((ep) << 4))) = (size))

/**
  * @brief      Get USB payload size (OUT data)
  *
  * @param[in]  ep The USB endpoint ID. M451 Series supports 8 endpoint ID. This parameter could be 0 ~ 7.
  *
  * @return     The value of USB_MXPLDx register.
  *
  * @details    Get the data length of OUT data transaction by reading USB_MXPLDx register.
  *
  */
#define USBD_GET_PAYLOAD_LEN(ep)        ((uint32_t)*((__IO uint32_t *) ((uint32_t)&USBD->EP[0].MXPLD + (uint32_t)((ep) << 4))))

/**
  * @brief      Configure endpoint
  *
  * @param[in]  ep The USB endpoint ID. M451 Series supports 8 hardware endpoint ID. This parameter could be 0 ~ 7.
  *
  * @param[in]  config The USB configuration.
  *
  * @return     None
  *
  * @details    This macro will write config parameter to USB_CFGx register of specified endpoint ID.
  *
  */
#define USBD_CONFIG_EP(ep, config)      (*((__IO uint32_t *) ((uint32_t)&USBD->EP[0].CFG + (uint32_t)((ep) << 4))) = (config))

/**
  * @brief      Set USB endpoint buffer
  *
  * @param[in]  ep The USB endpoint ID. M451 Series supports 8 hardware endpoint ID. This parameter could be 0 ~ 7.
  *
  * @param[in]  offset The SRAM offset.
  *
  * @return     None
  *
  * @details    This macro will set the SRAM offset for the specified endpoint ID.
  *
  */
#define USBD_SET_EP_BUF_ADDR(ep, offset)    (*((__IO uint32_t *) ((uint32_t)&USBD->EP[0].BUFSEG + (uint32_t)((ep) << 4))) = (offset))

/**
  * @brief      Get the offset of the specified USB endpoint buffer
  *
  * @param[in]  ep The USB endpoint ID. M451 Series supports 8 hardware endpoint ID. This parameter could be 0 ~ 7.
  *
  * @return     The offset of the specified endpoint buffer.
  *
  * @details    This macro will return the SRAM offset of the specified endpoint ID.
  *
  */
#define USBD_GET_EP_BUF_ADDR(ep)        ((uint32_t)*((__IO uint32_t *) ((uint32_t)&USBD->EP[0].BUFSEG + (uint32_t)((ep) << 4))))

/**
  * @brief       Set USB endpoint stall state
  *
  * @param[in]   ep  The USB endpoint ID. M451 Series supports 8 hardware endpoint ID. This parameter could be 0 ~ 7.
  *
  * @return      None
  *
  * @details     Set USB endpoint stall state for the specified endpoint ID. Endpoint will respond STALL token automatically.
  *
  */
#define USBD_SET_EP_STALL(ep)        (*((__IO uint32_t *) ((uint32_t)&USBD->EP[0].CFGP + (uint32_t)((ep) << 4))) |= USBD_CFGP_SSTALL_Msk)

/**
  * @brief       Clear USB endpoint stall state
  *
  * @param[in]   ep  The USB endpoint ID. M451 Series supports 8 hardware endpoint ID. This parameter could be 0 ~ 7.
  *
  * @return      None
  *
  * @details     Clear USB endpoint stall state for the specified endpoint ID. Endpoint will respond ACK/NAK token.
  */
#define USBD_CLR_EP_STALL(ep)        (*((__IO uint32_t *) ((uint32_t)&USBD->EP[0].CFGP + (uint32_t)((ep) << 4))) &= ~USBD_CFGP_SSTALL_Msk)

/**
  * @brief       Get USB endpoint stall state
  *
  * @param[in]   ep  The USB endpoint ID. M451 Series supports 8 hardware endpoint ID. This parameter could be 0 ~ 7.
  *
  * @retval      0      USB endpoint is not stalled.
  * @retval      Others USB endpoint is stalled.
  *
  * @details     Get USB endpoint stall state of the specified endpoint ID.
  *
  */
#define USBD_GET_EP_STALL(ep)        (*((__IO uint32_t *) ((uint32_t)&USBD->EP[0].CFGP + (uint32_t)((ep) << 4))) & USBD_CFGP_SSTALL_Msk)

/**
  * @brief      To support byte access between USB SRAM and system SRAM
  *
  * @param[in]  dest Destination pointer.
  *
  * @param[in]  src  Source pointer.
  *
  * @param[in]  size Byte count.
  *
  * @return     None
  *
  * @details    This function will copy the number of data specified by size and src parameters to the address specified by dest parameter.
  *
  */
static __INLINE void USBD_MemCopy(uint8_t *dest, uint8_t *src, int32_t size)
{
    while(size--) *dest++ = *src++;
}


/**
  * @brief       Set USB endpoint stall state
  *
  * @param[in]   epnum  USB endpoint number
  *
  * @return      None
  *
  * @details     Set USB endpoint stall state. Endpoint will respond STALL token automatically.
  *
  */
static __INLINE void USBD_SetStall(uint8_t epnum)
{
    uint32_t u32CfgAddr;
    uint32_t u32Cfg;
    int i;

    for(i = 0; i < USBD_MAX_EP; i++)
    {
        u32CfgAddr = (uint32_t)(i << 4) + (uint32_t)&USBD->EP[0].CFG; /* USBD_CFG0 */
        u32Cfg = *((__IO uint32_t *)(u32CfgAddr));

        if((u32Cfg & 0xf) == epnum)
        {
            u32CfgAddr = (uint32_t)(i << 4) + (uint32_t)&USBD->EP[0].CFGP; /* USBD_CFGP0 */
            u32Cfg = *((__IO uint32_t *)(u32CfgAddr));

            *((__IO uint32_t *)(u32CfgAddr)) = (u32Cfg | USBD_CFGP_SSTALL);
            break;
        }
    }
}

/**
  * @brief       Clear USB endpoint stall state
  *
  * @param[in]   epnum  USB endpoint number
  *
  * @return      None
  *
  * @details     Clear USB endpoint stall state. Endpoint will respond ACK/NAK token.
  */
static __INLINE void USBD_ClearStall(uint8_t epnum)
{
    uint32_t u32CfgAddr;
    uint32_t u32Cfg;
    int i;

    for(i = 0; i < USBD_MAX_EP; i++)
    {
        u32CfgAddr = (uint32_t)(i << 4) + (uint32_t)&USBD->EP[0].CFG; /* USBD_CFG0 */
        u32Cfg = *((__IO uint32_t *)(u32CfgAddr));

        if((u32Cfg & 0xf) == epnum)
        {
            u32CfgAddr = (uint32_t)(i << 4) + (uint32_t)&USBD->EP[0].CFGP; /* USBD_CFGP0 */
            u32Cfg = *((__IO uint32_t *)(u32CfgAddr));

            *((__IO uint32_t *)(u32CfgAddr)) = (u32Cfg & ~USBD_CFGP_SSTALL);
            break;
        }
    }
}

/**
  * @brief       Get USB endpoint stall state
  *
  * @param[in]   epnum  USB endpoint number
  *
  * @retval      0      USB endpoint is not stalled.
  * @retval      Others USB endpoint is stalled.
  *
  * @details     Get USB endpoint stall state.
  *
  */
static __INLINE uint32_t USBD_GetStall(uint8_t epnum)
{
    uint32_t u32CfgAddr;
    uint32_t u32Cfg;
    int i;

    for(i = 0; i < USBD_MAX_EP; i++)
    {
        u32CfgAddr = (uint32_t)(i << 4) + (uint32_t)&USBD->EP[0].CFG; /* USBD_CFG0 */
        u32Cfg = *((__IO uint32_t *)(u32CfgAddr));

        if((u32Cfg & 0xf) == epnum)
        {
            u32CfgAddr = (uint32_t)(i << 4) + (uint32_t)&USBD->EP[0].CFGP; /* USBD_CFGP0 */
            break;
        }
    }

    return ((*((__IO uint32_t *)(u32CfgAddr))) & USBD_CFGP_SSTALL);
}


extern volatile uint8_t g_usbd_RemoteWakeupEn;


typedef void (*VENDOR_REQ)(void);           /*!< Functional pointer type definition for Vendor class */
typedef void (*CLASS_REQ)(void);            /*!< Functional pointer type declaration for USB class request callback handler */
typedef void (*SET_INTERFACE_REQ)(void);    /*!< Functional pointer type declaration for USB set interface request callback handler */
typedef void (*SET_CONFIG_CB)(void);       /*!< Functional pointer type declaration for USB set configuration request callback handler */


/*--------------------------------------------------------------------*/
void USBD_Open(const S_USBD_INFO_T *param, CLASS_REQ pfnClassReq, SET_INTERFACE_REQ pfnSetInterface);
void USBD_Start(void);
void USBD_GetSetupPacket(uint8_t *buf);
void USBD_ProcessSetupPacket(void);
void USBD_StandardRequest(void);
void USBD_PrepareCtrlIn(uint8_t *pu8Buf, uint32_t u32Size);
void USBD_CtrlIn(void);
void USBD_PrepareCtrlOut(uint8_t *pu8Buf, uint32_t u32Size);
void USBD_CtrlOut(void);
void USBD_SwReset(void);
void USBD_SetVendorRequest(VENDOR_REQ pfnVendorReq);
void USBD_SetConfigCallback(SET_CONFIG_CB pfnSetConfigCallback);
void USBD_LockEpStall(uint32_t u32EpBitmap);

/*@}*/ /* end of group USBD_EXPORTED_FUNCTIONS */

/*@}*/ /* end of group USBD_Driver */

/*@}*/ /* end of group Standard_Driver */

#ifdef __cplusplus
}
#endif

#endif //__USBD_H__

/*** (C) COPYRIGHT 2014~2015 Nuvoton Technology Corp. ***/
