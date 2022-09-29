/**************************************************************************//**
 * @file     crypto_reg.h
 * @version  V1.00
 * @brief    CRYPTO register definition header file
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
#ifndef __CRYPTO_REG_H__
#define __CRYPTO_REG_H__

#if defined ( __CC_ARM   )
    #pragma anon_unions
#endif

/**
   @addtogroup REGISTER Control Register
   @{
*/

/**
    @addtogroup CRPT Cryptographic Accelerator (CRPT)
    Memory Mapped Structure for Cryptographic Accelerator
@{ */

typedef struct
{


    /**
     * @var CRPT_T::INTEN
     * Offset: 0x00  Crypto Interrupt Enable Control Register
     * ---------------------------------------------------------------------------------------------------
     * |Bits    |Field     |Descriptions
     * | :----: | :----:   | :---- |
     * |[0]     |AESIEN    |AES Interrupt Enable Bit
     * |        |          |0 = AES interrupt Disabled.
     * |        |          |1 = AES interrupt Enabled.
     * |        |          |In DMA mode, an interrupt will be triggered when amount of data set in AES_DMA_CNT is fed into the AES engine.
     * |        |          |In Non-DMA mode, an interrupt will be triggered when the AES engine finishes the operation.
     * |[1]     |AESEIEN   |AES Error Flag Enable Bit
     * |        |          |0 = AES error interrupt flag Disabled.
     * |        |          |1 = AES error interrupt flag Enabled.
     * |[16]    |PRNGIEN   |PRNG Interrupt Enable Bit
     * |        |          |0 = PRNG interrupt Disabled.
     * |        |          |1 = PRNG interrupt Enabled.
     * @var CRPT_T::INTSTS
     * Offset: 0x04  Crypto Interrupt Flag
     * ---------------------------------------------------------------------------------------------------
     * |Bits    |Field     |Descriptions
     * | :----: | :----:   | :---- |
     * |[0]     |AESIF     |AES Finish Interrupt Flag
     * |        |          |This bit is cleared by writing 1, and it has no effect by writing 0.
     * |        |          |0 = No AES interrupt.
     * |        |          |1 = AES encryption/decryption done interrupt.
     * |[1]     |AESEIF    |AES Error Flag
     * |        |          |This bit is cleared by writing 1, and it has no effect by writing 0.
     * |        |          |0 = No AES error.
     * |        |          |1 = AES encryption/decryption error interrupt.
     * |[16]    |PRNGIF    |PRNG Finish Interrupt Flag
     * |        |          |This bit is cleared by writing 1, and it has no effect by writing 0.
     * |        |          |0 = No PRNG interrupt.
     * |        |          |1 = PRNG key generation done interrupt.
     * @var CRPT_T::PRNG_CTL
     * Offset: 0x08  PRNG Control Register
     * ---------------------------------------------------------------------------------------------------
     * |Bits    |Field     |Descriptions
     * | :----: | :----:   | :---- |
     * |[0]     |START     |Start PRNG Engine
     * |        |          |0 = Stop PRNG engine.
     * |        |          |1 = Generate new key and store the new key to register CRPT_PRNG_KEYx, which will be cleared when the new key is generated.
     * |[1]     |SEEDRLD   |Reload New Seed for PRNG Engine
     * |        |          |0 = Generating key based on the current seed.
     * |        |          |1 = Reload new seed.
     * |[3:2]   |KEYSZ     |PRNG Generate Key Size
     * |        |          |00 = 64 bits.
     * |        |          |01 = 128 bits.
     * |        |          |10 = 192 bits.
     * |        |          |11 = 256 bits.
     * |[8]     |BUSY      |PRNG Busy (Read Only)
     * |        |          |0 = PRNG engine is idle.
     * |        |          |1 = Indicate that the PRNG engine is generating CRPT_PRNG_KEYx.
     * @var CRPT_T::PRNG_SEED
     * Offset: 0x0C  Seed for PRNG
     * ---------------------------------------------------------------------------------------------------
     * |Bits    |Field     |Descriptions
     * | :----: | :----:   | :---- |
     * |[31:0]  |SEED      |Seed for PRNG (Write Only)
     * |        |          |The bits store the seed for PRNG engine.
     * @var CRPT_T::PRNG_KEY[8]
     * Offset: 0x10 ~ 0x2C  PRNG Generated Key0 ~ Key7
     * ---------------------------------------------------------------------------------------------------
     * |Bits    |Field     |Descriptions
     * | :----: | :----:   | :---- |
     * |[31:0]  |KEY       |Store PRNG Generated Key (Read Only)
     * |        |          |The bits store the key that is generated by PRNG.
     * @var CRPT_T::AES_FDBCK[4]
     * Offset: 0x50 ~ 0x5C  AES Engine Output Feedback Data After Cryptographic Operation
     * ---------------------------------------------------------------------------------------------------
     * |Bits    |Field     |Descriptions
     * | :----: | :----:   | :---- |
     * |[31:0]  |FDBCK     |AES Feedback Information
     * |        |          |The feedback value is 128 bits in size.
     * |        |          |The AES engine uses the data from CRPT_AES_FDBCKx as the data inputted to CRPT_AES_IVx for the next block in DMA cascade mode.
     * |        |          |The AES engine outputs feedback information for IV in the next block's operation
     * |        |          |Software can store that feedback value temporarily
     * |        |          |After switching back, fill the stored feedback value to CRPT_AES_IVx in the same operation, and then continue the operation with the original setting.
     * @var CRPT_T::AES_CTL
     * Offset: 0x100  AES Control Register
     * ---------------------------------------------------------------------------------------------------
     * |Bits    |Field     |Descriptions
     * | :----: | :----:   | :---- |
     * |[0]     |START     |AES Engine Start
     * |        |          |0 = No effect.
     * |        |          |1 = Start AES engine. BUSY flag will be set.
     * |        |          |Note: This bit is always 0 when it's read back.
     * |[1]     |STOP      |AES Engine Stop
     * |        |          |0 = No effect.
     * |        |          |1 = Stop AES engine.
     * |        |          |Note: This bit is always 0 when it's read back.
     * |[3:2]   |KEYSZ     |AES Key Size
     * |        |          |This bit defines three different key size for AES operation.
     * |        |          |2'b00 = 128 bits key.
     * |        |          |2'b01 = 192 bits key.
     * |        |          |2'b10 = 256 bits key.
     * |        |          |2'b11 = Reserved.
     * |        |          |If the AES accelerator is operating and the corresponding flag BUSY is 1, updating this register has no effect.
     * |[5]     |DMALAST   |AES Last Block
     * |        |          |In DMA mode, this bit must be set as beginning the last DMA cascade round.
     * |        |          |In Non-DMA mode, this bit must be set when feeding in the last block of data in ECB, CBC, CTR, OFB, and CFB mode, and feeding in the (last-1) block of data at CBC-CS1, CBC-CS2, and CBC-CS3 mode.
     * |        |          |This bit is always 0 when it's read back. Must be written again once START is triggered.
     * |[6]     |DMACSCAD  |AES Engine DMA with Cascade Mode
     * |        |          |0 = DMA cascade function Disabled.
     * |        |          |1 = In DMA cascade mode, software can update DMA source address register, destination address register, and byte count register during a cascade operation, without finishing the accelerator operation.
     * |[7]     |DMAEN     |AES Engine DMA Enable Bit
     * |        |          |0 = AES DMA engine Disabled.
     * |        |          |The AES engine operates in Non-DMA mode. The data need to be written in CRPT_AES_DATIN.
     * |        |          |1 = AES_DMA engine Enabled.
     * |        |          |The AES engine operates in DMA mode, and data movement from/to the engine is done by DMA logic.
     * |[15:8]  |OPMODE    |AES Engine Operation Modes
     * |        |          |0x00 = ECB (Electronic Codebook Mode)  0x01 = CBC (Cipher Block Chaining Mode).
     * |        |          |0x02 = CFB (Cipher Feedback Mode).
     * |        |          |0x03 = OFB (Output Feedback Mode).
     * |        |          |0x04 = CTR (Counter Mode).
     * |        |          |0x10 = CBC-CS1 (CBC Ciphertext-Stealing 1 Mode).
     * |        |          |0x11 = CBC-CS2 (CBC Ciphertext-Stealing 2 Mode).
     * |        |          |0x12 = CBC-CS3 (CBC Ciphertext-Stealing 3 Mode).
     * |[16]    |ENCRYPTO  |AES Encryption/Decryption
     * |        |          |0 = AES engine executes decryption operation.
     * |        |          |1 = AES engine executes encryption operation.
     * |[22]    |OUTSWAP   |AES Engine Output Data Swap
     * |        |          |0 = Keep the original order.
     * |        |          |1 = The order that CPU outputs data from the accelerator will be changed from {byte3, byte2, byte1, byte0} to {byte0, byte1, byte2, byte3}.
     * |[23]    |INSWAP    |AES Engine Input Data Swap
     * |        |          |0 = Keep the original order.
     * |        |          |1 = The order that CPU feeds data to the accelerator will be changed from {byte3, byte2, byte1, byte0} to {byte0, byte1, byte2, byte3}.
     * |[30:26] |KEYUNPRT  |Unprotect Key
     * |        |          |Writing 0 to CRPT_AES_CTL[31] and "10110" to CRPT_AES_CTL[30:26] is to unprotect the AES key.
     * |        |          |The KEYUNPRT can be read and written
     * |        |          |When it is written as the AES engine is operating, BUSY flag is 1, there would be no effect on KEYUNPRT.
     * |[31]    |KEYPRT    |Protect Key
     * |        |          |Read as a flag to reflect KEYPRT.
     * |        |          |0 = No effect.
     * |        |          |1 = Protect the content of the AES key from reading
     * |        |          |The return value for reading CRPT_AES_KEYx is not the content of the registers CRPT_AES_KEYx
     * |        |          |Once it is set, it can be cleared by asserting KEYUNPRT
     * |        |          |And the key content would be cleared as well.
     * @var CRPT_T::AES_STS
     * Offset: 0x104  AES Engine Flag
     * ---------------------------------------------------------------------------------------------------
     * |Bits    |Field     |Descriptions
     * | :----: | :----:   | :---- |
     * |[0]     |BUSY      |AES Engine Busy
     * |        |          |0 = The AES engine is idle or finished.
     * |        |          |1 = The AES engine is under processing.
     * |[8]     |INBUFEMPTY|AES Input Buffer Empty
     * |        |          |0 = There are some data in input buffer waiting for the AES engine to process.
     * |        |          |1 = AES input buffer is empty
     * |        |          |Software needs to feed data to the AES engine
     * |        |          |Otherwise, the AES engine will be pending to wait for input data.
     * |[9]     |INBUFFULL |AES Input Buffer Full Flag
     * |        |          |0 = AES input buffer is not full. Software can feed the data into the AES engine.
     * |        |          |1 = AES input buffer is full
     * |        |          |Software cannot feed data to the AES engine
     * |        |          |Otherwise, the flag INBUFERR will be set to 1.
     * |[10]    |INBUFERR  |AES Input Buffer Error Flag
     * |        |          |0 = No error.
     * |        |          |1 = Error happens during feeding data to the AES engine.
     * |[12]    |CNTERR    |CRPT_AES_CNT Setting Error
     * |        |          |0 = No error in CRPT_AES_CNT setting.
     * |        |          |1 = CRPT_AES_CNT is 0 if DMAEN (CRPT_AES_CTL[7]) is enabled.
     * |[16]    |OUTBUFEMPTY|AES Out Buffer Empty
     * |        |          |0 = AES output buffer is not empty. There are some valid data kept in output buffer.
     * |        |          |1 = AES output buffer is empty
     * |        |          |Software cannot get data from CRPT_AES_DATOUT
     * |        |          |Otherwise, the flag OUTBUFERR will be set to 1 since the output buffer is empty.
     * |[17]    |OUTBUFFULL|AES Out Buffer Full Flag
     * |        |          |0 = AES output buffer is not full.
     * |        |          |1 = AES output buffer is full, and software needs to get data from CRPT_AES_DATOUT
     * |        |          |Otherwise, the AES engine will be pending since the output buffer is full.
     * |[18]    |OUTBUFERR |AES Out Buffer Error Flag
     * |        |          |0 = No error.
     * |        |          |1 = Error happens during getting the result from AES engine.
     * |[20]    |BUSERR    |AES DMA Access Bus Error Flag
     * |        |          |0 = No error.
     * |        |          |1 = Bus error will stop DMA operation and AES engine.
     * @var CRPT_T::AES_DATIN
     * Offset: 0x108  AES Engine Data Input Port Register
     * ---------------------------------------------------------------------------------------------------
     * |Bits    |Field     |Descriptions
     * | :----: | :----:   | :---- |
     * |[31:0]  |DATIN     |AES Engine Input Port
     * |        |          |CPU feeds data to AES engine through this port by checking CRPT_AES_STS. Feed data as INBUFFULL is 0.
     * @var CRPT_T::AES_DATOUT
     * Offset: 0x10C  AES Engine Data Output Port Register
     * ---------------------------------------------------------------------------------------------------
     * |Bits    |Field     |Descriptions
     * | :----: | :----:   | :---- |
     * |[31:0]  |DATOUT    |AES Engine Output Port
     * |        |          |CPU gets results from the AES engine through this port by checking CRPT_AES_STS
     * |        |          |Get data as OUTBUFEMPTY is 0.
     * @var CRPT_T::AES_KEY[8]
     * Offset: 0x110 ~ 0x12C  AES Key Word 0 ~ 7 Register
     * ---------------------------------------------------------------------------------------------------
     * |Bits    |Field     |Descriptions
     * | :----: | :----:   | :---- |
     * |[31:0]  |KEY       |CRPT_AESn_KEYx
     * |        |          |The KEY keeps the security key for AES operation.
     * |        |          |x = 0, 1..7.
     * |        |          |The security key for AES accelerator can be 128, 192, or 256 bits and four, six, or eight 32-bit registers are to store each security key
     * |        |          |{CRPT_AES_KEY3, CRPT_AES_KEY2, CRPT_AES_KEY1, CRPT_AES_KEY0} stores the 128-bit security key for AES operation
     * |        |          |{CRPT_AES_KEY5, CRPT_AES_KEY4, CRPT_AES_KEY3, CRPT_AES_KEY2, CRPT_AES_KEY1, CRPT_AES_KEY0} stores the 192-bit security key for AES operation
     * |        |          |{CRPT_AES_KEY7, CRPT_AES_KEY6, CRPT_AES_KEY5, CRPT_AES_KEY4, CRPT_AES_KEY3, CRPT_AES_KEY2, CRPT_AES_KEY1, CRPT_AES_KEY0} stores the 256-bit security key for AES operation.
     * @var CRPT_T::AES_IV[4]
     * Offset: 0x130 ~ 0x13C  AES Initial Vector Word 0 ~ 3 Register
     * ---------------------------------------------------------------------------------------------------
     * |Bits    |Field     |Descriptions
     * | :----: | :----:   | :---- |
     * |[31:0]  |IV        |AES Initial Vectors
     * |        |          |x = 0, 1..3.
     * |        |          |Four initial vectors (CRPT_AES_IV0, CRPT_AES_IV1, CRPT_AES_IV2, and CRPT_AES_IV3) are for AES operating in CBC, CFB, and OFB mode
     * |        |          |Four registers (CRPT_AES_IV0, CRPT_AES_IV1, CRPT_AES_IV2, and CRPT_AES_IV3) act as Nonce counter when the AES engine is operating in CTR mode.
     * @var CRPT_T::AES_SADDR
     * Offset: 0x140  AES DMA Source Address Register
     * ---------------------------------------------------------------------------------------------------
     * |Bits    |Field     |Descriptions
     * | :----: | :----:   | :---- |
     * |[31:0]  |SADDR     |AES DMA Source Address
     * |        |          |The AES accelerator supports DMA function to transfer the plain text between SRAM memory space and embedded FIFO
     * |        |          |The SADDR keeps the source address of the data buffer where the source text is stored
     * |        |          |Based on the source address, the AES accelerator can read the plain text (encryption) / cipher text (descryption) from SRAM memory space and do AES operation
     * |        |          |The start of source address should be located at word boundary
     * |        |          |In other words, bit 1 and 0 of SADDR are ignored.
     * |        |          |SADDR can be read and written
     * |        |          |Writing to SADDR while the AES accelerator is operating doesn't affect the current AES operation
     * |        |          |But the value of SADDR will be updated later on
     * |        |          |Consequently, software can prepare the DMA source address for the next AES operation.
     * |        |          |In DMA mode, software can update the next CRPT_AES_SADDR before triggering START.
     * |        |          |The value of CRPT_AES_SADDR and CRPT_AES_DADDR can be the same.
     * @var CRPT_T::AES_DADDR
     * Offset: 0x144  AES DMA Destination Address Register
     * ---------------------------------------------------------------------------------------------------
     * |Bits    |Field     |Descriptions
     * | :----: | :----:   | :---- |
     * |[31:0]  |DADDR     |AES DMA Destination Address
     * |        |          |The AES accelerator supports DMA function to transfer the cipher text between SRAM memory space and embedded FIFO
     * |        |          |The DADDR keeps the destination address of the data buffer where the engine output's text will be stored
     * |        |          |Based on the destination address, the AES accelerator can write the cipher text (encryption) / plain text (decryption) back to SRAM memory space after the AES operation is finished
     * |        |          |The start of destination address should be located at word boundary
     * |        |          |In other words, bit 1 and 0 of DADDR are ignored.
     * |        |          |DADDR can be read and written
     * |        |          |Writing to DADDR while the AES accelerator is operating doesn't affect the current AES operation
     * |        |          |But the value of DADDR will be updated later on
     * |        |          |Consequently, software can prepare the destination address for the next AES operation.
     * |        |          |In DMA mode, software can update the next CRPT_AES_DADDR before triggering START.
     * |        |          |The value of CRPT_AES_SADDR and CRPT_AES_DADDR can be the same.
     * @var CRPT_T::AES_CNT
     * Offset: 0x148  AES Byte Count Register
     * ---------------------------------------------------------------------------------------------------
     * |Bits    |Field     |Descriptions
     * | :----: | :----:   | :---- |
     * |[31:0]  |CNT       |AES Byte Count
     * |        |          |The CRPT_AES_CNT keeps the byte count of source text that is for the AES engine operating in DMA mode
     * |        |          |The CRPT_AES_CNT is 32-bit and the maximum of byte count is 4G bytes.
     * |        |          |CRPT_AESn_CNT can be read and written
     * |        |          |Writing to CRPT_AES_CNT while the AES accelerator is operating doesn't affect the current AES operation
     * |        |          |But the value of CRPT_AESn_CNT will be updated later on
     * |        |          |Consequently, software can prepare the byte count of data for the next AES operation.
     * |        |          |According to CBC-CS1, CBC-CS2, and CBC-CS3 standard, the count of operation data must be more than 16 bytes
     * |        |          |Operations that are qual or less than one block will output unexpected result.
     * |        |          |In Non-DMA ECB, CBC, CFB, OFB, and CTR mode, CRPT_AES_CNT must be set as byte count for the last block of data before feeding in the last block of data
     * |        |          |In Non-DMA CBC-CS1, CBC-CS2, and CBC-CS3 mode, CRPT_AES_CNT must be set as byte count for the last two blocks of data before feeding in the last two blocks of data.
     */
    __IO uint32_t INTEN;                 /*!< [0x0000] Crypto Interrupt Enable Control Register                         */
    __IO uint32_t INTSTS;                /*!< [0x0004] Crypto Interrupt Flag                                            */
    __IO uint32_t PRNG_CTL;              /*!< [0x0008] PRNG Control Register                                            */
    __O  uint32_t PRNG_SEED;             /*!< [0x000c] Seed for PRNG                                                    */
    __I  uint32_t PRNG_KEY[8];           /*!< [0x0010 ~ 0x002c] PRNG Generated Key0 ~ 7                                 */
    __I  uint32_t RESERVE0[8];
    __I  uint32_t AES_FDBCK[4];          /*!< [0x0050 ~ 0x005c] AES Engine Output Feedback Data After Cryptographic Operation*/
    __I  uint32_t RESERVE1[40];
    __IO uint32_t AES_CTL;               /*!< [0x0100] AES Control Register                                             */
    __I  uint32_t AES_STS;               /*!< [0x0104] AES Engine Flag                                                  */
    __IO uint32_t AES_DATIN;             /*!< [0x0108] AES Engine Data Input Port Register                              */
    __I  uint32_t AES_DATOUT;            /*!< [0x010c] AES Engine Data Output Port Register                             */
    __IO uint32_t AES_KEY[8];            /*!< [0x0110 ~ 0x012c] AES Key Word 0 ~ 7 Register                             */
    __IO uint32_t AES_IV[4];             /*!< [0x0130 ~ 0x013c] AES Initial Vector Word 0 ~3 Register                   */
    __IO uint32_t AES_SADDR;             /*!< [0x0140] AES DMA Source Address Register                                  */
    __IO uint32_t AES_DADDR;             /*!< [0x0144] AES DMA Destination Address Register                             */
    __IO uint32_t AES_CNT;               /*!< [0x0148] AES Byte Count Register                              */

} CRPT_T;

/**
    @addtogroup CRPT_CONST CRYPTO Bit Field Definition
    Constant Definitions for CRYPTO Controller
@{ */

#define CRPT_INTEN_AESIEN_Pos          (0)                                               /*!< CRPT_T::INTEN: AESIEN Position       */
#define CRPT_INTEN_AESIEN_Msk          (0x1ul << CRPT_INTEN_AESIEN_Pos)                /*!< CRPT_T::INTEN: AESIEN Mask           */

#define CRPT_INTEN_AESEIEN_Pos         (1)                                               /*!< CRPT_T::INTEN: AESEIEN Position      */
#define CRPT_INTEN_AESEIEN_Msk         (0x1ul << CRPT_INTEN_AESEIEN_Pos)               /*!< CRPT_T::INTEN: AESEIEN Mask          */

#define CRPT_INTEN_PRNGIEN_Pos         (16)                                              /*!< CRPT_T::INTEN: PRNGIEN Position      */
#define CRPT_INTEN_PRNGIEN_Msk         (0x1ul << CRPT_INTEN_PRNGIEN_Pos)               /*!< CRPT_T::INTEN: PRNGIEN Mask          */

#define CRPT_INTSTS_AESIF_Pos          (0)                                               /*!< CRPT_T::INTSTS: AESIF Position       */
#define CRPT_INTSTS_AESIF_Msk          (0x1ul << CRPT_INTSTS_AESIF_Pos)                /*!< CRPT_T::INTSTS: AESIF Mask           */

#define CRPT_INTSTS_AESEIF_Pos         (1)                                               /*!< CRPT_T::INTSTS: AESEIF Position      */
#define CRPT_INTSTS_AESEIF_Msk         (0x1ul << CRPT_INTSTS_AESEIF_Pos)               /*!< CRPT_T::INTSTS: AESEIF Mask          */

#define CRPT_INTSTS_PRNGIF_Pos         (16)                                              /*!< CRPT_T::INTSTS: PRNGIF Position      */
#define CRPT_INTSTS_PRNGIF_Msk         (0x1ul << CRPT_INTSTS_PRNGIF_Pos)               /*!< CRPT_T::INTSTS: PRNGIF Mask          */

#define CRPT_PRNG_CTL_START_Pos        (0)                                               /*!< CRPT_T::PRNG_CTL: START Position     */
#define CRPT_PRNG_CTL_START_Msk        (0x1ul << CRPT_PRNG_CTL_START_Pos)              /*!< CRPT_T::PRNG_CTL: START Mask         */

#define CRPT_PRNG_CTL_SEEDRLD_Pos      (1)                                               /*!< CRPT_T::PRNG_CTL: SEEDRLD Position   */
#define CRPT_PRNG_CTL_SEEDRLD_Msk      (0x1ul << CRPT_PRNG_CTL_SEEDRLD_Pos)            /*!< CRPT_T::PRNG_CTL: SEEDRLD Mask       */

#define CRPT_PRNG_CTL_KEYSZ_Pos        (2)                                               /*!< CRPT_T::PRNG_CTL: KEYSZ Position     */
#define CRPT_PRNG_CTL_KEYSZ_Msk        (0x3ul << CRPT_PRNG_CTL_KEYSZ_Pos)              /*!< CRPT_T::PRNG_CTL: KEYSZ Mask         */

#define CRPT_PRNG_CTL_BUSY_Pos         (8)                                               /*!< CRPT_T::PRNG_CTL: BUSY Position      */
#define CRPT_PRNG_CTL_BUSY_Msk         (0x1ul << CRPT_PRNG_CTL_BUSY_Pos)               /*!< CRPT_T::PRNG_CTL: BUSY Mask          */

#define CRPT_PRNG_SEED_SEED_Pos        (0)                                               /*!< CRPT_T::PRNG_SEED: SEED Position     */
#define CRPT_PRNG_SEED_SEED_Msk        (0xfffffffful << CRPT_PRNG_SEED_SEED_Pos)       /*!< CRPT_T::PRNG_SEED: SEED Mask         */

#define CRPT_PRNG_KEYx_KEY_Pos         (0)                                               /*!< CRPT_T::PRNG_KEY[8]:  KEY Position   */
#define CRPT_PRNG_KEYx_KEY_Msk         (0xfffffffful << CRPT_PRNG_KEYx_KEY_Pos)        /*!< CRPT_T::PRNG_KEY:[8]  KEY Mask       */

#define CRPT_AES_FDBCKx_FDBCK_Pos      (0)                                               /*!< CRPT_T::AES_FDBCK[4]: FDBCK Position */
#define CRPT_AES_FDBCKx_FDBCK_Msk      (0xfffffffful << CRPT_AES_FDBCKx_FDBCK_Pos)     /*!< CRPT_T::AES_FDBCK[4]: FDBCK Mask     */

#define CRPT_AES_CTL_START_Pos         (0)                                               /*!< CRPT_T::AES_CTL: START Position      */
#define CRPT_AES_CTL_START_Msk         (0x1ul << CRPT_AES_CTL_START_Pos)               /*!< CRPT_T::AES_CTL: START Mask          */

#define CRPT_AES_CTL_STOP_Pos          (1)                                               /*!< CRPT_T::AES_CTL: STOP Position       */
#define CRPT_AES_CTL_STOP_Msk          (0x1ul << CRPT_AES_CTL_STOP_Pos)                /*!< CRPT_T::AES_CTL: STOP Mask           */

#define CRPT_AES_CTL_KEYSZ_Pos         (2)                                               /*!< CRPT_T::AES_CTL: KEYSZ Position      */
#define CRPT_AES_CTL_KEYSZ_Msk         (0x3ul << CRPT_AES_CTL_KEYSZ_Pos)               /*!< CRPT_T::AES_CTL: KEYSZ Mask          */

#define CRPT_AES_CTL_DMALAST_Pos       (5)                                               /*!< CRPT_T::AES_CTL: DMALAST Position    */
#define CRPT_AES_CTL_DMALAST_Msk       (0x1ul << CRPT_AES_CTL_DMALAST_Pos)             /*!< CRPT_T::AES_CTL: DMALAST Mask        */

#define CRPT_AES_CTL_DMACSCAD_Pos      (6)                                               /*!< CRPT_T::AES_CTL: DMACSCAD Position   */
#define CRPT_AES_CTL_DMACSCAD_Msk      (0x1ul << CRPT_AES_CTL_DMACSCAD_Pos)            /*!< CRPT_T::AES_CTL: DMACSCAD Mask       */

#define CRPT_AES_CTL_DMAEN_Pos         (7)                                               /*!< CRPT_T::AES_CTL: DMAEN Position      */
#define CRPT_AES_CTL_DMAEN_Msk         (0x1ul << CRPT_AES_CTL_DMAEN_Pos)               /*!< CRPT_T::AES_CTL: DMAEN Mask          */

#define CRPT_AES_CTL_OPMODE_Pos        (8)                                               /*!< CRPT_T::AES_CTL: OPMODE Position     */
#define CRPT_AES_CTL_OPMODE_Msk        (0xfful << CRPT_AES_CTL_OPMODE_Pos)             /*!< CRPT_T::AES_CTL: OPMODE Mask         */

#define CRPT_AES_CTL_ENCRYPTO_Pos      (16)                                              /*!< CRPT_T::AES_CTL: ENCRYPTO Position   */
#define CRPT_AES_CTL_ENCRYPTO_Msk      (0x1ul << CRPT_AES_CTL_ENCRYPTO_Pos)            /*!< CRPT_T::AES_CTL: ENCRYPTO Mask       */

#define CRPT_AES_CTL_OUTSWAP_Pos       (22)                                              /*!< CRPT_T::AES_CTL: OUTSWAP Position    */
#define CRPT_AES_CTL_OUTSWAP_Msk       (0x1ul << CRPT_AES_CTL_OUTSWAP_Pos)             /*!< CRPT_T::AES_CTL: OUTSWAP Mask        */

#define CRPT_AES_CTL_INSWAP_Pos        (23)                                              /*!< CRPT_T::AES_CTL: INSWAP Position     */
#define CRPT_AES_CTL_INSWAP_Msk        (0x1ul << CRPT_AES_CTL_INSWAP_Pos)              /*!< CRPT_T::AES_CTL: INSWAP Mask         */

#define CRPT_AES_CTL_KEYUNPRT_Pos      (26)                                              /*!< CRPT_T::AES_CTL: KEYUNPRT Position   */
#define CRPT_AES_CTL_KEYUNPRT_Msk      (0x1ful << CRPT_AES_CTL_KEYUNPRT_Pos)           /*!< CRPT_T::AES_CTL: KEYUNPRT Mask       */

#define CRPT_AES_CTL_KEYPRT_Pos        (31)                                              /*!< CRPT_T::AES_CTL: KEYPRT Position     */
#define CRPT_AES_CTL_KEYPRT_Msk        (0x1ul << CRPT_AES_CTL_KEYPRT_Pos)              /*!< CRPT_T::AES_CTL: KEYPRT Mask         */

#define CRPT_AES_STS_BUSY_Pos          (0)                                               /*!< CRPT_T::AES_STS: BUSY Position       */
#define CRPT_AES_STS_BUSY_Msk          (0x1ul << CRPT_AES_STS_BUSY_Pos)                /*!< CRPT_T::AES_STS: BUSY Mask           */

#define CRPT_AES_STS_INBUFEMPTY_Pos    (8)                                               /*!< CRPT_T::AES_STS: INBUFEMPTY Position */
#define CRPT_AES_STS_INBUFEMPTY_Msk    (0x1ul << CRPT_AES_STS_INBUFEMPTY_Pos)          /*!< CRPT_T::AES_STS: INBUFEMPTY Mask     */

#define CRPT_AES_STS_INBUFFULL_Pos     (9)                                               /*!< CRPT_T::AES_STS: INBUFFULL Position  */
#define CRPT_AES_STS_INBUFFULL_Msk     (0x1ul << CRPT_AES_STS_INBUFFULL_Pos)           /*!< CRPT_T::AES_STS: INBUFFULL Mask      */

#define CRPT_AES_STS_INBUFERR_Pos      (10)                                              /*!< CRPT_T::AES_STS: INBUFERR Position   */
#define CRPT_AES_STS_INBUFERR_Msk      (0x1ul << CRPT_AES_STS_INBUFERR_Pos)            /*!< CRPT_T::AES_STS: INBUFERR Mask       */

#define CRPT_AES_STS_CNTERR_Pos        (12)                                              /*!< CRPT_T::AES_STS: CNTERR Position     */
#define CRPT_AES_STS_CNTERR_Msk        (0x1ul << CRPT_AES_STS_CNTERR_Pos)              /*!< CRPT_T::AES_STS: CNTERR Mask         */

#define CRPT_AES_STS_OUTBUFEMPTY_Pos   (16)                                              /*!< CRPT_T::AES_STS: OUTBUFEMPTY Position*/
#define CRPT_AES_STS_OUTBUFEMPTY_Msk   (0x1ul << CRPT_AES_STS_OUTBUFEMPTY_Pos)         /*!< CRPT_T::AES_STS: OUTBUFEMPTY Mask    */

#define CRPT_AES_STS_OUTBUFFULL_Pos    (17)                                              /*!< CRPT_T::AES_STS: OUTBUFFULL Position */
#define CRPT_AES_STS_OUTBUFFULL_Msk    (0x1ul << CRPT_AES_STS_OUTBUFFULL_Pos)          /*!< CRPT_T::AES_STS: OUTBUFFULL Mask     */

#define CRPT_AES_STS_OUTBUFERR_Pos     (18)                                              /*!< CRPT_T::AES_STS: OUTBUFERR Position  */
#define CRPT_AES_STS_OUTBUFERR_Msk     (0x1ul << CRPT_AES_STS_OUTBUFERR_Pos)           /*!< CRPT_T::AES_STS: OUTBUFERR Mask      */

#define CRPT_AES_STS_BUSERR_Pos        (20)                                              /*!< CRPT_T::AES_STS: BUSERR Position     */
#define CRPT_AES_STS_BUSERR_Msk        (0x1ul << CRPT_AES_STS_BUSERR_Pos)              /*!< CRPT_T::AES_STS: BUSERR Mask         */

#define CRPT_AES_DATIN_DATIN_Pos       (0)                                               /*!< CRPT_T::AES_DATIN: DATIN Position    */
#define CRPT_AES_DATIN_DATIN_Msk       (0xfffffffful << CRPT_AES_DATIN_DATIN_Pos)      /*!< CRPT_T::AES_DATIN: DATIN Mask        */

#define CRPT_AES_DATOUT_DATOUT_Pos     (0)                                               /*!< CRPT_T::AES_DATOUT: DATOUT Position  */
#define CRPT_AES_DATOUT_DATOUT_Msk     (0xfffffffful << CRPT_AES_DATOUT_DATOUT_Pos)    /*!< CRPT_T::AES_DATOUT: DATOUT Mask      */

#define CRPT_AES_KEYx_KEY_Pos          (0)                                               /*!< CRPT_T::AES_KEY[8]: KEY Position     */
#define CRPT_AES_KEYx_KEY_Msk          (0xfffffffful << CRPT_AES_KEYx_KEY_Pos)         /*!< CRPT_T::AES_KEY[8]: KEY Mask         */

#define CRPT_AES_IVx_IV_Pos            (0)                                               /*!< CRPT_T::AES_IV[4]: IV Position       */
#define CRPT_AES_IVx_IV_Msk            (0xfffffffful << CRPT_AES_IVx_IV_Pos)           /*!< CRPT_T::AES_IV[4]: IV Mask           */

#define CRPT_AES_SADDR_SADDR_Pos       (0)                                               /*!< CRPT_T::AES_SADDR: SADDR Position    */
#define CRPT_AES_SADDR_SADDR_Msk       (0xfffffffful << CRPT_AES_SADDR_SADDR_Pos)      /*!< CRPT_T::AES_SADDR: SADDR Mask        */

#define CRPT_AES_DADDR_DADDR_Pos       (0)                                               /*!< CRPT_T::AES_DADDR: DADDR Position    */
#define CRPT_AES_DADDR_DADDR_Msk       (0xfffffffful << CRPT_AES_DADDR_DADDR_Pos)      /*!< CRPT_T::AES_DADDR: DADDR Mask        */

#define CRPT_AES_CNT_CNT_Pos           (0)                                               /*!< CRPT_T::AES_CNT: CNT Position        */
#define CRPT_AES_CNT_CNT_Msk           (0xfffffffful << CRPT_AES_CNT_CNT_Pos)          /*!< CRPT_T::AES_CNT: CNT Mask            */

/**@}*/ /* CRPT_CONST CRYPTO */
/**@}*/ /* end of CRYPTO register group */
/**@}*/ /* end of REGISTER group */

#if defined ( __CC_ARM   )
    #pragma no_anon_unions
#endif


#endif /* __CRYPTO_REG_H__ */
