/*
// Copyright 2015 2016 Intel Corporation All Rights Reserved.
//
// The source code, information and material ("Material") contained herein is
// owned by Intel Corporation or its suppliers or licensors, and title
// to such Material remains with Intel Corporation or its suppliers or
// licensors. The Material contains proprietary information of Intel
// or its suppliers and licensors. The Material is protected by worldwide
// copyright laws and treaty provisions. No part of the Material may be used,
// copied, reproduced, modified, published, uploaded, posted, transmitted,
// distributed or disclosed in any way without Intel's prior express written
// permission. No license under any patent, copyright or other intellectual
// property rights in the Material is granted to or conferred upon you,
// either expressly, by implication, inducement, estoppel or otherwise.
// Any license under such intellectual property rights must be express and
// approved by Intel in writing.
//
// Unless otherwise agreed by Intel in writing,
// you may not remove or alter this notice or any other notice embedded in
// Materials by Intel or Intel's suppliers or licensors in any way.
*/

//   A simple example of performing filtering an image using a general integer rectangular kernel
// implemented with Intel IPP functions:
//     ippiFilterBorderGetSize
//     ippiFilterBorderInit_16s
//     ippiFilterBorder_8u_C1R


#include <stdio.h>
#include "ipp.h"

#define WIDTH  128  /* image width */
#define HEIGHT  64  /* image height */
const Ipp16s kernel[3*3] = {-1/8, -1/8, -1/8, -1/8, 16/8, -1/8, -1/8, -1/8, -1/8}; // Define high pass filter

/* Next two defines are created to simplify code reading and understanding */
#define EXIT_MAIN exitLine:                                  /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine; /* Go to Exit if IPP function returned status different from ippStsNoErr */

/* Results of ippMalloc() are not validated because Intel(R) Integrated Performance Primitives functions perform bad arguments check and will return an appropriate status  */

int main(void)
{
    IppStatus status = ippStsNoErr;
    Ipp8u* pSrc = NULL, *pDst = NULL;     /* Pointers to source/destination images */
    int srcStep = 0, dstStep = 0;         /* Steps, in bytes, through the source/destination images */
    IppiSize roiSize = { WIDTH, HEIGHT }; /* Size of source/destination ROI in pixels */
    IppiSize  kernelSize = { 3, 3 };
    Ipp8u *pBuffer = NULL;                /* Pointer to the work buffer */
    IppiFilterBorderSpec* pSpec = NULL;   /* context structure */
    int iTmpBufSize = 0, iSpecSize = 0;   /* Common work buffer size */
    IppiBorderType borderType = ippBorderRepl;
    Ipp8u borderValue = 0;
    int numChannels = 1;

    pSrc = ippiMalloc_8u_C1(roiSize.width, roiSize.height, &srcStep);
    pDst = ippiMalloc_8u_C1(roiSize.width, roiSize.height, &dstStep);

    check_sts( status = ippiFilterBorderGetSize(kernelSize, roiSize, ipp8u, ipp16s, numChannels, &iSpecSize, &iTmpBufSize) )

    pSpec = (IppiFilterBorderSpec *)ippsMalloc_8u(iSpecSize);
    pBuffer = ippsMalloc_8u(iTmpBufSize);

    check_sts( status = ippiFilterBorderInit_16s(kernel, kernelSize, 4, ipp8u, numChannels, ippRndNear, pSpec) )

    check_sts( status = ippiFilterBorder_8u_C1R(pSrc, srcStep, pDst, dstStep, roiSize, borderType, &borderValue, pSpec, pBuffer) )

EXIT_MAIN
    ippsFree(pBuffer);
    ippsFree(pSpec);
    ippiFree(pSrc);
    ippiFree(pDst);
    printf("Exit status %d (%s)\n", (int)status, ippGetStatusString(status));
    return (int)status;
}
