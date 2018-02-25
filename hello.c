


#include <stdio.h>
#include "ipp.h"

#define WIDTH  1920  /* image width */
#define HEIGHT  1080  /* image height */
const Ipp16s kernel[3*3] = {-1/8, -1/8, -1/8, -1/8, 16/8, -1/8, -1/8, -1/8, -1/8}; // Define high pass filter

/* Next two defines are created to simplify code reading and understanding */
#define EXIT_MAIN exitLine:                                  /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine; /* Go to Exit if IPP function returned status different from ippStsNoErr */

/* Results of ippMalloc() are not validated because Intel(R) Integrated Performance Primitives functions perform bad arguments check and will return an appropriate status  */

int main(void)
{
    IppStatus status = ippStsNoErr;
    Ipp32f *pIpp32fImage = NULL, *pFilteredImage = NULL;
    Ipp8u *pSrcImage = NULL, *pOutputImage = NULL;
    IppiSize  kernelSize = { 3, 3 };
    Ipp32f *pBuffer = NULL;                /* Pointer to the work buffer */
    IppiFilterBorderSpec* pSpec = NULL;   /* context structure */
    int iTmpBufSize = 0, iSpecSize = 0;   /* Common work buffer size */
    IppiBorderType borderType = ippBorderRepl;
    Ipp8u borderValue = 0;
    int numChannels = 1;

    //Variables used for image format conversion
    IppiSize roi;
    roi.width = WIDTH;
    roi.height = HEIGHT;

    //Compute the step in bytes of the 8u and 32f image
    //int 8uStep = roi.width * sizeof(Ipp8u); 
    //int 32fStep = roi.width * sizeof(Ipp32f);
    int 8uStep = 0; 
    int 32fStep = 0;

    //Get pointers to data
    pSrcImage = ippiMalloc_8u_C1(roi.width, roi.height, &8uStep);   //Get pointer to src image data 
    pIpp32fImage = ippiMalloc_32f_C1(roi.width, roi.height, &32fStep);  //Allocate buffer for converted image 
    pFilteredImage = ippiMalloc_32f_C1(roi.width, roi.height, &32fStep);  //Allocate buffer for converted image 
    pOutputImage = ippiMalloc_8u_C1(roi.width, roi.height, &8uStep);
        
    //Scale factor to normalize 32f image
    Ipp32f normFactor[3] = {1.0/255.0, 1.0/255.0, 1.0/255.0}; 
    Ipp32f scaleFactor[3] = {255.0, 255.0, 255.0}; 

    
    //The input image has to be normalized and single precission float type
    check_sts( status = ippiConvert_8u32f_C3R(pSrcImage, 8uStep, pIpp32fImage, 32fStep, roi) )
    check_sts( status = ippiMulC_32f_C3IR(normFactor, pIpp32fImage, 32fStep, roi) )

    //aplying high pass filter

    check_sts( status = ippiFilterBorderGetSize(kernelSize, roi, ipp32f, ipp32f, numChannels, &iSpecSize, &iTmpBufSize) )

    pSpec = (IppiFilterBorderSpec *)ippsMalloc_8u(iSpecSize);
    pBuffer = ippsMalloc_32f(iTmpBufSize);

    check_sts( status = ippiFilterBorderInit_32f(kernel, kernelSize, 1, ipp32f, numChannels, ippRndNear, pSpec) )

    check_sts( status = ippiFilterBorder_32f_C1R(pIpp32fImage, 32fStep, pFilteredImage, 32fStep, roi, borderType, &borderValue, pSpec, pBuffer) )

    //putting back everything
    check_sts( status = ippiMulC_32f_C3IR(scaleFactor, pFilteredImage, 32fStep, roi) )
    check_sts( status = ippiConvert_32f8u_C1R(pFilteredImage, 32fStep, pOutputImage , 8uStep, roi) )
    
    

EXIT_MAIN
    ippsFree(pBuffer);
    ippsFree(pSpec);
    ippiFree(pSrcImage);
    ippiFree(pIpp32fImage);
    ippiFree(pFilteredImage);
    ippiFree(pOutputImage);
    printf("Exit status %d (%s)\n", (int)status, ippGetStatusString(status));
    return (int)status;
}
