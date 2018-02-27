


#include <stdio.h>
#include "ipp.h"

#define WIDTH  1920  /* image width */
#define HEIGHT  1080  /* image height */
const Ipp32f kernel[3*3] = {-1/8, -1/8, -1/8, -1/8, 16/8, -1/8, -1/8, -1/8, -1/8}; // Define high pass filter

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
    Ipp32f borderValue = 0.0    ;
    int numChannels = 1;

    //Variables used for image format conversion
    IppiSize roi;
    roi.width = WIDTH;
    roi.height = HEIGHT;

    //Compute the step in bytes of the 8u and 32f image
    //int stepSize8u = roi.width * sizeof(Ipp8u); 
    //int stepSize32f = roi.width * sizeof(Ipp32f);
    int stepSize8u = 0; 
    int stepSize32f = 0;

    //Get pointers to data
    pSrcImage = ippiMalloc_8u_C1(roi.width, roi.height, &stepSize8u);   //Get pointer to src image data 
    pIpp32fImage = ippiMalloc_32f_C1(roi.width, roi.height, &stepSize32f);  //Allocate buffer for converted image 
    pFilteredImage = ippiMalloc_32f_C1(roi.width, roi.height, &stepSize32f);  //Allocate buffer for converted image 
    pOutputImage = ippiMalloc_8u_C1(roi.width, roi.height, &stepSize8u);
        
    //Scale factor to normalize 32f image
    Ipp32f normFactor[3] = {1.0/255.0, 1.0/255.0, 1.0/255.0}; 
    Ipp32f scaleFactor[3] = {255.0, 255.0, 255.0}; 

    
    //The input image has to be normalized and single precission float type
    check_sts( status = ippiConvert_8u32f_C3R(pSrcImage, stepSize8u, pIpp32fImage, stepSize32f, roi) )
    check_sts( status = ippiMulC_32f_C3IR(normFactor, pIpp32fImage, stepSize32f, roi) )

    //aplying high pass filter

    check_sts( status = ippiFilterBorderGetSize(kernelSize, roi, ipp32f, ipp32f, numChannels, &iSpecSize, &iTmpBufSize) )

    pSpec = (IppiFilterBorderSpec *)ippsMalloc_32f(iSpecSize);
    pBuffer = ippsMalloc_32f(iTmpBufSize);

    check_sts( status = ippiFilterBorderInit_32f(kernel, kernelSize, ipp32f, numChannels, ippRndNear, pSpec) )

    check_sts( status = ippiFilterBorder_32f_C1R(pIpp32fImage, stepSize32f, pFilteredImage, stepSize32f, roi, borderType, &borderValue, pSpec, pBuffer) )

    //putting back everything
    check_sts( status = ippiMulC_32f_C3IR(scaleFactor, pFilteredImage, stepSize32f, roi) )
    check_sts( status = ippiConvert_32f8u_C1R(pFilteredImage, stepSize32f, pOutputImage , stepSize8u, roi, ippRndNear) )
    
    

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
