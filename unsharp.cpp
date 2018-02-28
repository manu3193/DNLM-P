#include <ipp.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>

#define WIDTH  1920  /* image width */
#define HEIGHT  1080  /* image height */
const Ipp32f kernel[3*3] = {-1/8, -1/8, -1/8, -1/8, 16/8, -1/8, -1/8, -1/8, -1/8}; // Define high pass filter

/* Next two defines are created to simplify code reading and understanding */
#define EXIT_MAIN exitLine:                                  /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine; /* Go to Exit if IPP function returned status different from ippStsNoErr */

using namespace cv; 

int main(int, char**)
{
    IppStatus status = ippStsNoErr;
    Ipp32f *pIpp32fImage = NULL, *pFilteredImage = NULL;
    Ipp8u *pSrcImage = NULL, *pOutputImage = NULL;
    Mat inputImage, outputImage;
    IppiSize  kernelSize = { 3, 3 };
    Ipp8u *pBuffer = NULL;                /* Pointer to the work buffer */
    IppiFilterBorderSpec* pSpec = NULL;   /* context structure */
    int iTmpBufSize = 0, iSpecSize = 0;   /* Common work buffer size */
    IppiBorderType borderType = ippBorderRepl;
    Ipp32f borderValue = 0.0;
    int numChannels = 1;

    //Variables used for image format conversion
    IppiSize roi;
    roi.width = WIDTH;
    roi.height = HEIGHT;

    //Step in bytes of 32f image
    int stepSize32f = 0;

    //Loading image
    inputImage = imread('lena.bmp', IMREAD_GRAYSCALE);
    //The output image has the same shape of the input one
    outputImage = inputImage.clone();

    printf("Allocating image buffers\n");

    //Get pointers to data
    pIpp32fImage = ippiMalloc_32f_C1(roi.width, roi.height, &stepSize32f);  //Allocate buffer for converted image 
    pFilteredImage = ippiMalloc_32f_C1(roi.width, roi.height, &stepSize32f);  //Allocate buffer for converted image 
    //Get pointer to cv::Mat data 
    pSrcImage = (Ipp8u*)&inputImage.data[0]   
    pOutputImage = (Ipp8u*)&outputImage.data[0];
        
    //Scale factor to normalize 32f image
    Ipp32f normFactor[3] = {1.0/255.0, 1.0/255.0, 1.0/255.0}; 
    Ipp32f scaleFactor[3] = {255.0, 255.0, 255.0}; 

    
    printf("Converting image to 32f\n"); 
    //The input image has to be normalized and single precission float type
    check_sts( status = ippiConvert_8u32f_C3R(pSrcImage, inputImage.step[0], pIpp32fImage, stepSize32f, roi) )
    printf("Normalizing image to get values from 0 to 1\n");
    check_sts( status = ippiMulC_32f_C3IR(normFactor, pIpp32fImage, stepSize32f, roi) )

    //aplying high pass filter
    printf("Calculating filter buffer size\n");
    check_sts( status = ippiFilterBorderGetSize(kernelSize, roi, ipp32f, ipp32f, numChannels, &iSpecSize, &iTmpBufSize) )

    printf("Allocating filter buffer and specification\n");
    pSpec = (IppiFilterBorderSpec *)ippsMalloc_8u(iSpecSize);
    pBuffer = ippsMalloc_8u(iTmpBufSize);

    printf("Initializing filter\n");
    check_sts( status = ippiFilterBorderInit_32f(kernel, kernelSize, ipp32f, numChannels, ippRndNear, pSpec) )
    printf("Applying filter\n");
    check_sts( status = ippiFilterBorder_32f_C1R(pIpp32fImage, stepSize32f, pFilteredImage, stepSize32f, roi, borderType, &borderValue, pSpec, pBuffer) )

    //putting back everything
    printf("Denormalizing image\n");
    check_sts( status = ippiMulC_32f_C3IR(scaleFactor, pFilteredImage, stepSize32f, roi) )
    printf("Converting image back to 8u\n");
    check_sts( status = ippiConvert_32f8u_C1R(pFilteredImage, stepSize32f, pOutputImage , outputImage.step[0], roi, ippRndNear) )
    printf("Done..\n");
    imread('lena_sharp.bmp', pOutputImage);
    

EXIT_MAIN
    printf("Freeing memory..\n");
    ippsFree(pBuffer);
    ippsFree(pSpec);
    ippiFree(pIpp32fImage); //Dont know why freeing the memory pointed by pIpp32fImage gives segfault
    ippiFree(pFilteredImage);
    printf("Exit status %d (%s)\n", (int)status, ippGetStatusString(status));
    return (int)status;
}
