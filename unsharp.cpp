#include <ipp.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
//#include <timer.h>

#define WIDTH  1920  /* image width */
#define HEIGHT  1080  /* image height */
#define KERNEL_LENGTH 5
double elapsedTime;

/* Next two defines are created to simplify code reading and understanding */
#define EXIT_MAIN exitLine:                                  /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine; /* Go to Exit if IPP function returned status different from ippStsNoErr */

using namespace cv; 



// Row major version
void IppSepFilterRC( Ipp32f* pDst,  int dstStep,  Ipp32f* pSrc,  int srcStep,  IppiSize&  roiSize,  Ipp32f* hr,  int Nr,  Ipp32f* hc,  int Nc) 
{
    IppStatus sts;

    int sizerow, sizecol;
    Ipp32f *pTmp = NULL;
    Ipp32f *pTmpLocal = NULL;
    Ipp8u *pBufferCol = NULL, *pBufferRow = NULL;

    //  flip the kernels and align the memory to please IPP 
    //Ipp32f *hc_flipped = (Ipp32f *)ippsMalloc_32f( Nc );
    //Ipp32f *hr_flipped = (Ipp32f *)ippsMalloc_32f( Nr );

    //ippsFlip_32f((const Ipp32f*)hc, hc_flipped, Nc );
    //ippsFlip_32f((const Ipp32f*)hr, hr_flipped, Nr );
    //
    Ipp32f *hc_flipped = hc;
    Ipp32f *hr_flipped = hr;

    //  compute the kernel semisizes
    int Ncss = Nc >> 1;
    int Nrss = Nr >> 1;

    //  compute the kernel offsets (0 -> odd, 1 -> even)
    int co = 1 - ( Nc % 2 );
    int ro = 1 - ( Nr % 2 );

    //  allocate temporary dst buffer
    int tmpStep;
    int tmpw;

    //  The IPP filter functions seem to need 1 more row allocated
    //  than is obvious or they sometimes crash.
    int tmpHeight = roiSize.height+Nc+1;
    int tmpWidth  = roiSize.width;

    pTmpLocal = ippiMalloc_32f_C1( roiSize.width, roiSize.height + Nc + 1, &tmpStep ) ;
    pTmp = pTmpLocal;
    tmpw = tmpStep / sizeof(Ipp32f);

    Ipp32f **ppSrc, **ppDst;
    ppSrc = (Ipp32f**) ippsMalloc_32f( roiSize.height + Nc + 1 );
    ppDst = (Ipp32f**) ippsMalloc_32f( roiSize.height );


    //  size of temporary buffers
    sts = ippiFilterRowBorderPipelineGetBufferSize_32f_C1R( roiSize, Nr, &sizerow);

    sts = ippiFilterColumnPipelineGetBufferSize_32f_C1R( roiSize, Nc, &sizecol);

    //  allocate temporary buffers
    pBufferCol = ippsMalloc_8u( sizecol );
        
    pBufferRow = ippsMalloc_8u( sizerow );

    Nrss -= ro;
    Ncss -= co;

    // organize dst buffer
    for( int ii=0,jj=Ncss;ii < roiSize.height; ++ii, ++jj ){
        ppDst[ii] = pTmp + jj * tmpw;
        ppSrc[jj] = pTmp + jj * tmpw;
    }

    IppiBorderType borderType = ippBorderRepl;

    for( int ii=0,jj=roiSize.height+Ncss;ii < roiSize.height;++ii, ++jj)        {

                ppSrc[ii] = ppSrc[Ncss];

                ppSrc[jj] = ppSrc[roiSize.height+Ncss-1];

            }



    // perform the actual convolutions
    sts = ippiFilterRowBorderPipeline_32f_C1R( (const Ipp32f*)pSrc, srcStep, 
        ppDst, roiSize, hr_flipped, Nr, Nrss, borderType, 1, pBufferRow);

    sts = ippiFilterColumnPipeline_32f_C1R( (const Ipp32f**)ppSrc, pDst, dstStep, 
        roiSize, hc_flipped, Nc, pBufferCol);

    ippsFree(ppSrc);     
    ippsFree(ppDst);      
    ippiFree(pTmpLocal);  
    ippsFree(pBufferCol); 
    ippsFree(pBufferRow); 
}







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
    int stepSize32f = 0, stepBytesFiltred = 0;

    //Loading image
    inputImage = imread("lena.bmp", IMREAD_GRAYSCALE);
    //The output image has the same shape of the input one
    outputImage = inputImage.clone();

    //Allocating image buffers

    //Get pointers to data
    pIpp32fImage = ippiMalloc_32f_C1(roi.width, roi.height, &stepSize32f);  //Allocate buffer for converted image 
    pFilteredImage = ippiMalloc_32f_C1(roi.width, roi.height, &stepBytesFiltred);  //Allocate buffer for converted image 
    //Get pointer to cv::Mat data 
    pSrcImage = (Ipp8u*)&inputImage.data[0];   
    pOutputImage = (Ipp8u*)&outputImage.data[0];

    //Generate kernel
    Ipp32f * pKernel = ippsMalloc_32f(KERNEL_LENGTH);

    for (int i = 0; i < KERNEL_LENGTH; ++i)
    {
        pKernel[i] = 1;
    }
        
    //Scale factor to normalize 32f image
    Ipp32f normFactor = 1.0/255.0; 
    Ipp32f scaleFactor = 255.0; 

    
    //Converting image to 32f
    //timerStart();
    //The input image has to be normalized and single precission float type
    check_sts( status = ippiConvert_8u32f_C1R(pSrcImage, inputImage.step[0], pIpp32fImage, stepSize32f, roi) )
    //Normalizing image to get values from 0 to 1
    check_sts( status = ippiMulC_32f_C1IR(normFactor, pIpp32fImage, stepSize32f, roi) )

    int num = 5;

    IppSepFilterRC(pFilteredImage, stepBytesFiltred, pIpp32fImage, stepSize32f, roi, pKernel, num, pKernel, num);
    
    //putting back everything
    //Denormalizing image
    check_sts( status = ippiMulC_32f_C1IR(scaleFactor, pFilteredImage, stepBytesFiltred, roi) )
    //Converting image back to 8u\n
    check_sts( status = ippiConvert_32f8u_C1R(pFilteredImage, stepBytesFiltred, pOutputImage , outputImage.step[0], roi, ippRndFinancial) )
    //elapsedTime = timerStop();
    imwrite("lena_sharp.bmp", outputImage);
    //printf("Elapsed time CV version:%d\n", elapsedTime);
    

EXIT_MAIN
    printf("Freeing memory..\n");
    ippiFree(pIpp32fImage); //Dont know why freeing the memory pointed by pIpp32fImage gives segfault
    ippiFree(pFilteredImage);
    ippsFree(pKernel);
    printf("Exit status %d (%s)\n", (int)status, ippGetStatusString(status));
    return (int)status;
}











//##################################################3
//##################################################3
