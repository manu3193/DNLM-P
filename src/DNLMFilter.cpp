
#include "DNLMFilter.hpp"

// Pre-process input and select appropriate filter.
int DNLMFilter::dnlmFilter(const Ipp32f* pSrc, int stepBytesSrc, int srcType, Ipp32f* pUSMImage, int stepByteUSM, Ipp32f* pDst, int stepBytesDst, IppiSize imageSize, int w, int w_n, float sigma_r){

    int status;

     if (srcType == CV_32FC3){
    
        //apply DNLM for color images   
        //status = this->dnlmFilter(pSrc, stepBytesSrc, pUSMImage, stepByteUSM, pDst, stepBytesDst, imageSize, w, w_n, sigma_r);

     }

    else if (srcType == CV_32FC1 ){
        //apply DNLM for grayscale images   
        status = this->dnlmFilterBW(pSrc, stepBytesSrc, pUSMImage, stepByteUSM, pDst, stepBytesDst, imageSize, w, w_n, sigma_r);
    }
    
    else
         cout<< "Error in image format" << endl;
    return status;
}

//Implements dnlm filter for grayscale images.
int DNLMFilter::dnlmFilterBW(const Ipp32f* pSrc, int stepBytesSrc, Ipp32f* pUSMImage, int stepBytesUSM, Ipp32f* pDst, int stepBytesDst, IppiSize imageSize, int w, int w_n, float sigma_r){
    //Variable to store status
    int status, iMin, iMax, jMin, jMax;
    Ipp32f *pSrcBorder = NULL, *pSqrDist = NULL;
    int stepBytesSrcBorder = 0, stepBytesSqrDist = 0;
    //Configuration of the distance calculation algorthm
    IppEnum normL2AlgCfg = (IppEnum)(ippAlgAuto | ippiNormNone | ippiROIValid);
    Ipp8u *pBuffer;
    int bufSize=0;
    //Variable to store summation result of filter response dor normalization
    Ipp64f sumExpTerm = 0, filterResult = 0;

    //Compute border offset for border replicated image
    int windowTopLeftOffset = floor(w_n/2);
    int imageTopLeftOffset = floor(w/2) + windowTopLeftOffset;
    int tplStartOffset = imageTopLeftOffset - floor(w_n/2);

    IppiSize imageBorderSize = {imageSize.width + 2*imageTopLeftOffset, imageSize.height + 2*imageTopLeftOffset};

    //Compute the sliding window size
    IppiSize windowSize = {w, w};
    IppiSize windowBorderSize = {w + 2*windowTopLeftOffset, w + 2*windowTopLeftOffset};
    IppiSize tplSize = {w_n, w_n};


    //Allocate memory for image with borders
    pSrcBorder = ippiMalloc_32f_C1(imageBorderSize.width, imageBorderSize.height, &stepBytesSrcBorder);
    //Allocate memory for sqrtDist matrix
    pSqrDist = ippiMalloc_32f_C1(windowSize.width, windowSize.height, &stepBytesSqrDist);

    // Replicate border for full image filtering
    status = ippiCopyReplicateBorder_32f_C1R(pSrc, stepBytesSrc, imageSize, pSrcBorder, stepBytesSrcBorder, imageBorderSize, imageTopLeftOffset, imageTopLeftOffset);

    //Configure sqrDist template matching algorithm
    status = ippiSqrDistanceNormGetBufferSize(windowBorderSize, tplSize, normL2AlgCfg, &bufSize);
    //Allocate buffer for template matching algorithm
    pBuffer = ippsMalloc_8u( bufSize );

    Ipp32f *pWindowStart, *pTplStart, *pUSMWindowStart;

    cout << "Image H: "<< imageSize.height << " W: " << imageSize.width <<endl;
    cout << "Image w/border H: "<< imageBorderSize.height << " W: " << imageBorderSize.width <<endl;
    cout << "Window H: "<< windowSize.height << " W: " << windowSize.width <<endl;
    cout << "Window w/border H: "<< windowBorderSize.height << " W: " << windowBorderSize.width <<endl;
    cout << "Patch H: "<< tplSize.height << " W: " << tplSize.width <<endl;

    for (int j = 0; j < imageSize.height; ++j)
    {
        for (int i = 0; i < imageSize.width; ++i)
        {
            
            pWindowStart = pSrcBorder + (j*(stepBytesSrcBorder/sizeof(Ipp32f))+i); 
            pTplStart = pSrcBorder + (j + tplStartOffset)*(stepBytesSrcBorder/sizeof(Ipp32f))+(i + tplStartOffset);
            pUSMWindowStart = pUSMImage + (j*(stepBytesUSM/sizeof(Ipp32f))+i);
            status = ippiSqrDistanceNorm_32f_C1R( pWindowStart, stepBytesSrcBorder, windowBorderSize, pTplStart, stepBytesSrcBorder, tplSize, pSqrDist, stepBytesSqrDist, normL2AlgCfg, pBuffer);
            
            status = ippiDivC_32f_C1IR((Ipp32f) -(sigma_r * sigma_r), pSqrDist, stepBytesSqrDist, windowSize);
            status = ippiExp_32f_C1IR(pSqrDist, stepBytesSqrDist, windowSize);
            status = ippiSum_32f_C1R(pSqrDist, stepBytesSqrDist, windowSize, &sumExpTerm, ippAlgHintNone);
            status = ippiMul_32f_C1IR(pUSMWindowStart, stepBytesUSM, pSqrDist, stepBytesSqrDist, windowSize);
            status = ippiDivC_32f_C1IR(sumExpTerm, pSqrDist, stepBytesSqrDist, windowSize);
            status = ippiSum_32f_C1R(pSqrDist, stepBytesSqrDist, windowSize, &filterResult, ippAlgHintNone);

            pDst[i*(stepBytesDst/sizeof(Ipp32f))+j] = (Ipp32f) filterResult;

            if(status!=ippStsNoErr) cout << "Error " << status << endl;

        }
    }


    ippiFree(pSrcBorder);
    ippiFree(pSqrDist);
    ippsFree(pBuffer);

    return 1;
    
}
