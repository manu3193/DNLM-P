
#include "DNLMFilter.hpp"

void DNLM_OpenACC(const Npp32f*, int, Npp32f*, int, int, int, NppiSize, NppiSize, NppiSize, float);

// Pre-process input and select appropriate filter.
//int DNLMFilter::dnlmFilter(const Npp32f* pSrcBorder, int stepBytesSrcBorder, int srcType, const Npp32f* pUSMImage, int stepByteUSM, Npp32f* pDst, int stepBytesDst, NppiSize imageSize, int w, int w_n, float sigma_r){
int DNLMFilter::dnlmFilter(const Npp32f* pSrcBorder, int stepBytesSrcBorder, int srcType, Npp32f* pDst, int stepBytesDst, NppiSize imageSize, int w, int w_n, float sigma_r){
    int status;

     if (srcType == CV_32FC3){
    
        //apply DNLM for color images   
        //status = this->dnlmFilter(pSrcBorder, stepBytesSrc, pUSMImage, stepByteUSM, pDst, stepBytesDst, imageSize, w, w_n, sigma_r);

     }

    else if (srcType == CV_32FC1 ){
        //apply DNLM for grayscale images   
        //OLD status = this->dnlmFilterBW(pSrcBorder, stepBytesSrcBorder, pUSMImage, stepByteUSM, pDst, stepBytesDst, imageSize, w, w_n, sigma_r);
        status = this->dnlmFilterBW(pSrcBorder, stepBytesSrcBorder, pDst, stepBytesDst, imageSize, w, w_n, sigma_r);
    }
    
    else
         cout<< "Error in image format" << endl;
    return status;
}

//Implements dnlm filter for grayscale images.
//OLD int DNLMFilter::dnlmFilterBW(const Npp32f* pSrcBorder, int stepBytesSrcBorder, const Npp32f* pUSMImage, int stepBytesUSM, Npp32f* pDst, int stepBytesDst, NppiSize imageSize, int w, int w_n, float sigma_r){
int DNLMFilter::dnlmFilterBW(const Npp32f* pSrcBorder, int stepBytesSrcBorder, Npp32f* pDst, int stepBytesDst, NppiSize imageSize, int w, int w_n, float sigma_r){
    //Compute border offset for image
    const int neighborRadius = floor(w_n/2);
    //Compute window radius
    const int windowRadius = floor(w/2);

    //Compute the sliding window size
    const NppiSize windowSize = {w, w};
    const NppiSize neighborhoodSize = {w_n, w_n};

    DNLM_OpenACC(pSrcBorder, stepBytesSrcBorder, pDst, stepBytesDst, windowRadius, neighborRadius, imageSize, windowSize, neighborhoodSize, sigma_r);  

    return 1;
}
