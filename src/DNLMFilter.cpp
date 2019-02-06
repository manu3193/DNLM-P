
#include "DNLMFilter.hpp"

// Pre-process input and select appropriate filter.
int DNLMFilter::dnlmFilter(const Npp32f* pSrcBorder, int stepBytesSrcBorder, int srcType, const Npp32f* pUSMImage, int stepByteUSM, Npp32f* pDst, int stepBytesDst, NppiSize imageSize, int w, int w_n, float sigma_r){

    int status;

     if (srcType == CV_32FC3){
    
        //apply DNLM for color images   
        //status = this->dnlmFilter(pSrcBorder, stepBytesSrc, pUSMImage, stepByteUSM, pDst, stepBytesDst, imageSize, w, w_n, sigma_r);

     }

    else if (srcType == CV_32FC1 ){
        //apply DNLM for grayscale images   
        status = this->dnlmFilterBW(pSrcBorder, stepBytesSrcBorder, pUSMImage, stepByteUSM, pDst, stepBytesDst, imageSize, w, w_n, sigma_r);
    }
    
    else
         cout<< "Error in image format" << endl;
    return status;
}

//Implements dnlm filter for grayscale images.
int DNLMFilter::dnlmFilterBW(const Npp32f* pSrcBorder, int stepBytesSrcBorder, const Npp32f* pUSMImage, int stepBytesUSM, Npp32f* pDst, int stepBytesDst, NppiSize imageSize, int w, int w_n, float sigma_r){
    
    //Compute border offset for border replicated image
    const int windowTopLeftOffset = floor(w_n/2);
    const int imageTopLeftOffset = floor(w/2) + windowTopLeftOffset;
    const int neighborhoodStartOffset = imageTopLeftOffset - floor(w_n/2);

    //Compute the sliding window size
    const NppiSize windowSize = {w, w};
    const NppiSize windowBorderSize = {w + 2*windowTopLeftOffset, w + 2*windowTopLeftOffset};
    const NppiSize neighborhoodSize = {w_n, w_n};

   
    //Variable to store status
    Npp32f *pEuclDist;
    Npp8u *pNormBuffer, *pSumBuffer;
    int stepBytesEuclDist = 0;
    int normBufferSize = 0;
    int sumBufferSize = 0;
    int error = 0;
    //Variable to store summation result of filter response dor normalization
    Npp64f sumExpTerm = 0, filterResult = 0, euclDistResult = 0;
    //Compute buffer size 
    nppiNormDiffL2GetBufferHostSize_32f_C1R(neighborhoodSize, &normBufferSize);
    nppiSumGetBufferHostSize_32f_C1R(windowSize, &sumBufferSize);
    //Allocate memory for sqrtDist matrix
    pEuclDist = nppiMalloc_32f_C1(windowSize.width, windowSize.height, &stepBytesEuclDist);
    pNormBuffer = nppsMalloc_8u(normBufferSize);
    pSumBuffer = nppsMalloc_8u(sumBufferSize);
    for (int j = 0; j < imageSize.height; ++j)
    {
        for (int i = 0; i < imageSize.width; ++i)
        {
            const int indexPdstBase = j*(stepBytesDst/sizeof(Npp32f));
            const int indexWindowStartBase = j*(stepBytesSrcBorder/sizeof(Npp32f));
            const int indexNeighborIJBase = (j + neighborhoodStartOffset)*(stepBytesSrcBorder/sizeof(Npp32f));
            const int indexUSMWindowBase =(j + windowTopLeftOffset)*(stepBytesUSM/sizeof(Npp32f));
            const Npp32f *pWindowStart = &pSrcBorder[indexWindowStartBase+i]; 
            const Npp32f *pNeighborhoodStartIJ = &pSrcBorder[indexNeighborIJBase + (i + neighborhoodStartOffset)];
            const Npp32f *pUSMWindowStart = &pUSMImage[indexUSMWindowBase+(i + windowTopLeftOffset)];

                
            for (int n = 0; n < windowSize.height; ++n)
            {
                const int indexEuclDistBase = n * (stepBytesEuclDist/sizeof(Npp32f));
                const int indexNeighborNMBase = n * (stepBytesSrcBorder/sizeof(Npp32f));

                    
                for (int m = 0; m < windowSize.width; ++m)
                {
                    const Npp32f *pNeighborhoodStartNM = &pWindowStart[indexNeighborNMBase + m];
                    error = nppiNormDiff_L2_32f_C1R(pNeighborhoodStartNM, stepBytesSrcBorder, pNeighborhoodStartIJ, stepBytesSrcBorder, neighborhoodSize, &euclDistResult, pNormBuffer);
                    error = nppiSet_32f_C1R(euclDistResult, &pEuclDist[indexEuclDistBase + m], stepBytesEuclDist, {1,1});
                }
            }
                
            error = nppiDivC_32f_C1IR((Npp32f) -(sigma_r * sigma_r), pEuclDist, stepBytesEuclDist, windowSize);
            error = nppiExp_32f_C1IR(pEuclDist, stepBytesEuclDist, windowSize);
            error = nppiSum_32f_C1R(pEuclDist, stepBytesEuclDist, windowSize, pSumBuffer, &sumExpTerm);
            error = nppiMul_32f_C1IR(pUSMWindowStart, stepBytesUSM, pEuclDist, stepBytesEuclDist, windowSize);
            error = nppiSum_32f_C1R(pEuclDist, stepBytesEuclDist, windowSize, pSumBuffer, &filterResult);
                
            nppiSet_32f_C1R((Npp32f) (filterResult/ sumExpTerm), &pDst[indexPdstBase+i], 1, {1,1});
        }
    }
    nppiFree(pEuclDist);
    return 1;
}
