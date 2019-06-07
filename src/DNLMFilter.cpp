
#include "DNLMFilter.hpp"

// Pre-process input and select appropriate filter.
int DNLMFilter::dnlmFilter(const Ipp32f* pSrcBorder, int stepBytesSrcBorder, int srcType, const Ipp32f* pUSMImage, int stepByteUSM,  const Ipp32f* pSqrIntegralImage, int stepBytesSqrIntegral, Ipp32f* pDst, int stepBytesDst, IppiSize imageSize, int w, int w_n, float sigma_r){

    int status;

     if (srcType == CV_32FC3){
    
        //apply DNLM for color images   
        //status = this->dnlmFilter(pSrcBorder, stepBytesSrc, pUSMImage, stepByteUSM, pDst, stepBytesDst, imageSize, w, w_n, sigma_r);

     }

    else if (srcType == CV_32FC1 ){
        //apply DNLM for grayscale images   
        status = this->dnlmFilterBW(pSrcBorder, stepBytesSrcBorder, pUSMImage, stepByteUSM, pSqrIntegralImage, stepBytesSqrIntegral, pDst, stepBytesDst, imageSize, w, w_n, sigma_r);
    }
    
    else
         cout<< "Error in image format" << endl;
    return status;
}

//Implements dnlm filter for grayscale images.
int DNLMFilter::dnlmFilterBW(const Ipp32f* pSrcBorder, int stepBytesSrcBorder, const Ipp32f* pUSMImage, int stepBytesUSM, const Ipp32f* pSqrIntegralImage, int stepBytesSqrIntegral, Ipp32f* pDst, int stepBytesDst, IppiSize imageSize, int w, int w_n, float sigma_r){
    //Configuration for the correlation primitive
    IppEnum corrAlgCfg = (IppEnum) (ippAlgFFT | ippiNormNone | ippiROIValid);
    //Compute border offset for border replicated image
    const int windowTopLeftOffset = floor(w_n/2);
    const int imageTopLeftOffset = floor(w/2) + windowTopLeftOffset;
    const int neighborhoodStartOffset = imageTopLeftOffset - floor(w_n/2);
    const int iIRightBottomOffset = w_n;

    //Compute the sliding window size
    const IppiSize windowSize = {w, w};
    const IppiSize windowBorderSize = {w + 2*windowTopLeftOffset, w + 2*windowTopLeftOffset};
    const IppiSize neighborhoodSize = {w_n, w_n};
    //buffsize
    int bufSize=0;

    //Get correlation method buffer size
    ippiCrossCorrNormGetBufferSize(windowBorderSize, neighborhoodSize, corrAlgCfg, &bufSize);

    #pragma omp parallel shared(pDst,pSrcBorder,pUSMImage,pSqrIntegralImage)
    {
        //Variable definition
        Ipp32f *pWindowIJCorr __attribute__((aligned(64)));
        Ipp32f *pEuclDist __attribute__((aligned(64)));
        Ipp32f *pChunkMem __attribute__((aligned(64)));
        int stepBytesWindowIJCorr = 0, stepBytesEuclDist = 0;
        Ipp8u *pBuffer __attribute__((aligned(64)));
        //Variable to store summation result of filter response dor normalization
        Ipp64f sumExpTerm = 0, filterResult = 0;
        //Allocate memory for correlation result and buffer
        pChunkMem = ippiMalloc_32f_C1(windowSize.width, 2*windowSize.height, &stepBytesEuclDist);
        //Pointer arithmetic
        pEuclDist = pChunkMem;
        pWindowIJCorr = &pChunkMem[windowSize.height * stepBytesEuclDist/sizeof(Ipp32f)];
        stepBytesWindowIJCorr = stepBytesEuclDist;
        //Allocate working buffer
        pBuffer = ippsMalloc_8u( bufSize );

        #pragma omp for collapse(2)
        for (int j = 0; j < imageSize.height; ++j)
        {
            for (int i = 0; i < imageSize.width; ++i)
            {
                const int indexPdstBase = j*(stepBytesDst/sizeof(Ipp32f));
                const int indexWindowStartBase = j*(stepBytesSrcBorder/sizeof(Ipp32f));
                const int indexNeighborIJBase = (j + neighborhoodStartOffset)*(stepBytesSrcBorder/sizeof(Ipp32f));
                const int indexUSMWindowBase =(j + windowTopLeftOffset)*(stepBytesUSM/sizeof(Ipp32f));
                const int indexIINeighborIJBase = (j + neighborhoodStartOffset)*(stepBytesSqrIntegral/sizeof(Ipp32f));
                const int indexIINeighborIJBaseWOffset = (j + neighborhoodStartOffset + iIRightBottomOffset)*(stepBytesSqrIntegral/sizeof(Ipp32f));
                //Get summation of (i,j) neighborhood area
                const Ipp32f sqrSumIJNeighborhood = pSqrIntegralImage[indexIINeighborIJBaseWOffset + (i + neighborhoodStartOffset+ iIRightBottomOffset)] 
                                                    + pSqrIntegralImage[indexIINeighborIJBase + (i + neighborhoodStartOffset)] 
                                                    - pSqrIntegralImage[indexIINeighborIJBase + (i + neighborhoodStartOffset + iIRightBottomOffset)] 
                                                    - pSqrIntegralImage[indexIINeighborIJBaseWOffset + (i + neighborhoodStartOffset)];

                const Ipp32f *pWindowStart = &pSrcBorder[indexWindowStartBase + i]; 
                const Ipp32f *pNeighborhoodStartIJ = &pSrcBorder[indexIINeighborIJBase + (i + neighborhoodStartOffset)];
                const Ipp32f *pUSMWindowStart = (Ipp32f *) &pUSMImage[indexUSMWindowBase+(i + windowTopLeftOffset)];

                
                ippiCrossCorrNorm_32f_C1R( pWindowStart, stepBytesSrcBorder, windowBorderSize, pNeighborhoodStartIJ, stepBytesSrcBorder, neighborhoodSize, pWindowIJCorr, stepBytesWindowIJCorr, corrAlgCfg, pBuffer);

                #pragma vector aligned
                for (int n = 0; n < windowSize.height; ++n)
                {
                    const int indexEuclDistBase = n*(stepBytesEuclDist/sizeof(Ipp32f));
                    const int indexWindowIJCorr = n*(stepBytesWindowIJCorr/sizeof(Ipp32f));
                    const int indexIINeighborMNBase = (j + n )*(stepBytesSqrIntegral/sizeof(Ipp32f));
                    const int indexIINeighborMNBaseWOffset = (j + n + iIRightBottomOffset)*(stepBytesSqrIntegral/sizeof(Ipp32f));

                    #pragma vector aligned
                    for (int m = 0; m < windowSize.width; ++m)
                    {

                        //Get summation of (m,n) neighborhood area
                        const Ipp32f sqrSumMNNeighborhood = pSqrIntegralImage[indexIINeighborMNBaseWOffset + (i + m  + iIRightBottomOffset)] 
                                                            + pSqrIntegralImage[indexIINeighborMNBase + (i + m )] 
                                                            - pSqrIntegralImage[indexIINeighborMNBase + (i + m  + iIRightBottomOffset)] 
                                                            - pSqrIntegralImage[indexIINeighborMNBaseWOffset + (i + m )];

                        pEuclDist[indexEuclDistBase + m] = sqrSumMNNeighborhood + sqrSumIJNeighborhood -2*pWindowIJCorr[indexWindowIJCorr + m];
                    
                        
                    }
                }

                ippiDivC_32f_C1IR((Ipp32f) -(sigma_r * sigma_r), pEuclDist, stepBytesEuclDist, windowSize);
                ippiExp_32f_C1IR(pEuclDist, stepBytesEuclDist, windowSize);
                ippiSum_32f_C1R(pEuclDist, stepBytesEuclDist, windowSize, &sumExpTerm, ippAlgHintNone);
                ippiMul_32f_C1IR(pUSMWindowStart, stepBytesUSM, pEuclDist, stepBytesEuclDist, windowSize);
                ippiSum_32f_C1R(pEuclDist, stepBytesEuclDist, windowSize, &filterResult, ippAlgHintNone);

                pDst[indexPdstBase+i] = (Ipp32f) (filterResult/ sumExpTerm);
            }
        }
        ippiFree(pChunkMem);
        ippsFree(pBuffer);
    }


    



    return 1;
    
}
