
#include "DNLMFilter.hpp"

// Pre-process input and select appropriate filter.
int DNLMFilter::dnlmFilter(const Ipp32f* pSrcBorder, int stepBytesSrcBorder, int srcType, const Ipp32f* pUSMImage, int stepByteUSM, Ipp32f* pDst, int stepBytesDst, IppiSize imageSize, int w, int w_n, float sigma_r){

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
int DNLMFilter::dnlmFilterBW(const Ipp32f* pSrcBorder, int stepBytesSrcBorder, const Ipp32f* pUSMImage, int stepBytesUSM, Ipp32f* pDst, int stepBytesDst, IppiSize imageSize, int w, int w_n, float sigma_r){
    
    //Compute border offset for border replicated image
    int windowTopLeftOffset = floor(w_n/2);
    int imageTopLeftOffset = floor(w/2) + windowTopLeftOffset;
    int neighborhoodStartOffset = imageTopLeftOffset - floor(w_n/2);

    //Compute the sliding window size
    const IppiSize windowSize = {w, w};
    const IppiSize windowBorderSize = {w + 2*windowTopLeftOffset, w + 2*windowTopLeftOffset};
    const IppiSize neighborhoodSize = {w_n, w_n};

    #pragma omp parallel shared(pDst,pUSMImage,pSrcBorder) 
    {
        //Variable to store status
        Ipp32f *pEuclDist __attribute__((aligned(64)));
        int stepBytesEuclDist = 0;
         //Variable to store summation result of filter response dor normalization
        Ipp64f sumExpTerm = 0, filterResult = 0, euclDistResult = 0;

        //Allocate memory for sqrtDist matrix
        pEuclDist = ippiMalloc_32f_C1(windowSize.width, windowSize.height, &stepBytesEuclDist);

        #pragma omp for collapse(2)
        for (int j = 0; j < imageSize.height; ++j)
        {
            for (int i = 0; i < imageSize.width; ++i)
            {
                const int indexPdstBase = j*(stepBytesDst/sizeof(Ipp32f));
                const int indexWindowStartBase = j*(stepBytesSrcBorder/sizeof(Ipp32f));
                const int indexNeighborIJBase = (j + neighborhoodStartOffset)*(stepBytesSrcBorder/sizeof(Ipp32f));
                const int indexUSMWindowBase =(j + windowTopLeftOffset)*(stepBytesUSM/sizeof(Ipp32f));
                const Ipp32f *pWindowStart = &pSrcBorder[indexWindowStartBase+i]; 
                const Ipp32f *pNeighborhoodStartIJ = &pSrcBorder[indexNeighborIJBase + (i + neighborhoodStartOffset)];
                const Ipp32f *pUSMWindowStart = &pUSMImage[indexUSMWindowBase+(i + windowTopLeftOffset)];

                #pragma vector aligned
                for (int n = 0; n < windowSize.height; ++n)
                {
                    const int indexEuclDistBase = n * (stepBytesEuclDist/sizeof(Ipp32f));
                    const int indexNeighborNMBase = n * (stepBytesSrcBorder/sizeof(Ipp32f));

                    #pragma vector aligned
                    for (int m = 0; m < windowSize.width; ++m)
                    {
                        const Ipp32f *pNeighborhoodStartNM = &pWindowStart[indexNeighborNMBase + m];
                        ippiNormDiff_L2_32f_C1R(pNeighborhoodStartNM, stepBytesSrcBorder, pNeighborhoodStartIJ, stepBytesSrcBorder, neighborhoodSize, &euclDistResult, ippAlgHintNone);
                        pEuclDist[indexEuclDistBase + m] = (Ipp32f) euclDistResult;
                    }
                }
                
                ippiDivC_32f_C1IR((Ipp32f) -(sigma_r * sigma_r), pEuclDist, stepBytesEuclDist, windowSize);
                ippiExp_32f_C1IR(pEuclDist, stepBytesEuclDist, windowSize);
                ippiSum_32f_C1R(pEuclDist, stepBytesEuclDist, windowSize, &sumExpTerm, ippAlgHintNone);
                ippiMul_32f_C1IR(pUSMWindowStart, stepBytesUSM, pEuclDist, stepBytesEuclDist, windowSize);
                ippiSum_32f_C1R(pEuclDist, stepBytesEuclDist, windowSize, &filterResult, ippAlgHintNone);

                pDst[indexPdstBase+i] = (Ipp32f) (filterResult/ sumExpTerm);

                //Need to do better error handling 
                //if(status!=ippStsNoErr) cout << "Error " << status << endl;

            }
        }
        ippiFree(pEuclDist);
    }

    

    return 1;
    
}
