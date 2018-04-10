
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
    //Variable to store status
    int status, iMin, iMax, jMin, jMax;
    Ipp32f *pEuclDist = NULL;
    int stepBytesEuclDist = 0;

    //Variable to store summation result of filter response dor normalization
    Ipp64f sumExpTerm = 0, filterResult = 0, euclDistResult = 0;

    //Compute border offset for border replicated image
    int windowTopLeftOffset = floor(w_n/2);
    int imageTopLeftOffset = floor(w/2) + windowTopLeftOffset;
    int neighborhoodStartOffset = imageTopLeftOffset - floor(w_n/2);

    //Compute the sliding window size
    IppiSize windowSize = {w, w};
    IppiSize windowBorderSize = {w + 2*windowTopLeftOffset, w + 2*windowTopLeftOffset};
    IppiSize neighborhoodSize = {w_n, w_n};


    //Allocate memory for sqrtDist matrix
    pEuclDist = ippiMalloc_32f_C1(windowSize.width, windowSize.height, &stepBytesEuclDist);

    Ipp32f *pWindowStart, *pNeighborhoodStartIJ, *pNeighborhoodStartNM, *pUSMWindowStart;

    /*cout << "Image H: "<< imageSize.height << " W: " << imageSize.width <<endl;
    cout << "Image w/border H: "<< imageBorderSize.height << " W: " << imageWBorderSize.width <<endl;
    cout << "Window H: "<< windowSize.height << " W: " << windowSize.width <<endl;
    cout << "Window w/border H: "<< windowBorderSize.height << " W: " << windowBorderSize.width <<endl;
    cout << "Neighborhood H: "<< neighborhoodSize.height << " W: " << neighborhoodSize.width <<endl;*/

    // cout << "image : "<<endl;
    // for (int r = 0; r < imageBorderSize.height; ++r)
    // {
    //     for (int s = 0; s < imageBorderSize.width; ++s)
    //     {
    //         cout << pSrcBorder[r*(stepBytesSrcBorder/sizeof(Ipp32f)) + s] << " ";
    //     }
    //     cout <<endl;
    // }

    for (int j = 0; j < imageSize.height; ++j)
    {
        const int indexPdstBase = j*(stepBytesDst/sizeof(Ipp32f));
        const int indexWindowStartBase = j*(stepBytesSrcBorder/sizeof(Ipp32f));
        const int indexNeighborIJBase = (j + neighborhoodStartOffset)*(stepBytesSrcBorder/sizeof(Ipp32f));
        const int indexUSMWindowBase =j*(stepBytesUSM/sizeof(Ipp32f));

        for (int i = 0; i < imageSize.width; ++i)
        {
            
            pWindowStart = (Ipp32f *) &pSrcBorder[indexWindowStartBase+i]; 
            pNeighborhoodStartIJ = (Ipp32f *) &pSrcBorder[indexNeighborIJBase + (i + neighborhoodStartOffset)];
            pUSMWindowStart = (Ipp32f *) &pUSMImage[indexUSMWindowBase+i];

            // cout << "window : "<<endl;
            // for (int r = 0; r < windowBorderSize.height; ++r)
            // {
            //     for (int s = 0; s < windowBorderSize.width; ++s)
            //     {
            //         cout << pWindowStart[r*(stepBytesSrcBorder/sizeof(Ipp32f)) + s] << " ";
            //     }
            //     cout <<endl;
            // }
            // cout << "IJ : "<<endl;
            // for (int r = 0; r < neighborhoodSize.height; ++r)
            // {
            //     for (int s = 0; s < neighborhoodSize.width; ++s)
            //     {
            //         cout << pNeighborhoodStartIJ[r*(stepBytesSrcBorder/sizeof(Ipp32f)) + s] << " ";
            //     }
            //     cout <<endl;
            // }

            for (int n = 0; n < windowSize.height; ++n)
            {
                const int indexEuclDistBase = n * (stepBytesEuclDist/sizeof(Ipp32f));
                const int indexNeighborNMBase = n * (stepBytesSrcBorder/sizeof(Ipp32f));

                for (int m = 0; m < windowSize.width; ++m)
                {
                    pNeighborhoodStartNM = &pWindowStart[indexNeighborNMBase + m];

                     // cout << "NM : " <<endl;
                     // for (int r = 0; r < neighborhoodSize.height; ++r)
                     // {
                     //     for (int s = 0; s < neighborhoodSize.width; ++s)
                     //     {
                     //         cout << pNeighborhoodStartNM[r*(stepBytesSrcBorder/sizeof(Ipp32f)) + s] << " ";
                     //     }
                     //     cout <<endl;
                     // }


                    status = ippiNormDiff_L2_32f_C1R(pNeighborhoodStartNM, stepBytesSrcBorder, pNeighborhoodStartIJ, stepBytesSrcBorder, neighborhoodSize, &euclDistResult, ippAlgHintNone);
                    pEuclDist[indexEuclDistBase + m] = (Ipp32f) euclDistResult;
                }
            }

            // cout << "euclDistResult : "<<endl;
            // for (int r = 0; r < windowSize.height; ++r)
            // {
            //     for (int s = 0; s < windowSize.width; ++s)
            //     {
            //         cout << pEuclDist[r*(stepBytesEuclDist/sizeof(Ipp32f)) + s] << " ";
            //     }
            //     cout <<endl;
            // }
            
            status = ippiDivC_32f_C1IR((Ipp32f) -(sigma_r * sigma_r), pEuclDist, stepBytesEuclDist, windowSize);
            status = ippiExp_32f_C1IR(pEuclDist, stepBytesEuclDist, windowSize);
            status = ippiSum_32f_C1R(pEuclDist, stepBytesEuclDist, windowSize, &sumExpTerm, ippAlgHintNone);
            status = ippiMul_32f_C1IR(pUSMWindowStart, stepBytesUSM, pEuclDist, stepBytesEuclDist, windowSize);
            status = ippiSum_32f_C1R(pEuclDist, stepBytesEuclDist, windowSize, &filterResult, ippAlgHintNone);

            pDst[indexPdstBase+i] = (Ipp32f) (filterResult/ sumExpTerm);
            //cout << pDst[j*(stepBytesDst/sizeof(Ipp32f))+i] << " ";

            if(status!=ippStsNoErr) cout << "Error " << status << endl;

        }
    }


    ippiFree(pEuclDist);

    return 1;
    
}
