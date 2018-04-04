
#include "DNLMFilter.hpp"

// Pre-process input and select appropriate filter.
int DNLMFilter::dnlmFilter(const Ipp32f* pSrc, int stepBytesSrc, int srcType, const Ipp32f* pUSMImage, int stepByteUSM, Ipp32f* pDst, int stepBytesDst, IppiSize imageSize, int w, int w_n, float sigma_r){

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
int DNLMFilter::dnlmFilterBW(const Ipp32f* pSrc, int stepBytesSrc, const Ipp32f* pUSMImage, int stepBytesUSM, Ipp32f* pDst, int stepBytesDst, IppiSize imageSize, int w, int w_n, float sigma_r){
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
            
            pWindowStart = &pSrcBorder[j*(stepBytesSrcBorder/sizeof(Ipp32f))+i]; 
            pNeighborhoodStartIJ = &pSrcBorder[(j + neighborhoodStartOffset)*(stepBytesSrcBorder/sizeof(Ipp32f))+(i + neighborhoodStartOffset)];
            pUSMWindowStart = (Ipp32f *) &pUSMImage[j*(stepBytesUSM/sizeof(Ipp32f))+i];

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
                for (int m = 0; m < windowSize.width; ++m)
                {
                    pNeighborhoodStartNM = &pWindowStart[n * (stepBytesSrcBorder/sizeof(Ipp32f)) + m];

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
                    pEuclDist[n * (stepBytesEuclDist/sizeof(Ipp32f)) + m] = (Ipp32f) euclDistResult;
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
>>>>>>> b0dd51a... make image type const to prevent modification
            
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
