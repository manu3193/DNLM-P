
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
    //Variable to store status
    int status;
    Ipp32f *pWindowIJCorr = NULL, *pEuclDist;
    int stepBytesWindowIJCorr = 0, stepBytesEuclDist = 0;
    //Configuration for the correlation primitive
    IppEnum corrAlgCfg = (IppEnum) (ippAlgAuto | ippiNormNone | ippiROIValid);
    Ipp8u *pBuffer;
    int bufSize=0;

    //Variable to store summation result of filter response dor normalization
    Ipp64f sumExpTerm = 0, filterResult = 0;

    //Compute border offset for border replicated image
    const int windowTopLeftOffset = floor(w_n/2);
    const int imageTopLeftOffset = floor(w/2) + windowTopLeftOffset;
    const int neighborhoodStartOffset = imageTopLeftOffset - floor(w_n/2);
    const int iIRightBottomOffset = w_n;

    //Compute the sliding window size
    IppiSize windowSize = {w, w};
    IppiSize windowBorderSize = {w + 2*windowTopLeftOffset, w + 2*windowTopLeftOffset};
    IppiSize neighborhoodSize = {w_n, w_n};

    //Get correlation method buffer size
    status = ippiCrossCorrNormGetBufferSize(windowBorderSize, neighborhoodSize, corrAlgCfg, &bufSize);


    //Allocate memory for correlation result and buffer
    pWindowIJCorr = ippiMalloc_32f_C1(windowSize.width, windowSize.height, &stepBytesWindowIJCorr);
    pEuclDist = ippiMalloc_32f_C1(windowSize.width, windowSize.height, &stepBytesEuclDist);
    pBuffer = ippsMalloc_8u( bufSize );


    //DEBUG
     /*cout << "Window H: "<< windowSize.height << " W: " << windowSize.width <<endl;
     cout << "Window w/border H: "<< windowBorderSize.height << " W: " << windowBorderSize.width <<endl;
     cout << "Neighborhood H: "<< neighborhoodSize.height << " W: " << neighborhoodSize.width <<endl;
    *///


    for (int j = 0; j < imageSize.height; ++j)
    {
        const int indexPdstBase = j*(stepBytesDst/sizeof(Ipp32f));
        const int indexWindowStartBase = j*(stepBytesSrcBorder/sizeof(Ipp32f));
        const int indexNeighborIJBase = (j + neighborhoodStartOffset)*(stepBytesSrcBorder/sizeof(Ipp32f));
        const int indexUSMWindowBase =(j + windowTopLeftOffset)*(stepBytesUSM/sizeof(Ipp32f));
        const int indexIINeighborIJBase = (j + neighborhoodStartOffset)*(stepBytesSqrIntegral/sizeof(Ipp32f));
        const int indexIINeighborIJBaseWOffset = (j + neighborhoodStartOffset + iIRightBottomOffset)*(stepBytesSqrIntegral/sizeof(Ipp32f));

        
        for (int i = 0; i < imageSize.width; ++i)
        {
            //Get summation of (i,j) neighborhood area
            const Ipp32f sqrSumIJNeighborhood = pSqrIntegralImage[indexIINeighborIJBaseWOffset + (i + neighborhoodStartOffset+ iIRightBottomOffset)] 
                                                + pSqrIntegralImage[indexIINeighborIJBase + (i + neighborhoodStartOffset)] 
                                                - pSqrIntegralImage[indexIINeighborIJBase + (i + neighborhoodStartOffset + iIRightBottomOffset)] 
                                                - pSqrIntegralImage[indexIINeighborIJBaseWOffset + (i + neighborhoodStartOffset)];

            const Ipp32f *pWindowStart = &pSrcBorder[indexWindowStartBase + i]; 
            const Ipp32f *pNeighborhoodStartIJ = &pSrcBorder[indexIINeighborIJBase + (i + neighborhoodStartOffset)];
            const Ipp32f *pUSMWindowStart = (Ipp32f *) &pUSMImage[indexUSMWindowBase+(i + windowTopLeftOffset)];

            
            status = ippiCrossCorrNorm_32f_C1R( pWindowStart, stepBytesSrcBorder, windowBorderSize, pNeighborhoodStartIJ, stepBytesSrcBorder, neighborhoodSize, pWindowIJCorr, stepBytesWindowIJCorr, corrAlgCfg, pBuffer);
            
            //DEBUG
            /*cout << "Image Window"<<endl;
            for (int ii = 0; ii < windowBorderSize.width; ++ii)
            {
                for (int jj = 0; jj < windowBorderSize.height; ++jj)
                {
                    cout << pWindowStart[(ii*stepBytesSrcBorder/sizeof(Ipp32f))+jj] <<" ";
                }
                cout << endl;
            }
            cout << "Sqr Integral Window"<<endl;
            for (int ii = 0; ii < windowBorderSize.width+1; ++ii)
            {
                for (int jj = 0; jj < windowBorderSize.height+1; ++jj)
                {
                    cout << pSqrIntegralImage[(ii*stepBytesSqrIntegral/sizeof(Ipp32f))+jj] <<" ";
                }
                cout << endl;
            }
            cout << "Neighborhood IJ"<<endl;
            for (int ii = 0; ii < neighborhoodSize.width; ++ii)
            {
                for (int jj = 0; jj < neighborhoodSize.height; ++jj)
                {
                    cout << pNeighborhoodStartIJ[(ii*stepBytesSrcBorder/sizeof(Ipp32f))+jj] <<" ";
                }
                cout << endl;
            }
            cout << "Correlation IJ"<<endl;
            for (int ii = 0; ii < windowSize.width; ++ii)
            {
                for (int jj = 0; jj < windowSize.height; ++jj)
                {
                    cout << pWindowIJCorr[(ii*stepBytesWindowIJCorr/sizeof(Ipp32f))+jj] <<" ";
                }
                cout << endl;
            }
            cout <<"Neighborhood IJ summ: "<<sqrSumIJNeighborhood<<endl;
            cout << "IJ bottom_right: "<<pSqrIntegralImage[indexIINeighborIJBaseWOffset + (i + +neighborhoodStartOffset+iIRightBottomOffset)] <<endl;
            cout << "IJ top_left: "<<pSqrIntegralImage[indexIINeighborIJBase + i+neighborhoodStartOffset]<<endl; 
            cout << "IJ top_right: "<<pSqrIntegralImage[indexIINeighborIJBase + (i + neighborhoodStartOffset+iIRightBottomOffset)]<<endl; 
            cout << "IJ bottom_left: "<<pSqrIntegralImage[indexIINeighborIJBaseWOffset + i+neighborhoodStartOffset]<<endl;
           */
            //

            for (int n = 0; n < windowSize.height; ++n)
            {
                const int indexEuclDistBase = n*(stepBytesEuclDist/sizeof(Ipp32f));
                const int indexWindowIJCorr = n*(stepBytesWindowIJCorr/sizeof(Ipp32f));
                const int indexIINeighborMNBase = (j + n + neighborhoodStartOffset)*(stepBytesSqrIntegral/sizeof(Ipp32f));
                const int indexIINeighborMNBaseWOffset = (j + n + neighborhoodStartOffset + iIRightBottomOffset)*(stepBytesSqrIntegral/sizeof(Ipp32f));

                for (int m = 0; m < windowSize.width; ++m)
                {

                    //Get summation of (m,n) neighborhood area
                    const Ipp32f sqrSumMNNeighborhood = pSqrIntegralImage[indexIINeighborMNBaseWOffset + (i + m + neighborhoodStartOffset + iIRightBottomOffset)] 
                                                        + pSqrIntegralImage[indexIINeighborMNBase + (i + m + neighborhoodStartOffset)] 
                                                        - pSqrIntegralImage[indexIINeighborMNBase + (i + m + neighborhoodStartOffset + iIRightBottomOffset)] 
                                                        - pSqrIntegralImage[indexIINeighborMNBaseWOffset + (i + m + neighborhoodStartOffset)];

                    pEuclDist[indexEuclDistBase + m] = sqrSumMNNeighborhood + sqrSumIJNeighborhood -2*pWindowIJCorr[indexWindowIJCorr + m];
                
                    //DEBUG
                    /*cout <<"Neighborhood MN summ: "<<sqrSumMNNeighborhood<<endl;
                    cout << "MN bottom_right: "<<pSqrIntegralImage[indexIINeighborMNBaseWOffset + (i + m+neighborhoodStartOffset+iIRightBottomOffset)] <<endl;
                    cout << "MN top_left: "<<pSqrIntegralImage[indexIINeighborMNBase + i+m+neighborhoodStartOffset]<<endl; 
                    cout << "MN top_right: "<<pSqrIntegralImage[indexIINeighborMNBase + (i + m+neighborhoodStartOffset+iIRightBottomOffset)]<<endl; 
                    cout << "MN bottom_left: "<<pSqrIntegralImage[indexIINeighborMNBaseWOffset + i+m+neighborhoodStartOffset]<<endl;
*/
                }
            }


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


    ippiFree(pWindowIJCorr);
    ippiFree(pEuclDist);
    ippsFree(pBuffer);

    return 1;
    
}
