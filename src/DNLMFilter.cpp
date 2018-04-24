
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
    //Ipp status variable
    int status;
    //Variable definition
    Ipp32f *pEuclDist= NULL, *pSqrDiff= NULL, *pTmp = NULL, *pWeightsAcumm = NULL, *pTmpAcumm = NULL;
    int stepSqrDiff = 0, stepBytesEuclDist = 0, stepBytesTmp= 0, stepBytesTmpAcumm = 0, stepBytesWeightsAcumm = 0;
    //Pointer to work buffer and buffer size for the BoxFilter
    Ipp8u *pBuffer = NULL;                
    int iTmpBufSize = 0;                  
    IppiBorderType borderType = ippBorderRepl;
    const Ipp8u borderValue = 0;
    int bufSize=0;

    //Variable to store summation result of filter response dor normalization
    Ipp64f sumExpTerm = 0, filterResult = 0;

    //Compute neighborhood and window half length
    const int nHalfLen = floor(w_n/2);
    const int wHalfLen = floor(w/2);
    


    //Compute the sliding window size
    const IppiSize wROISize = {w, w};
    //Compute the neighborhood size
    const IppiSize nROISize = {w_n, w_n};
    //Compute buffer max size
    const IppiSize convROISize = {imageSize.width-2*nHalfLen-1, imageSize.height-2*nHalfLen-1};

    //Allocate memory for correlation result and buffer
    pSqrDiff = ippiMalloc_32f_C1(imageSize.width, imageSize.height, &stepSqrDiff);
    pEuclDist = ippiMalloc_32f_C1(imageSize.width, imageSize.height, &stepBytesEuclDist);
    pWeightsAcumm = ippiMalloc_32f_C1(imageSize.width, imageSize.height, &stepBytesWeightsAcumm);
    pTmpAcumm = ippiMalloc_32f_C1(imageSize.width, imageSize.height, &stepBytesTmpAcumm);
    pTmp = ippiMalloc_32f_C1(convROISize.width, convROISize.height, &stepBytesTmp);
    

    //Get buffer size for moving average filter
    status = ippiFilterBoxBorderGetBufferSize(convROISize, nROISize, ipp32f, 1, &bufSize);
    pBuffer = ippsMalloc_8u( bufSize );

    status = ippiMulC_32f_C1R(pUSMImage, stepBytesUSM, (Ipp32f) 0.01f, pTmpAcumm, stepBytesTmpAcumm, imageSize);



    for (int dn = 0; dn <= wHalfLen+1; ++dn)
    {
        
        const int n_min = max(min(nHalfLen-dn, imageSize.height-nHalfLen),nHalfLen+1);
        const int n_max = min(max(imageSize.height-nHalfLen-1-dn, nHalfLen),imageSize.height-nHalfLen-1);

        const int indexSrcImageBase = n_min*(stepBytesSrcBorder/sizeof(Ipp32f));
        const int indexSrcImageBaseWOffset = (n_min + dn)*(stepBytesSrcBorder/sizeof(Ipp32f));
        const int indexWeightsAcummBase= n_min*(stepBytesWeightsAcumm/sizeof(Ipp32f));
        const int indexWeightsAcummBaseWOffset= (n_min + dn)*(stepBytesWeightsAcumm/sizeof(Ipp32f));
        const int indexUSMImageBase = n_min*(stepBytesUSM/sizeof(Ipp32f));
        const int indexUSMImageBaseWOffset = (n_min + dn)*(stepBytesUSM/sizeof(Ipp32f));
        const int indexDstBase= n_min*(stepBytesDst/sizeof(Ipp32f));
        const int indexDstBaseWOffset= (n_min + dn)*(stepBytesDst/sizeof(Ipp32f));
        
        for (int dm = 0; dm < wHalfLen+1; ++dm)
        {
            
            if (dn>0 || (dn==0 && dm>0))
            {

                const int m_min = max(min(nHalfLen-dm,imageSize.width-nHalfLen),nHalfLen+1);
                const int m_max = min(max(imageSize.width-nHalfLen-1-dm, nHalfLen),imageSize.width-nHalfLen-1);

                const IppiSize euclROISize = {m_max-m_min+1,n_max-n_min+1};

                status = ippiSub_32f_C1R(&pSrcBorder[indexSrcImageBaseWOffset + (m_min + dm)], stepBytesSrcBorder, 
                    &pSrcBorder[indexSrcImageBase + m_min], stepBytesSrcBorder, pSqrDiff, stepSqrDiff, euclROISize);
                status = ippiSqr_32f_C1IR(pSqrDiff, stepSqrDiff, euclROISize);
                status = ippiFilterBoxBorder_32f_C1R(pSqrDiff, stepSqrDiff, pEuclDist, stepBytesEuclDist, euclROISize, nROISize, borderType, borderValue, pBuffer);

                status = ippiSqrt_32f_C1IR(pEuclDist, stepBytesEuclDist, euclROISize);
                cout << "exp res : "<<endl;
                  for (int r = 0; r < euclROISize.height; ++r)
                  {
                      for (int s = 0; s < euclROISize.width; ++s)
                      {
                          cout << pEuclDist[r*(stepBytesEuclDist/sizeof(Ipp32f)) + s] << " ";
                      }
                      cout <<endl;
                  }
                status = ippiSqr_32f_C1IR(pEuclDist, stepBytesEuclDist, euclROISize);
                status = ippiDivC_32f_C1IR((Ipp32f) -(sigma_r * sigma_r), pEuclDist, stepBytesEuclDist, euclROISize);
                status = ippiExp_32f_C1IR(pEuclDist, stepBytesEuclDist, euclROISize);



                //Performing filtering
                status = ippiMul_32f_C1R(pEuclDist, stepBytesEuclDist, &pUSMImage[indexUSMImageBaseWOffset + (m_min + dm)], stepBytesUSM, pTmp, stepBytesTmp, euclROISize);
                //Accumulate signal and weights
                status = ippiAdd_32f_C1IR(pTmp, stepBytesTmp, &pDst[indexDstBase + m_min], stepBytesDst, euclROISize);
                status = ippiAdd_32f_C1IR(pTmp, stepBytesTmp, &pWeightsAcumm[indexWeightsAcummBase + m_min], stepBytesWeightsAcumm, euclROISize);
                
                //Exploiting weights symmetry
                status = ippiMul_32f_C1R(pEuclDist, stepBytesEuclDist, &pUSMImage[indexUSMImageBase + m_min], stepBytesUSM, pTmp, stepBytesTmp, euclROISize);
                //Accumulate signal and weights 
                status = ippiAdd_32f_C1IR(pTmp, stepBytesTmp, &pDst[indexDstBaseWOffset + (m_min + dm)], stepBytesDst, euclROISize);
                status = ippiAdd_32f_C1IR(pTmp, stepBytesTmp, &pWeightsAcumm[indexWeightsAcummBaseWOffset + (m_min + dm)], stepBytesWeightsAcumm, euclROISize);
                

            }

            else if (dn==0 && dm==0)
            {
                status = ippiAddC_32f_C1IR((Ipp32f) 0.01f, pWeightsAcumm, stepBytesWeightsAcumm, imageSize );
                status = ippiAdd_32f_C1IR(pTmpAcumm, stepBytesTmpAcumm, pDst, stepBytesDst, imageSize);
            }
            

        }
    }

    status = ippiDiv_32f_C1IR(pWeightsAcumm, stepBytesWeightsAcumm, pDst, stepBytesDst, imageSize);


    //ippiFree(pSqrDiff);
    //ippiFree(pEuclDist);
    //ippiFree(pTmpAcumm);
    //ippiFree(pTmp);
    //ippiFree(pWeightsAcumm);
    //ippsFree(pBuffer);

    return 1;
    
}
