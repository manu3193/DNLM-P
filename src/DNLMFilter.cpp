
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
    //Variable definition
    Ipp32f *pWeightsAcumm __attribute__((aligned(64)));//, *pTmpAcumm = NULL;
    int stepBytesWeightsAcumm = 0;
    //Compute neighborhood and window half length
    const int nHalfLen = floor(w_n/2);
    const int wHalfLen = floor(w/2);
    //Compute the neighborhood size
    const IppiSize nROISize = {w_n, w_n};
    //Compute buffer max size
    int bufSize=0;
    const IppiSize convROISize = {imageSize.width-2*nHalfLen-1, imageSize.height-2*nHalfLen-1};

    //Allocate memory for correlation result and buffer
    pWeightsAcumm = ippiMalloc_32f_C1(imageSize.width, imageSize.height, &stepBytesWeightsAcumm);
    //pTmpAcumm = ippiMalloc_32f_C1(imageSize.width, imageSize.height, &stepBytesTmpAcumm);
    
    //Set buffer to 0
    ippiSet_32f_C1R((Ipp32f) 0.0f, pWeightsAcumm, stepBytesWeightsAcumm, imageSize);

    //Get buffer size for moving average filter
    ippiFilterBoxBorderGetBufferSize(convROISize, nROISize, ipp32f, 1, &bufSize);
    
    #pragma omp parallel shared(pWeightsAcumm,pDst,pUSMImage,pSrcBorder) //private(dn,dm,pBuffer,pEuclDist,pSumSqrDiff,pTmp)
    {
        Ipp32f *pEuclDist __attribute__((aligned(64)));
        Ipp32f *pSumSqrDiff __attribute__((aligned(64)));
        Ipp32f *pTmp __attribute__((aligned(64)));
        Ipp32f *pTmp2 __attribute__((aligned(64)));
        Ipp32f *pThreadWeightsAcumm __attribute__((aligned(64)));
        Ipp32f *pThreadDst __attribute__((aligned(64)));
        Ipp32f *pChunkMem __attribute__((aligned(64)));
        Ipp32f *pChunkMem2 __attribute__((aligned(64)));
        int stepSumSqrDiff = 0, stepBytesEuclDist = 0, stepBytesTmp= 0, stepBytesThreadWeights = 0, stepBytesThreadDst = 0;
        IppiBorderType borderType = ippBorderConst;
        const Ipp32f borderValue = 0;
        //Pointer to work buffer and buffer size for the BoxFilter
        Ipp8u *pBuffer __attribute__((aligned(64)));;               

        pBuffer = ippsMalloc_8u( bufSize );
        //Allocate one big chunk of memory to store thread's private weights and dst buffers
        pChunkMem = ippiMalloc_32f_C1(imageSize.width, 2*imageSize.height, &stepBytesThreadWeights);
	//Init buffer to 0
        ippiSet_32f_C1R((Ipp32f) 0.0f, pChunkMem, stepBytesThreadWeights, {imageSize.width, 2*imageSize.height});
        //Perform pointer arithmetics
        pThreadWeightsAcumm = pChunkMem;
        pThreadDst = (Ipp32f*) &pChunkMem[imageSize.height * stepBytesThreadWeights/sizeof(Ipp32f)];
        stepBytesThreadDst = stepBytesThreadWeights;
        
         __itt_resume(); //Intel Advisor starts recording performance data
        __SSC_MARK(0xFACE);

        //For each distance between window patches
        #pragma omp for collapse(2) schedule(runtime)
        for (int dn = 0; dn < wHalfLen+1; ++dn)
        {   
            //#pragma omp parallel for shared(pWeightsAcumm,pDst) private(dn,pBuffer,pEuclDist,pSumSqrDiff,pTmp)
            for (int dm = 0; dm < wHalfLen+1; ++dm)
            {
                //Exploit symmetry
                if (dn>0 || (dn==0 && dm>0))
                {
                    //Compute edges of the ROI        
                    const int n_min = max(min(nHalfLen-dn, imageSize.height-nHalfLen),nHalfLen+1);
                    const int n_max = min(max(imageSize.height-nHalfLen-1-dn, nHalfLen),imageSize.height-nHalfLen-1);
                    //Compute array base index
                    const int indexSrcImageBase = n_min*(stepBytesSrcBorder/sizeof(Ipp32f));
                    const int indexSrcImageBaseWOffset = (n_min + dn)*(stepBytesSrcBorder/sizeof(Ipp32f));
                    const int indexWeightsAcummBase= n_min*(stepBytesThreadWeights/sizeof(Ipp32f));
                    const int indexWeightsAcummBaseWOffset= (n_min + dn)*(stepBytesThreadWeights/sizeof(Ipp32f));
                    const int indexUSMImageBase = n_min*(stepBytesUSM/sizeof(Ipp32f));
                    const int indexUSMImageBaseWOffset = (n_min + dn)*(stepBytesUSM/sizeof(Ipp32f));
                    const int indexDstBase= n_min*(stepBytesDst/sizeof(Ipp32f));
                    const int indexDstBaseWOffset= (n_min + dn)*(stepBytesDst/sizeof(Ipp32f));
                    //Compute edges of ROI
                    const int m_min = max(min(nHalfLen-dm,imageSize.width-nHalfLen),nHalfLen+1);
                    const int m_max = min(max(imageSize.width-nHalfLen-1-dm, nHalfLen),imageSize.width-nHalfLen-1);
                    //Compute ROI
                    const IppiSize euclROISize = {m_max-m_min+1,n_max-n_min+1};
                    //Allocate buffers memory
                    pChunkMem2 = ippiMalloc_32f_C1(euclROISize.width, 4*euclROISize.height, &stepBytesEuclDist);
                    pEuclDist = pChunkMem2;
                    pSumSqrDiff = &pChunkMem2[euclROISize.height * stepBytesEuclDist/sizeof(Ipp32f)];
                    pTmp = &pChunkMem2[2 * euclROISize.height * stepBytesEuclDist/sizeof(Ipp32f)];
                    pTmp2 = &pChunkMem2[3 * euclROISize.height * stepBytesEuclDist/sizeof(Ipp32f)];
                    stepSumSqrDiff = stepBytesTmp = stepBytesEuclDist;
                    //The ippiFilterBorder function operates as a inplace function, thus it is necessary to initialize
                    ippiSet_32f_C1R((Ipp32f) 0.0f, pEuclDist, stepBytesEuclDist, euclROISize);
                     
                    //Compute squared diference 
                    ippiSub_32f_C1R((Ipp32f*) (pSrcBorder +indexSrcImageBaseWOffset + (m_min + dm)), stepBytesSrcBorder, 
                        (Ipp32f*) (pSrcBorder + indexSrcImageBase + m_min), stepBytesSrcBorder, pSumSqrDiff, stepSumSqrDiff, euclROISize);
                    ippiSqr_32f_C1IR(pSumSqrDiff, stepSumSqrDiff, euclROISize);
                    //Compute Squared Sum Difference unsing FilterBox
                    ippiFilterBoxBorder_32f_C1R(pSumSqrDiff, stepSumSqrDiff, pEuclDist, stepBytesEuclDist, euclROISize, nROISize, borderType, &borderValue, pBuffer);        
                    ippiSqr_32f_C1IR(pEuclDist, stepBytesEuclDist, euclROISize);
                    ippiSqrt_32f_C1IR(pEuclDist, stepBytesEuclDist, euclROISize);
                    
                    //Compute weights
                    ippiDivC_32f_C1IR((Ipp32f) -(2*sigma_r * sigma_r), pEuclDist, stepBytesEuclDist, euclROISize);
                    ippiExp_32f_C1IR(pEuclDist, stepBytesEuclDist, euclROISize);          

                    //Performing filtering
                    ippiMul_32f_C1R(pEuclDist, stepBytesEuclDist, (Ipp32f*) (pUSMImage +indexUSMImageBaseWOffset + (m_min + dm)), stepBytesUSM, pTmp, stepBytesTmp, euclROISize);
                    //Exploiting weights symmetry
                    ippiMul_32f_C1R(pEuclDist, stepBytesEuclDist, (Ipp32f*) (pUSMImage + indexUSMImageBase + m_min), stepBytesUSM, pTmp2, stepBytesTmp, euclROISize);
                      
                    //Accumulate signal and weights
                    ippiAdd_32f_C1IR(pTmp, stepBytesTmp, (Ipp32f*) (pThreadDst + indexDstBase + m_min), stepBytesThreadDst, euclROISize);
                    ippiAdd_32f_C1IR(pEuclDist, stepBytesEuclDist, (Ipp32f*) (pThreadWeightsAcumm + indexWeightsAcummBase + m_min), stepBytesThreadWeights, euclROISize);
                    //Accumulate signal and weights 
                    ippiAdd_32f_C1IR(pTmp2, stepBytesTmp, (Ipp32f*) (pThreadDst + indexDstBaseWOffset + (m_min + dm)), stepBytesThreadDst, euclROISize);
                    ippiAdd_32f_C1IR(pEuclDist, stepBytesEuclDist, &pThreadWeightsAcumm[indexWeightsAcummBaseWOffset + (m_min + dm)], stepBytesThreadWeights, euclROISize);

                    ippiFree(pChunkMem2);
                }

                else if (dn==0 && dm==0)
                {   
                    //Acummulate weights and filter result
                    ippiAddC_32f_C1IR((Ipp32f) 1.0f, pThreadWeightsAcumm, stepBytesThreadWeights, imageSize );
                    ippiAdd_32f_C1IR(pUSMImage, stepBytesUSM, pThreadDst, stepBytesThreadDst, imageSize);
                }
            }
        }        
        #pragma omp critical
        {
            ippiAdd_32f_C1IR(pThreadWeightsAcumm, stepBytesThreadWeights, pWeightsAcumm, stepBytesWeightsAcumm, imageSize);
            ippiAdd_32f_C1IR(pThreadDst, stepBytesThreadDst, pDst, stepBytesDst, imageSize);
        }

        __SSC_MARK(0xDEAD);
        __itt_resume(); //Intel Advisor starts recording performance data 

        //Free resources
        ippiFree(pChunkMem);
        ippsFree(pBuffer);

    }

    
    //Threshold image
    ippiThreshold_32f_C1IR(pWeightsAcumm, stepBytesWeightsAcumm, imageSize, 1e-20f,ippCmpLess);
    //Normalize
    ippiDiv_32f_C1IR(pWeightsAcumm, stepBytesWeightsAcumm, pDst, stepBytesDst, imageSize);

    ippiFree(pWeightsAcumm);

    return 1;
    
}
