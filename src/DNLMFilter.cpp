
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
    //Variable definition
    Npp32f *pWeightsAcumm __attribute__((aligned(64)));//, *pTmpAcumm = NULL;
    int stepBytesWeightsAcumm = 0;
    //Compute neighborhood and window half length
    const int nHalfLen = floor(w_n/2);
    const int wHalfLen = floor(w/2);
    //Compute the neighborhood size
    const NppiSize nROISize = {w_n, w_n};
    NppiPoint offset = { 0, 0 };
    NppiPoint anchor = { (nROISize.width - 1) / 2, (nROISize.height - 1) / 2 };

    //Allocate memory for correlation result and buffer
    pWeightsAcumm = nppiMalloc_32f_C1(imageSize.width, imageSize.height, &stepBytesWeightsAcumm);
    
    //Set buffer to 0
    //OLD nppiSet_32f_C1R((Npp32f) 0.0f, pWeightsAcumm, stepBytesWeightsAcumm, imageSize);

    Npp32f *pEuclDist __attribute__((aligned(64)));
    Npp32f *pSumSqrDiff __attribute__((aligned(64)));
    Npp32f *pTmp __attribute__((aligned(64)));
    Npp32f *pTmp2 __attribute__((aligned(64)));
    Npp32f *pThreadWeightsAcumm __attribute__((aligned(64)));
    Npp32f *pThreadDst __attribute__((aligned(64)));
    Npp32f *pChunkMem __attribute__((aligned(64)));
    Npp32f *pChunkMem2 __attribute__((aligned(64)));
    int stepSumSqrDiff = 0, stepBytesEuclDist = 0, stepBytesTmp= 0, stepBytesThreadWeights = 0, stepBytesThreadDst = 0;
    
    //Allocate one big chunk of memory to store thread's private weights and dst buffers
    pChunkMem = nppiMalloc_32f_C1(imageSize.width, 2*imageSize.height, &stepBytesThreadWeights);
    //Init buffer to 0
    //OLD nppiSet_32f_C1R((Npp32f) 0.0f, pChunkMem, stepBytesThreadWeights, {imageSize.width, 2*imageSize.height});
    //Perform pointer arithmetics
    pThreadWeightsAcumm = pChunkMem;
    pThreadDst = (Npp32f*) &pChunkMem[imageSize.height * stepBytesThreadWeights/sizeof(Npp32f)];
    stepBytesThreadDst = stepBytesThreadWeights;
    
    //Allocate buffers memory
    pChunkMem2 = nppiMalloc_32f_C1(imageSize.width, 4*imageSize.height, &stepBytesEuclDist);
    cudaProfilerStart();
    //For each distance between window patches
    for (int dn = 0; dn < wHalfLen+1; ++dn)
    {   
        for (int dm = 0; dm < wHalfLen+1; ++dm)
        {
            //Exploit symmetry
            if (dn>0 || (dn==0 && dm>0))
            {
                //Compute edges of the ROI        
                const int n_min = max(min(nHalfLen-dn, imageSize.height-nHalfLen),nHalfLen+1);
                const int n_max = min(max(imageSize.height-nHalfLen-1-dn, nHalfLen),imageSize.height-nHalfLen-1);
                //Compute array base index
                const int indexSrcImageBase = n_min*(stepBytesSrcBorder/sizeof(Npp32f));
                const int indexSrcImageBaseWOffset = (n_min + dn)*(stepBytesSrcBorder/sizeof(Npp32f));
                const int indexWeightsAcummBase= n_min*(stepBytesThreadWeights/sizeof(Npp32f));
                const int indexWeightsAcummBaseWOffset= (n_min + dn)*(stepBytesThreadWeights/sizeof(Npp32f));
                const int indexUSMImageBase = n_min*(stepBytesUSM/sizeof(Npp32f));
                const int indexUSMImageBaseWOffset = (n_min + dn)*(stepBytesUSM/sizeof(Npp32f));
                const int indexDstBase= n_min*(stepBytesDst/sizeof(Npp32f));
                const int indexDstBaseWOffset= (n_min + dn)*(stepBytesDst/sizeof(Npp32f));
                //Compute edges of ROI
                const int m_min = max(min(nHalfLen-dm,imageSize.width-nHalfLen),nHalfLen+1);
                const int m_max = min(max(imageSize.width-nHalfLen-1-dm, nHalfLen),imageSize.width-nHalfLen-1);
                //Compute ROI
                const NppiSize euclROISize = {m_max-m_min+1,n_max-n_min+1};
                //Get respective pointer from memory chunk 
                pEuclDist = pChunkMem2;
                pSumSqrDiff = &pChunkMem2[euclROISize.height * stepBytesEuclDist/sizeof(Npp32f)];
                pTmp = &pChunkMem2[2 * euclROISize.height * stepBytesEuclDist/sizeof(Npp32f)];
                pTmp2 = &pChunkMem2[3 * euclROISize.height * stepBytesEuclDist/sizeof(Npp32f)];
                stepSumSqrDiff = stepBytesTmp = stepBytesEuclDist;
                //The ippiFilterBorder function operates as a inplace function, thus it is necessary to initialize
                //OLD nppiSet_32f_C1R((Npp32f) 0.0f, pEuclDist, stepBytesEuclDist, euclROISize);
                  
                //Compute squared diference 
                nppiSub_32f_C1R((Npp32f*) (pSrcBorder +indexSrcImageBaseWOffset + (m_min + dm)), stepBytesSrcBorder, 
                    (Npp32f*) (pSrcBorder + indexSrcImageBase + m_min), stepBytesSrcBorder, pSumSqrDiff, stepSumSqrDiff, euclROISize);
                nppiSqr_32f_C1IR(pSumSqrDiff, stepSumSqrDiff, euclROISize);
                //Compute Squared Sum Difference unsing FilterBox
                nppiFilterBoxBorder_32f_C1R(pSumSqrDiff, stepSumSqrDiff, euclROISize, offset, pEuclDist, stepBytesEuclDist, euclROISize, nROISize, anchor, NPP_BORDER_REPLICATE );        
                nppiSqr_32f_C1IR(pEuclDist, stepBytesEuclDist, euclROISize);
                nppiSqrt_32f_C1IR(pEuclDist, stepBytesEuclDist, euclROISize);
                    
                //Compute weights
                nppiDivC_32f_C1IR((Npp32f) -(2*sigma_r * sigma_r), pEuclDist, stepBytesEuclDist, euclROISize);
                nppiExp_32f_C1IR(pEuclDist, stepBytesEuclDist, euclROISize);          

                //Performing filtering
                nppiMul_32f_C1R(pEuclDist, stepBytesEuclDist, (Npp32f*) (pUSMImage +indexUSMImageBaseWOffset + (m_min + dm)), stepBytesUSM, pTmp, stepBytesTmp, euclROISize);
                //Exploiting weights symmetry
                nppiMul_32f_C1R(pEuclDist, stepBytesEuclDist, (Npp32f*) (pUSMImage + indexUSMImageBase + m_min), stepBytesUSM, pTmp2, stepBytesTmp, euclROISize);
                      
                //Accumulate signal and weights
                nppiAdd_32f_C1IR(pTmp, stepBytesTmp, (Npp32f*) (pThreadDst + indexDstBase + m_min), stepBytesThreadDst, euclROISize);
                nppiAdd_32f_C1IR(pEuclDist, stepBytesEuclDist, (Npp32f*) (pThreadWeightsAcumm + indexWeightsAcummBase + m_min), stepBytesThreadWeights, euclROISize);
                //Accumulate signal and weights 
                nppiAdd_32f_C1IR(pTmp2, stepBytesTmp, (Npp32f*) (pThreadDst + indexDstBaseWOffset + (m_min + dm)), stepBytesThreadDst, euclROISize);
                nppiAdd_32f_C1IR(pEuclDist, stepBytesEuclDist, &pThreadWeightsAcumm[indexWeightsAcummBaseWOffset + (m_min + dm)], stepBytesThreadWeights, euclROISize);

            }

            else if (dn==0 && dm==0)
            {   
                //Acummulate weights and filter result
                nppiAddC_32f_C1IR((Npp32f) 1.0f, pThreadWeightsAcumm, stepBytesThreadWeights, imageSize );
                nppiAdd_32f_C1IR(pUSMImage, stepBytesUSM, pThreadDst, stepBytesThreadDst, imageSize);
            }
        
 
            nppiAdd_32f_C1IR(pThreadWeightsAcumm, stepBytesThreadWeights, pWeightsAcumm, stepBytesWeightsAcumm, imageSize);
            nppiAdd_32f_C1IR(pThreadDst, stepBytesThreadDst, pDst, stepBytesDst, imageSize);
        
        }
    }
    
    //Threshold image
    nppiThreshold_32f_C1IR(pWeightsAcumm, stepBytesWeightsAcumm, imageSize, 1e-20f,NPP_CMP_LESS);
    //Normalize
    nppiDiv_32f_C1IR(pWeightsAcumm, stepBytesWeightsAcumm, pDst, stepBytesDst, imageSize);
    cudaProfilerStop();
    //Free resources
    nppiFree(pChunkMem);
    nppiFree(pChunkMem2);
    nppiFree(pWeightsAcumm);

    return 1;
    
}
