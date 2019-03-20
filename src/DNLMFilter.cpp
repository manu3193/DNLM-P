
#include "DNLMFilter.hpp"

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

   
    //Variable to store status
    Npp32f *pEuclDist;
    Npp8u *pNormBuffer, *pSumBuffer;
    int stepBytesEuclDist = 0;
    int normBufferSize = 0;
    int sumBufferSize = 0;
    int status = 0;
    //Variable to store summation result of filter response dor normalization
    Npp64f *pDoubleArray, *pSumExpTerm , *pFilterResult , *pEuclDistResult;
    Npp32f *pFloatArray, *pSumExpTerm32f, *pFilterResult32f, *pEuclDistResult32f;
    //Allocate memory for sqrtDist matrix
    pEuclDist = nppiMalloc_32f_C1(windowSize.width, windowSize.height, &stepBytesEuclDist);
    if (pEuclDist == NULL) cout << "error allocating pEuclDist" << endl;        
    //Compute buffer size 
    status = nppiNormDiffL2GetBufferHostSize_32f_C1R(neighborhoodSize, &normBufferSize);
    if (status !=0) cout << "error computing normdiff buffer size" << endl;
    status = nppiSumGetBufferHostSize_32f_C1R(windowSize, &sumBufferSize);
    if (status !=0) cout << "error computing sum buffer size" << endl;
    pNormBuffer = nppsMalloc_8u(normBufferSize);
    if (pNormBuffer ==NULL) cout << "error allocating pNormBuffer" << endl;
    pSumBuffer = nppsMalloc_8u(sumBufferSize);
    if (pSumBuffer ==NULL) cout << "error allocating pSumBuffer" << endl;
    //Allocate array of doubles to store pEuclDistResult, pFilterResult and pSumExpTerm
    pDoubleArray = nppsMalloc_64f(3);
    pEuclDistResult = (Npp64f*) (pDoubleArray);
    pSumExpTerm = (Npp64f*) (pDoubleArray + 1);
    pFilterResult= (Npp64f*) (pDoubleArray + 2);
    
    pFloatArray = nppsMalloc_32f(3);
    pEuclDistResult32f = (Npp32f*) (pFloatArray);
    pSumExpTerm32f = (Npp32f*) (pFloatArray + 1);
    pFilterResult32f= (Npp32f*) (pFloatArray + 2);
    cout << "image size "<< imageSize.width<<"x"<<imageSize.height<<endl;
    cout << "window size "<< windowSize.width<<"x"<<windowSize.height<<endl;
    cout << "neighborhood size "<<neighborhoodSize.width<<"x"<<neighborhoodSize.height<<endl;

    for (int j = 0; j < imageSize.height; j++)
    {
    	//Compute window row boundary
    	const int j_w = max(j-windowRadius, neighborRadius);
    	//Compute neighborhood row boundary
    	const int j_n = max(j-neighborRadius, 0);

        for (int i = 0; i < imageSize.width; i++)
        {
            //Compute window col boundary
            const int i_w = max(i - windowRadius, neighborRadius);
            //Compute neighborhood col boundary
            const int i_n = max(i - neighborRadius, 0);
            //Compute base address for each array
            const int indexPdstBase = j*(stepBytesDst/sizeof(Npp32f));
            const int indexWindowStartBase = j_w*(stepBytesSrcBorder/sizeof(Npp32f));
            const int indexNeighborIJBase = j_n*(stepBytesSrcBorder/sizeof(Npp32f));
            //OLD const int indexUSMWindowBase = j_w*(stepBytesUSM/sizeof(Npp32f));
            //Compute start pointer for each array
            const Npp32f *pWindowStart = (Npp32f*) (pSrcBorder+indexWindowStartBase + i_w);
            const Npp32f *pNeighborhoodStartIJ = (Npp32f*) (pSrcBorder + indexNeighborIJBase + i_n);
            //OLD const Npp32f *pUSMWindowStart = (Npp32f*) (pUSMImage + indexUSMWindowBase + i_w);

            int n,m;    
            for (n = 0; n < windowSize.height; n++)
            {
            	//Compute neighborhood row boundary
            	const int n_min = j_w - neighborRadius + n;
            	//Compute base address for each array
                const int indexEuclDistBase = n * (stepBytesEuclDist/sizeof(Npp32f));
                const int indexNeighborNMBase = n_min * (stepBytesSrcBorder/sizeof(Npp32f));

                    
                for (m = 0; m < windowSize.width; m++)
                {
               	    const int m_min = i_w - neighborRadius + m;
                    const Npp32f *pNeighborhoodStartNM = (Npp32f*) (pSrcBorder + indexNeighborNMBase + m_min);
                    status = nppiNormDiff_L2_32f_C1R(pNeighborhoodStartNM, stepBytesSrcBorder, pNeighborhoodStartIJ, stepBytesSrcBorder, neighborhoodSize, pEuclDistResult, pNormBuffer);
                    if (status !=NPP_SUCCESS) cout << " error normDiff "<< status<<endl;
                    status = nppsConvert_64f32f(pEuclDistResult, (Npp32f*) (pEuclDist+indexEuclDistBase + m), 1);
                    if (status !=NPP_SUCCESS) cout << " error converting 64f to 32f " << status << endl;
                }
            }
            status = nppiDivC_32f_C1IR((Npp32f) -(sigma_r * sigma_r), pEuclDist, stepBytesEuclDist, windowSize);
            if (status !=NPP_SUCCESS) cout << " error div " << status << endl;
            status = nppiExp_32f_C1IR(pEuclDist, stepBytesEuclDist, windowSize);
            if (status !=NPP_SUCCESS) cout << " error exp " << status << endl;
            status = nppiSum_32f_C1R(pEuclDist, stepBytesEuclDist, windowSize, pSumBuffer, pSumExpTerm);
            if (status !=NPP_SUCCESS) cout << " error sum " << status << endl;
            //OLD status = nppiMul_32f_C1IR(pUSMWindowStart, stepBytesUSM, pEuclDist, stepBytesEuclDist, windowSize);
            status = nppiMul_32f_C1IR(pWindowStart, stepBytesSrcBorder, pEuclDist, stepBytesEuclDist, windowSize);
            if (status !=NPP_SUCCESS) cout << " error mul " << status << endl;
            status = nppiSum_32f_C1R(pEuclDist, stepBytesEuclDist, windowSize, pSumBuffer, pFilterResult);
            if (status !=NPP_SUCCESS) cout << " error sum2 " << status << endl;
            status = nppsDiv_64f_I(pSumExpTerm, pFilterResult, 1);
            if (status !=NPP_SUCCESS) cout << " error divs " << status << endl;
            status = nppsConvert_64f32f(pFilterResult, pFilterResult32f, 1);
            if (status !=NPP_SUCCESS) cout << " error convert 64f to 32f " << status << endl;
            status = nppsCopy_32f((Npp32f*) pFilterResult32f, (Npp32f*) (pDst +indexPdstBase+i), 1);
            if (status !=NPP_SUCCESS) cout << " error set2 " << status << endl;
        }
    }

    nppiFree(pEuclDist);
    nppsFree(pNormBuffer);
    nppsFree(pSumBuffer);
    nppsFree(pDoubleArray);
    nppsFree(pFloatArray);
    return 1;
}
