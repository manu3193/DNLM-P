#include <stdlib.h> 
#include <stdio.h>
#include <accelmath.h>
#include <complex.h>
#include "fft.h"

unsigned int next_pow2(unsigned int);



void DNLM_OpenACC(const float* pSrcBorder, int stepBytesSrcBorder, const float* pSqrIntegralImage, int stepBytesSqrIntegral, float* pDst, int stepBytesDst, int windowRadius, int neighborRadius, int imageWidth, int imageHeight, int windowWidth, int windowHeight, int neighborWidth, int neighborHeight, float sigma_r)

{
    int extWindowWidth, extWindowHeight = windowWidth + 2 * neighborRadius;
    int  paddedSize = (int) next_pow2((unsigned int) extWindowWidth);
    //Array to store window matrix for euclidean distance
    float * restrict pEuclDist = (float*) malloc(windowHeight * windowWidth * sizeof(float));

    double * restrict pWindowIJCorr = (double*) malloc(paddedSize * paddedSize * sizeof(double));
    double * restrict pNeighborhoodIJPadded = (double *) malloc(paddedSize * paddedSize * sizeof(double));
    double * restrict pWindowPadded = (double *) malloc(paddedSize * paddedSize * sizeof(double));
    double _Complex * restrict pNeighborhoodIJFreq = (double _Complex *) malloc(paddedSize * paddedSize * sizeof(double _Complex));
    double _Complex * restrict pWindowFreq = (double _Complex *) malloc(paddedSize * paddedSize * sizeof(double _Complex));
    double _Complex * restrict pWindowIJCorrFreq = (double _Complex*) malloc(paddedSize * paddedSize * sizeof(double _Complex));
    double _Complex * restrict pBuffer = (double _Complex*) malloc(paddedSize * paddedSize * sizeof(double _Complex));

    const int stepBD = stepBytesDst/sizeof(float);
    const int stepBSI = stepBytesSqrIntegral/sizeof(float);
    const int stepBSB = stepBytesSrcBorder/sizeof(float); 
    //Region de datos privada peucldist (create)
    //Nivel de paralelismo gangs collapse
    #pragma acc data deviceptr(pSrcBorder, pSqrIntegralImage, pDst) 
    {
        #pragma acc parallel private(pEuclDist[0:windowHeight*windowWidth], pWindowIJCorr[0:paddedSize*paddedSize], pNeighborhoodIJPadded[0:paddedSize*paddedSize], pWindowPadded[0:paddedSize*paddedSize], pNeighborhoodIJFreq[0:paddedSize*paddedSize], pWindowFreq[0:paddedSize*paddedSize], pWindowIJCorrFreq[0:paddedSize*paddedSize], pBuffer[0:paddedSize*paddedSize])

        {
            #pragma acc loop gang collapse(2) 
            for(int j = 0; j < imageHeight; j++)
            {
                for (int i = 0; i < imageWidth; i++)
                {
                    //Compute base address for each array
                    const int indexPdstBase = j * (stepBD);
                    const int indexWindowStartBase = j * stepBSB;
                    const int indexNeighborIJBase = (j + windowRadius) * stepBSB;
                    const int indexIINeighborIJBase = (j + windowRadius) * stepBSI;
                    const int indexIINeighborIJBaseWOffset = (j + windowRadius + neighborWidth) * stepBSI;
                    //Get sum area of neighborhood IJ
                    const float sqrSumIJNeighborhood = pSqrIntegralImage[indexIINeighborIJBaseWOffset + (i + windowRadius + neighborWidth)]
                                                    + pSqrIntegralImage[indexIINeighborIJBase + (i + windowRadius)] 
                                                    - pSqrIntegralImage[indexIINeighborIJBase + (i + windowRadius + neighborWidth)]
                                                    - pSqrIntegralImage[indexIINeighborIJBaseWOffset + (i + windowRadius)];
                    //Compute start pointer for each array
                    float * restrict pWindowStart = (float*) &pSrcBorder[indexWindowStartBase + i];
                    float * restrict pNeighborhoodStartIJ = (float*) &pSrcBorder[indexNeighborIJBase + i + windowRadius];
		    
                    //Pad window with 0's to match vector length                    
                    zeroPadding(pWindowStart, stepBSB, pWindowPadded, extWindowWidth, extWindowHeight, paddedSize);
                    //Compute FFT of padded window
                    compute2D_R2CFFT(pWindowPadded, paddedSize, pWindowFreq, paddedSize, paddedSize, pBuffer);

                    //Compute window correlation with IJ neighborhood
                    zeroPadding(pNeighborhoodStartIJ, stepBSB, pNeighborhoodIJPadded, neighborWidth, neighborHeight, paddedSize);    
                    compute2D_R2CFFT(pNeighborhoodIJPadded, paddedSize, pNeighborhoodIJFreq, paddedSize, paddedSize, pBuffer);
                    computeCorr(pWindowFreq, pNeighborhoodIJFreq, pWindowIJCorrFreq,  paddedSize, paddedSize);
                    compute2D_C2RInvFFT(pWindowIJCorrFreq, paddedSize, pWindowIJCorr, paddedSize, paddedSize, pBuffer); 
                    
                    #pragma acc loop vector collapse(2) 
                    for (int n = 0; n < windowHeight; n++)
                    {
                        for (int m = 0; m < windowWidth; m++)
                        {
                            //Compute base address for each array
                            const int indexIINeighborMNBase = (j + n) * stepBSI;
                            const int indexIINeighborMNBaseWOffset = (j + n + neighborWidth) * stepBSI;
                            
                            const float sqrSumMNNeighborhood = pSqrIntegralImage[indexIINeighborMNBaseWOffset + (i + m  + neighborWidth)]
                                                            + pSqrIntegralImage[indexIINeighborMNBase + (i + m )] 
                                                            - pSqrIntegralImage[indexIINeighborMNBase + (i + m  + neighborWidth)]
                                                            - pSqrIntegralImage[indexIINeighborMNBaseWOffset + (i + m )];
                            pEuclDist[n*windowWidth + m]= sqrSumIJNeighborhood + sqrSumMNNeighborhood -2* (float) pWindowIJCorr[(n+neighborRadius)*windowWidth + (m+neighborRadius)];
                        }
                    }

                    float sumExpTerm = 0;
                    #pragma acc loop vector collapse(2) reduction(+:sumExpTerm)
                    for(int row = 0; row < windowHeight; row++)
                    {
                        for(int col = 0; col < windowWidth; col++)
                        {
                            float filterWeight = expf(pEuclDist[col + row * windowWidth] *  1/(-2*(sigma_r * sigma_r)));
                            pEuclDist[col + row * windowWidth] = filterWeight;
                            sumExpTerm += filterWeight;
                        }
                    }

                    float filterResult = 0;            
                    //Reduce(+) 
                    #pragma acc loop vector collapse(2) reduction(+:filterResult) 
                    for(int row = 0; row < windowHeight; row++)
                    {
                        for(int col = 0; col < windowWidth; col++)
                        {
                            float filterRes = pEuclDist[col + row * windowWidth] * pWindowStart[(col+neighborRadius) + (row+neighborRadius) * stepBSB];
                            filterResult += filterRes;                    
                        }
                    }
                    pDst[indexPdstBase + i] = filterResult/sumExpTerm;
                }   
            }
        }   
    }

    free(pEuclDist);
    free(pWindowIJCorr);
    free(pNeighborhoodIJPadded);
    free(pWindowPadded);
    free(pNeighborhoodIJFreq);
    free(pWindowFreq);
    free(pWindowIJCorrFreq);
    free(pBuffer);
}

unsigned int next_pow2(unsigned int n){
    n--; 
    n |= n >> 1; 
    n |= n >> 2; 
    n |= n >> 4; 
    n |= n >> 8; 
    n |= n >> 16; 
    n++; 
    return n; 
} 
