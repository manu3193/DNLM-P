#include <stdlib.h> 
#include <accelmath.h>

void DNLM_OpenACC(const float* pSrcBorder, int stepBytesSrcBorder, const float* pSqrIntegralImage, int stepBytesSqrIntegral, float* pDst, int stepBytesDst, int windowRadius, int neighborRadius, int imageWidth, int imageHeight, int windowWidth, int windowHeight, int neighborWidth, int neighborHeight, float sigma_r)

{

    //Array to store window matrix for euclidean distance
    float * restrict pEuclDist = (float*) malloc(windowHeight * windowWidth * sizeof(float));
    float * restrict pWindowIJCorr = (float*) malloc(windowHeight * windowWidth * sizeof(float));
    const int stepBD = stepBytesDst/sizeof(float);
    const int stepBSI = stepBytesSqrIntegral/sizeof(float);
    const int stepBSB = stepBytesSrcBorder/sizeof(float); 
    //Region de datos privada peucldist (create)
    //Nivel de paralelismo gangs collapse
    #pragma acc data deviceptr(pSrcBorder, pSqrIntegralImage, pDst)
    {
        #pragma acc parallel  
        {
    	    #pragma acc loop gang collapse(2) private(pEuclDist[0:windowHeight*windowWidth], pWindowIJCorr[0:windowHeight*windowWidth])     
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
                    const float * restrict pWindowStart = (float*) &pSrcBorder[indexWindowStartBase + i];
                    const float * restrict pNeighborhoodStartIJ = (float*) &pSrcBorder[indexNeighborIJBase + i + windowRadius];
                    
                    //Compute window correlation with IJ neighborhood
                    #pragma acc loop vector collapse(2)
                    for(int row_w = 0; row_w < windowHeight; row_w++)
                    {
                        for(int col_w = 0; col_w < windowWidth; col_w++)
                        {
                            float neighborCorrSum = 0;
                            for(int row_n = 0; row_n < neighborHeight; row_n++)
                            {
                                for(int col_n = 0; col_n < neighborWidth; col_n++)
                                {
                                    neighborCorrSum += pWindowStart[(col_w+col_n)+((row_w+row_n)*stepBSB)] * pNeighborhoodStartIJ[col_n + (row_n * stepBSB)];
                                }
                            }
                            pWindowIJCorr[col_w + row_w * windowWidth] = neighborCorrSum; 
                       }
                    }
                   
                    //#pragma acc loop
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
                            pEuclDist[n*windowWidth + m]= sqrSumIJNeighborhood + sqrSumMNNeighborhood -2*pWindowIJCorr[n*windowWidth + m];
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
} 
