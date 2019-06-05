#include <algorithm> 
#include <cmath>

void DNLM_OpenACC(const float* pSrcBorder, int stepBytesSrcBorder, const float* pSqrIntegralImage, int stepBytesSqrIntegral, float* pDst, int stepBytesDst, int windowRadius, int neighborRadius, int imageWidth, int imageHeight, int windowWidth, int windowHeight, int neighborWidth, int neighborHeight, float sigma_r)

{

    //Array to store window matrix for euclidean distance
    float * restrict pEuclDist = (float*) malloc(windowHeight * windowWidth * sizeof(float));
    float * restrict pWindowIJCorr = (float*) malloc(windowHeight * windowWidth * sizeof(float)); 
    //Region de datos privada peucldist (create)
    //Nivel de paralelismo gangs collapse
    #pragma acc data deviceptr(pSrcBorder[(windowHeight+2*(windowRadius+neighborRadius))*(windowWidth+2*(windowRadius+neighborRadius))], pSqrIntegralImage[(windowHeight+2*(windowRadius+neighborRadius)+1)*(windowWidth+2*(windowRadius+neighborRadius)+1)], pDst[imageHeight*imageWidth]) create(pEuclDist[0:windowHeight*windowWidth], pWindowIJCorr[windowHeight*windowWidth]) 
    //#pragma acc data present(pSrcBorder[0:(imageWidth+windowRadius+neighborRadius)+stepBytesSrcBorder/sizeof(float)*(imageHeight+windowRadius+neighborRadius)],  pSqrIntegralImage[(windowHeight+2*(windowRadius+neighborRadius)+1)*(windowWidth+2*(windowRadius+neighborRadius)+1)], pDst[0:imageWidth+stepBytesDst/sizeof(float)*imageHeight]) create(pEuclDist[0:windowHeight*windowWidth], pWindowIJCorr[windowHeight*windowWidth])
    {
        #pragma acc parallel 
        {   
    	    #pragma acc loop collapse(2)  private(pEuclDist[0:windowHeight*windowWidth], pWindowIJCorr[0:windowHeight*windowWidth])   
            for(int j = 0; j < imageHeight; j++)
    	    {
               
                for (int i = 0; i < imageWidth; i++)
                {
                    //Compute base address for each array
                    const int indexPdstBase = j * (stepBytesDst/sizeof(float));
                    const int indexWindowStartBase = j * stepBytesSrcBorder/sizeof(float);
                    const int indexNeighborIJBase = (j + windowRadius) * stepBytesSrcBorder/sizeof(float);
                    const int indexIINeighborIJBase = (j + windowRadius) * stepBytesSqrIntegral/sizeof(float);
                    const int indexIINeighborIJBaseWOffset = (j + windowRadius + neighborWidth) * stepBytesSqrIntegral/sizeof(float);
                    //Get sum area of neighborhood IJ
                    const float sqrSumIJNeighborhood = pSqrIntegralImage[indexIINeighborIJBaseWOffset + (i + windowRadius + neighborWidth)]
                                                    + pSqrIntegralImage[indexIINeighborIJBase + (i + windowRadius)] 
                                                    - pSqrIntegralImage[indexIINeighborIJBase + (i + windowRadius + neighborWidth)]
                                                    - pSqrIntegralImage[indexIINeighborIJBaseWOffset + (i + windowRadius)];
                    //Compute start pointer for each array
                    const float * restrict pWindowStart = (float*) &pSrcBorder[indexWindowStartBase + i];
                    const float * restrict pNeighborhoodStartIJ = (float*) &pSrcBorder[indexNeighborIJBase + i + windowRadius];
                    
                    //Compute window correlation with IJ neighborhood
                    #pragma acc loop collapse(2)
                    for(int row_w = 0; row_w < windowHeight; row_w++)
                    {
                        for(int col_w = 0; col_w < windowWidth; col_w++)
                        {
                            float neighborCorrSum = 0;
                            #pragma acc loop vector reduction(+:neighborCorrSum)
                            for(int row_n = 0; row_n < neighborHeight; row_n++)
                            {
                                for(int col_n = 0; col_n < neighborWidth; col_n++)
                                {
                                    neighborCorrSum += pWindowStart[(col_w+col_n)+((row_w+row_n)*stepBytesSrcBorder/sizeof(float))] * pNeighborhoodStartIJ[col_n + (row_n * stepBytesSrcBorder/sizeof(float))]; 
                                }
                            }
                        pWindowIJCorr[col_w + row_w * windowWidth] = neighborCorrSum;
                        }
                    }
                   
                    //#pragma acc loop
                    #pragma acc loop  collapse(2) private(sqrSumIJNeighborhood, i, j)
                    for (int n = 0; n < windowHeight; n++)
                    {

                        for (int m = 0; m < windowWidth; m++)
                        {
                            //Compute base address for each array
                            const int indexIINeighborMNBase = (j + n) * stepBytesSqrIntegral/sizeof(float);
                            const int indexIINeighborMNBaseWOffset = (j + n + neighborWidth) * stepBytesSqrIntegral/sizeof(float);
                            
                            const float sqrSumMNNeighborhood = pSqrIntegralImage[indexIINeighborMNBaseWOffset + (i + m  + neighborWidth)]
                                                            + pSqrIntegralImage[indexIINeighborMNBase + (i + m )] 
                                                            - pSqrIntegralImage[indexIINeighborMNBase + (i + m  + neighborWidth)]
                                                            - pSqrIntegralImage[indexIINeighborMNBaseWOffset + (i + m )];
                            //#pragma acc loop seq
                            pEuclDist[n*windowWidth + m]= sqrSumIJNeighborhood + sqrSumMNNeighborhood -2*pWindowIJCorr[n*windowWidth + m];
                        }
                    }

                    float sumExpTerm = 0;
                    #pragma acc loop collapse(2) reduction(+:sumExpTerm)
                    for(int row = 0; row < windowHeight; row++)
                    {
                        for(int col = 0; col < windowWidth; col++)
                        {
                            pEuclDist[col + row * windowWidth] = pEuclDist[col + row * windowWidth] *  -1/(2*(sigma_r * sigma_r));
                            pEuclDist[col + row * windowWidth] = expf(pEuclDist[col + row * windowWidth]);
                            sumExpTerm += pEuclDist[col + row * windowWidth];
                        }
                    }

                    float filterResult = 0;            
                    //Reduce(+) 
                    #pragma acc loop collapse(2) reduction(+:filterResult) 
                    for(int row = 0; row < windowHeight; row++)
                    {
                        for(int col = 0; col < windowWidth; col++)
                        {
                            pEuclDist[col + row * windowWidth] = pEuclDist[col + row * windowWidth] * pWindowStart[(col+neighborRadius) + (row+neighborRadius) * stepBytesSrcBorder/sizeof(float)];
                            filterResult += pEuclDist[col + row * windowWidth];                    
                        }
                    }
                    pDst[indexPdstBase + i] = filterResult/sumExpTerm;
                }   
            }
        }   
    }
} 
