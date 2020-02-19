
#include <algorithm> 
#include <cmath>
#include <accelmath.h>


void DNLM_OpenACC(const float* pSrcBorder, int stepBytesSrcBorder, float* pDst, int stepBytesDst, int windowRadius, int neighborRadius, int imageWidth, int imageHeight, int windowWidth, int windowHeight, int neighborWidth, int neighborHeight, float sigma_r)

{

    //Array to store window matrix for euclidean distance
    float * restrict pEuclDist = (float*) malloc(windowHeight * windowWidth * sizeof(float));
    const int stepBSB = stepBytesSrcBorder/sizeof(float);
    const int stepBD = stepBytesDst/sizeof(float);
    
    #pragma acc data deviceptr(pSrcBorder[(imageHeight+2*(windowRadius+neighborRadius))*(imageWidth+2*(windowRadius+neighborRadius))], pDst[imageHeight*imageWidth])  
    {
        #pragma acc parallel   
        {   
            #pragma acc loop gang collapse(2) private(pEuclDist[0:windowHeight*windowWidth]) 
            for(int j = 0; j < imageHeight; j++)
            {
                for (int i = 0; i < imageWidth; i++)
                {
                    //Compute base address for each array
                    const int indexPdstBase = j * stepBD;
                    const int indexWindowStartBase = j * stepBSB;
                    const int indexNeighborIJBase = (j + windowRadius) * stepBSB;
                    //Compute start pointer for each array
                    const float * restrict pWindowStart = (float*) &pSrcBorder[indexWindowStartBase + i];
                    const float * restrict pNeighborhoodStartIJ = (float*) &pSrcBorder[indexNeighborIJBase + i + windowRadius];
                    #pragma acc loop vector collapse(2) 
                    for (int n = 0; n < windowHeight; n++)
                    {
                        for (int m = 0; m < windowWidth; m++)
                        {
                            //Compute base address for each array
                            const int indexNeighborNMBase = n * stepBSB;
                            const float * restrict pNeighborhoodStartNM = (float*) &pWindowStart[indexNeighborNMBase + m];
                            float sumNeighborhoodDiff = 0;
                            for(int row = 0; row < neighborHeight; row++)
                            {
                                for(int col = 0; col < neighborWidth; col++)
                                {
                                    float diff = pNeighborhoodStartNM[row*stepBSB + col] - pNeighborhoodStartIJ[row*stepBSB + col];
                                    diff = diff*diff;
                                    sumNeighborhoodDiff += diff;
                                }
                            }
                            pEuclDist[n*windowWidth + m]= sqrt(sumNeighborhoodDiff);
                        }
                    }
                    float sumExpTerm = 0;
                    #pragma acc loop vector collapse(2) reduction(+:sumExpTerm) 
                    for(int row = 0; row < windowHeight; row++)
                    {
                        for(int col = 0; col < windowWidth; col++)
                        {
                            pEuclDist[col + row * windowWidth] = pEuclDist[col + row * windowWidth] *  -1/(sigma_r * sigma_r);
                            pEuclDist[col + row * windowWidth] = expf(pEuclDist[col + row * windowWidth]);
                            sumExpTerm += pEuclDist[col + row * windowWidth];
                        }
                    }
                    float filterResult = 0;            
                    #pragma acc loop collapse(2) reduction(+:filterResult) 
                    for(int row = 0; row < windowHeight; row++)
                    {
                        for(int col = 0; col < windowWidth; col++)
                        {
                            pEuclDist[col + row * windowWidth] = pEuclDist[col + row * windowWidth] * pWindowStart[(col+neighborRadius) + (row+neighborRadius) * stepBSB];
                            filterResult += pEuclDist[col + row * windowWidth];                    
                        }
                    }
                    pDst[indexPdstBase + i] = filterResult/sumExpTerm;
                }   
            }
        } 
    }
} 
