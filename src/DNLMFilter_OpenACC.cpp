#include <algorithm> 
#include <cmath>

void DNLM_OpenACC(const float* pSrcBorder, int stepBytesSrcBorder, const float* pSqrIntegralImage, int stepBytesSqrIntegral, float* pDst, int stepBytesDst, int windowRadius, int neighborRadius, int imageWidth, int imageHeight, int windowWidth, int windowHeight, int neighborWidth, int neighborHeight, float sigma_r)

{

    //Array to store window matrix for euclidean distance
    float * restrict pEuclDist = (float*) malloc(windowHeight * windowWidth * sizeof(float));
    float * restrict pNeighborhoodDiff = (float*) malloc(neighborHeight * neighborWidth * sizeof(float)); 
    //Region de datos privada peucldist (create)
    //Nivel de paralelismo gangs collapse
    #pragma acc data deviceptr(pSrcBorder[(windowHeight+2*(windowRadius+neighborRadius))*(windowWidth+2*(windowRadius+neighborRadius))], pDst[imageHeight*imageWidth]) create(pEuclDist[0:windowHeight*windowWidth], pNeighborhoodDiff[0:neighborHeight*neighborWidth]) 
    //#pragma acc data present(pSrcBorder[0:(imageWidth+windowRadius+neighborRadius)+stepBytesSrcBorder/sizeof(float)*(imageHeight+windowRadius+neighborRadius)], pDst[0:imageWidth+stepBytesDst/sizeof(float)*imageHeight]) create(pEuclDist[0:windowHeight*windowWidth], pNeighborhoodDiff[0:neighborHeight*neighborWidth])
    {
        #pragma acc parallel vector_length(64)
        {   
    	    #pragma acc loop independent collapse(2)  private(pEuclDist[0:windowHeight*windowWidth])   
            for(int j = 0; j < imageHeight; j++)
    	    {
               
                for (int i = 0; i < imageWidth; i++)
                {
                    //Compute base address for each array
                    const int indexPdstBase = j * (stepBytesDst/sizeof(float));
                    const int indexWindowStartBase = j * stepBytesSrcBorder/sizeof(float);
                    const int indexNeighborIJBase = (j + windowRadius) * stepBytesSrcBorder/sizeof(float);
                    //Compute start pointer for each array
                    const float * restrict pWindowStart = (float*) &pSrcBorder[indexWindowStartBase + i];
                    const float * restrict pNeighborhoodStartIJ = (float*) &pSrcBorder[indexNeighborIJBase + i + windowRadius];
                    //OLD const Npp32f *pUSMWindowStart = (Npp32f*) (pUSMImage + indexUSMWindowBase + i_w);
                    //#pragma acc loop independent collapse(2)  private(pNeighborhoodStartIJ[0:neighborHeight*neighborWidth], pWindowStart[0:windowHeight*windowWidth]) 
                    //#pragma acc loop seq
                    #pragma acc loop independent  collapse(2) private(pNeighborhoodDiff[0:neighborHeight*neighborWidth])
                    for (int n = 0; n < windowHeight; n++)
                    {

                        for (int m = 0; m < windowWidth; m++)
                        {
                            //Compute base address for each array
                            const int indexNeighborNMBase = n * (stepBytesSrcBorder/sizeof(float));
                            const float * restrict pNeighborhoodStartNM = (float*) &pWindowStart[indexNeighborNMBase + m];

                            float sumNeighborhoodDiff = 0;
                            //loop reduce(+)
                            //#pragma acc loop seq
                            #pragma acc loop  independent collapse(2) reduction(+:sumNeighborhoodDiff) 
                            for(int row = 0; row < neighborHeight; row++)
                            {
                                for(int col = 0; col < neighborWidth; col++)
                                {
                                    pNeighborhoodDiff[row*neighborWidth + col] = pNeighborhoodStartNM[row*stepBytesSrcBorder/sizeof(float) + col] - pNeighborhoodStartIJ[row*stepBytesSrcBorder/sizeof(float) + col];
                                    pNeighborhoodDiff[row*neighborWidth + col] = pow(pNeighborhoodDiff[row*neighborWidth + col], 2);
                                    sumNeighborhoodDiff += pNeighborhoodDiff[row*neighborWidth + col];
                                }
                            }
                        pEuclDist[n*windowWidth + m]= sqrt(sumNeighborhoodDiff);
                        }
                    }

                    float sumExpTerm = 0;
                    #pragma acc loop independent collapse(2) reduction(+:sumExpTerm)
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
                    //Reduce(+) 
                    #pragma acc loop independent collapse(2) reduction(+:filterResult) 
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
