#include <NoAdaptiveUSM.hpp>




int NoAdaptiveUSM::noAdaptiveUSM(const Ipp32f* pSrc, Ipp32f* pDst, IppiSize roiSize, double sigma, float lambda,  int kernelLen){
	
	if (pSrc == NULL || pDst == NULL)
	{
		//ToDo better error handling
		return -1;
	}

	IppStatus status = ippStsNoErr;
	Ipp32f *pFilteredImage = NULL, *pFilteredAbsImage = NULL;
	Ipp8u *pBuffer = NULL;                /* Pointer to the work buffer */
    IppiFilterBorderSpec* pSpec = NULL;   /* context structure */
    int iTmpBufSize = 0, iSpecSize = 0;   /* Common work buffer size */
    IppiBorderType borderType = ippBorderRepl;
    Ipp32f borderValue = 0.0;
    IppiSize  kernelSize = { kernelLen, kernelLen };
    //Apply USM only in L component of Lab colorspase. Working with grayscale at the momment. 
    int numChannels = 1;
    //Step in bytes of 32f image
    int stepSize32f = 0;
    //Filtered image min and max values, for normalization
    Ipp32f minSrc, maxSrc, minFilt, maxFilt;

    //Allocate memory for laplacian kernel
    Ipp32f* pKernel =  ippiMalloc_32f_C1(kernelLen, kernelLen, &stepSize32f);
    //Allocate memory for filtered image
    pFilteredImage = ippiMalloc_32f_C1(roiSize.width, roiSize.height, &stepSize32f); 
    pFilteredAbsImage = ippiMalloc_32f_C1(roiSize.width, roiSize.height, &stepSize32f); 

    //Generate laplacian of gaussian kernel
    int code =this->generateLoGKernel(kernelLen, sigma, pKernel);

    //Error handling
    if (code != 1)
    {
    	/*handle error*/
    }

    //aplying high pass filter
    //Calculating filter buffer size
    status = ippiFilterBorderGetSize(kernelSize, roiSize, ipp32f, ipp32f, numChannels, &iSpecSize, &iTmpBufSize);

    //Allocating filter buffer and specification
    pSpec = (IppiFilterBorderSpec *)ippsMalloc_8u(iSpecSize);
    pBuffer = ippsMalloc_8u(iTmpBufSize);

    //Initializing filter
    status = ippiFilterBorderInit_32f(pKernel, kernelSize, ipp32f, numChannels, ippRndFinancial, pSpec);
    //Applying filter
    status = ippiFilterBorder_32f_C1R(pSrc, stepSize32f, pFilteredImage, stepSize32f, roiSize, borderType, &borderValue, pSpec, pBuffer); 

    //Normalization
    //Get Src image max and min values
    status = ippiMinMax_32f_C1R(pSrc, stepSize32f, roiSize, &minSrc, &maxSrc);
    //Get Filtered image max and min values from abs(pFilteredImage)
    status = ippiAbs_32f_C1R(pFilteredImage, stepSize32f, pFilteredAbsImage, stepSize32f, roiSize);
    status = ippiMinMax_32f_C1R(pFilteredAbsImage, stepSize32f, roiSize, &minFilt, &maxFilt);
    //Normalize
    Ipp32f normFactor = (Ipp32f) maxSrc / maxFilt;
    status = ippiMulC_32f_C1IR(normFactor, pFilteredImage, stepSize32f, roiSize);

    //Apply USM
    status = ippiMulC_32f_C1IR((Ipp32f) lambda, pFilteredImage, stepSize32f, roiSize);
    status = ippiAdd_32f_C1R(pSrc, stepSize32f, pFilteredImage, stepSize32f, pDst, stepSize32f, roiSize);

    //ToDo Error handling
    //if (status!=ippStsNoErr)
    //{
    //	return -1;
    //}

    return 1;
}





/**
 * @brief      Generate Laplacian of Gaussian kernel for input parameters. 
 *
 * @param[in]  size     The kernel length (number of rows or cols assuming squared kernel)
 * @param[in]  sigma    The std of the kernel
 * @param      pKernel  Pointer to kernel
 *
 * @return     returns simple error code at the momment: 1 if success, any other number means error. 
 */
int NoAdaptiveUSM::generateLoGKernel(int size, double sigma, Ipp32f* pKernel ){
	
	Ipp32f sumExpTerm;
	Ipp64f* pSumLaplTerm;
	int halfSize  = (size - 1) / 2;
	int stepSize32f = 0;
	Ipp64f std2 = (Ipp64f) sigma*sigma;
	Ipp32f *pMin = NULL, *pMax = NULL;
	IppiSize roiSize;
    roiSize.width = size;
    roiSize.height = size;

	//Allocate memory for matrix to store (x*x + y*y) term. 
	Ipp32f* pRadXY =  ippiMalloc_32f_C1(size, size, &stepSize32f);
	//Allocate memory for matrix to store exponential term. 
	Ipp32f* pExpTerm =  ippiMalloc_32f_C1(size, size, &stepSize32f);
	//Copy Dst buffer dir to pointer for laplacian term. 
	Ipp32f* pLaplTerm =  pKernel;


	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size; ++j)
		{
			//Compute radial distance term (x*x + y*y) and exponential term
			pRadXY[i*stepSize32f + j] = (Ipp32f) ((i - halfSize) * (i - halfSize) + (j - halfSize) * (j - halfSize));
			pExpTerm[i*stepSize32f + j] = (Ipp32f) exp(pRadXY[i*stepSize32f + j] / (-2*std2));
			//Store summation of the exponential result to normalize it
			sumExpTerm += pExpTerm[i*stepSize32f + j];
		}
	}

	ippiMinMax_32f_C1R(pExpTerm, stepSize32f, roiSize, pMin, pMax);
	ippiThreshold_Val_32f_C1IR(pExpTerm, stepSize32f, roiSize, (Ipp32f) (ipp_eps52 * (*pMax)), (Ipp32f) 0.0, ippCmpLess);

	if (sumExpTerm != (Ipp32f) 0.0f)
	{
		//Normalize
		ippiDivC_32f_C1IR(sumExpTerm, pExpTerm, stepSize32f, roiSize);
	}

	//Compute laplacian
	ippiAddC_32f_C1R(pRadXY, stepSize32f, (Ipp32f) (-2*std2), pLaplTerm, stepSize32f, roiSize);
	ippiDivC_32f_C1IR((Ipp32f) (std2*std2), pLaplTerm, stepSize32f, roiSize);
	ippiMul_32f_C1IR(pExpTerm, stepSize32f, pLaplTerm, stepSize32f, roiSize);

	ippiSum_32f_C1R(pLaplTerm, stepSize32f, roiSize, pSumLaplTerm, ippAlgHintNone);
	ippiAddC_32f_C1IR((Ipp32f) -(*pSumLaplTerm)/(size*size), pLaplTerm, stepSize32f, roiSize);

	//Release memory
	ippiFree(pRadXY);
	ippiFree(pExpTerm);

	//Error code handling to be implemented.
	return 1;
}