#include <NoAdaptiveUSM.hpp>




int NoAdaptiveUSM::noAdaptiveUSM(const Ipp32f* pSrc, int stepBytesSrc, Ipp32f* pDst, int stepBytesDst, IppiSize roiSize, float sigma, float lambda,  int kernelLen){
	
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
    //Step in bytes of images
    int stepBytesFiltered = 0;
    int stepBytesAbs = 0;
    //Filtered image min and max values, for normalization
    Ipp32f minSrc, maxSrc, minFilt, maxFilt;

    //Declare array for laplacian kernel
    Ipp32f pKernel[kernelLen * kernelLen];
    //Allocate memory for filtered image
    pFilteredImage = ippiMalloc_32f_C1(roiSize.width, roiSize.height, &stepBytesFiltered); 
	//Allocate memory for filtered abs image
    pFilteredAbsImage = ippiMalloc_32f_C1(roiSize.width, roiSize.height, &stepBytesAbs); 

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
    status = ippiFilterBorder_32f_C1R(pSrc, stepBytesSrc, pFilteredImage, stepBytesFiltered, roiSize, borderType, &borderValue, pSpec, pBuffer); 

    //Normalization
    //Get Src image max and min values
    status = ippiMinMax_32f_C1R(pSrc, stepBytesSrc, roiSize, &minSrc, &maxSrc);
    //Get Filtered image max and min values from abs(pFilteredImage)
    status = ippiAbs_32f_C1R(pFilteredImage, stepBytesFiltered, pFilteredAbsImage, stepBytesAbs, roiSize);
    status = ippiMinMax_32f_C1R(pFilteredAbsImage, stepBytesAbs, roiSize, &minFilt, &maxFilt);
    //Normalize
    Ipp32f normFactor = (Ipp32f) maxSrc / maxFilt;
    status = ippiMulC_32f_C1IR(normFactor, pFilteredImage, stepBytesFiltered, roiSize);

    //Apply USM
    status = ippiMulC_32f_C1IR((Ipp32f) lambda, pFilteredImage, stepBytesFiltered, roiSize);
    status = ippiAdd_32f_C1R(pSrc, stepBytesSrc, pFilteredImage, stepBytesFiltered, pDst, stepBytesDst, roiSize);
    
    //ToDo Error handling
    if (status!=ippStsNoErr)
    {
    	return -1;
    }
    
    

    //Free memory
    ippsFree(pBuffer);
    ippsFree(pSpec);
    //ippiFree(pKernel);
    ippiFree(pFilteredImage);
    ippiFree(pFilteredAbsImage);

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
int NoAdaptiveUSM::generateLoGKernel(int size, float sigma, Ipp32f* pKernel ){
	
	IppStatus status = ippStsNoErr;
	Ipp32f sumExpTerm = 0;
	Ipp64f sumLaplTerm = 0;
	int halfSize  = (size - 1) / 2;
	int stepBytesRadXY = 0;
	int stepBytesExpTerm = 0;
    int stepBytesLaplTerm = 0;
	Ipp32f std2 = (Ipp32f) sigma*sigma;
	Ipp32f expMin, expMax;
	IppiSize roiSize;
    roiSize.width = size;
    roiSize.height = size;

	//Allocate memory for matrix to store (x*x + y*y) term. 
	Ipp32f* pRadXY =  ippiMalloc_32f_C1(size, size, &stepBytesRadXY);
	//Allocate memory for matrix to store exponential term. 
	Ipp32f* pExpTerm =  ippiMalloc_32f_C1(size, size, &stepBytesExpTerm);
	//Copy Dst buffer dir to pointer for laplacian term. 
	Ipp32f* pLaplTerm =  ippiMalloc_32f_C1(size, size, &stepBytesLaplTerm);

	int indexRadXY = 0;
	int indexExpTerm = 0;
	for (int j = 0; j < size; ++j)
	{
		for (int i = 0; i < size; ++i)
		{
			indexRadXY = j*(stepBytesRadXY/sizeof(Ipp32f)) + i;
			indexExpTerm = j*(stepBytesExpTerm/sizeof(Ipp32f)) + i;
			//Compute radial distance term (x*x + y*y) and exponential term
			pRadXY[indexRadXY] = (Ipp32f) ((j - halfSize) * (j - halfSize) + (i - halfSize) * (i - halfSize));
			pExpTerm[indexExpTerm] = (Ipp32f) exp(pRadXY[indexRadXY] / (-2*std2));
			//Store summation of the exponential result to normalize it
			sumExpTerm += pExpTerm[indexExpTerm];
		}
	}


	status = ippiMinMax_32f_C1R(pExpTerm, stepBytesExpTerm, roiSize, &expMin, &expMax);
	status = ippiThreshold_Val_32f_C1IR(pExpTerm, stepBytesExpTerm, roiSize, (Ipp32f) (IPP_EPS23 * expMax), (Ipp32f) 0.0, ippCmpLess);


	if (sumExpTerm != (Ipp32f) 1.0f)
	{
		//Normalize
		status = ippiDivC_32f_C1IR(sumExpTerm, pExpTerm, stepBytesExpTerm, roiSize);
	}

	//Compute laplacian
	status = ippiAddC_32f_C1R(pRadXY, stepBytesRadXY, (Ipp32f) (-2*std2), pLaplTerm, stepBytesLaplTerm, roiSize);
	status = ippiDivC_32f_C1IR((Ipp32f) (std2 * std2), pLaplTerm, stepBytesLaplTerm, roiSize);

	status = ippiMul_32f_C1IR(pExpTerm, stepBytesExpTerm, pLaplTerm, stepBytesLaplTerm, roiSize);
	status = ippiSum_32f_C1R(pLaplTerm, stepBytesLaplTerm, roiSize, &sumLaplTerm, ippAlgHintNone);	
	status = ippiAddC_32f_C1IR((Ipp32f) -sumLaplTerm/(size*size), pLaplTerm, stepBytesLaplTerm, roiSize);

    for (int j = 0; j < size; ++j)
    {
        for (int i = 0; i < size; ++i)
        {
            pKernel[j*(size)+i] = -pLaplTerm[j*(stepBytesLaplTerm/sizeof(Ipp32f)) + i];
        }
    }

	//Release memory
	ippiFree(pRadXY);
	ippiFree(pExpTerm);
    ippiFree(pLaplTerm);


	//Error code handling to be implemented.
	return 1;
}