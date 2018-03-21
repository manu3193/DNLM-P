#include <NoAdaptiveUSM.hpp>




int NoAdaptiveUSM::noAdaptiveUSM(const Ipp32f* pSrc, Ipp32f* pDst, IppiSize roiSize, float lambda, int kernelLen){
	

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
	//Allocate memory for matrix to store laplacian term. 
	Ipp32f* pLaplTerm =  ippiMalloc_32f_C1(size, size, &stepSize32f);


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

	//Assign computed kernel to dst pointer argument 
	pKernel = pLaplTerm;

	//Release memory
	ippiFree(pRadXY);
	ippiFree(pExpTerm);
	ippiFree(pLaplTerm);

	//Error code handling to be implemented.
	return 1;
}