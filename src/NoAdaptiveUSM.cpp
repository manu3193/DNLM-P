#include <NoAdaptiveUSM.hpp>




int NoAdaptiveUSM::noAdaptativeUSM(const Ipp32f* pSrc, Ipp32f* pDst, float lambda, IppiMaskSize mask){
	
}





int NoAdaptiveUSM::generateLoGKernel(int size, double sigma, Ipp32f* pKernel ){
	
	Ipp32f min, max, termSum;
	int halfSize  = (size - 1) / 2;
	int stepSize32f = 0;
	Ipp32f std2 = (Ipp32f) sigma*sigma;

	//Allocate memory for matrix to store (x*x + y*y) term. 
	Ipp32f* radXY =  ippiMalloc_32f_C1(size, size, &stepSize32f);
	//Allocate memory for matrix to store gaussian term. 
	Ipp32f* expRes =  ippiMalloc_32f_C1(size, size, &stepSize32f);

	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size; ++j)
		{
			radXY[i*stepSize32f + j] = (Ipp32f) ((i - halfSize) * (i - halfSize) + (j - halfSize) * (j - halfSize));
			expRes[i*stepSize32f + j] = (Ipp32f) ();
		}
	}

}