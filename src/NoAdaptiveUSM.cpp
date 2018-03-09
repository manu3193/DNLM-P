#include <NoAdaptiveUSM.hpp>



int NoAdaptiveUSM::NoAdaptativeUSM(){
	this->pKernel = NULL;
	this->kernelStepSize = 0;
	this->kernelSize = 0;
}

int NoAdaptiveUSM::noAdaptativeUSM(const Ipp32f* pSrc, Ipp32f* pDst, float lambda, IppiMaskSize mask){
	
}





int NoAdaptiveUSM::generateLoGKernel(int size, float sigma, Ipp32f* pKernel ){
	//Variables used to especify size of kernel
    IppiSize roi;
    roi.width = size;
    roi.height = size;
    //
	//Allocate memory to store generated kernel
	pKernel = ippiMalloc_32f_C1(roi.width, roi.height, &stepSize32f);
}