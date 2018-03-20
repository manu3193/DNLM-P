#ifndef NOADAPTIVEUSM_HPP_
#define NOADAPTIVEUSM_HPP_

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ipp.h>

using namespace cv;
using namespace std;

class NoAdaptiveUSM{

public:
	NoAdaptiveUSM();
	int generateLoGKernel(int size, float sigma, Ipp32f* pKernel );
	int noAdaptiveUSM(const Ipp32f* pSrc, Ipp32f* pDst, float lambda, IppiMaskSize mask);
private:
	
};
#endif /* NOADAPTIVEUSM_HPP_ */
