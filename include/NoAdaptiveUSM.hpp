#ifndef NOADAPTIVEUSM_HPP_
#define NOADAPTIVEUSM_HPP_

#include <cstdio>
#include <opencv2/opencv.hpp>
#include <ipp.h>

using namespace cv;
using namespace std;

class NoAdaptiveUSM{

public:
	NoAdaptiveUSM();
	int generateLoGKernel(int size, double sigma, Ipp32f* pKernel );
	int noAdaptiveUSM(const Ipp32f* pSrc, Ipp32f* pDst, IppiSize roiSize, float lambda, int kernelLen);
private:
	const Ipp64f ipp_eps52 = 2.2204460492503131e-016; 
};
#endif /* NOADAPTIVEUSM_HPP_ */
