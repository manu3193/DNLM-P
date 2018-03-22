#ifndef NOADAPTIVEUSM_HPP_
#define NOADAPTIVEUSM_HPP_

#include <cstdio>
#include <opencv2/opencv.hpp>
#include <ipp.h>

using namespace cv;
using namespace std;

class NoAdaptiveUSM{

public:
	int generateLoGKernel(int size, float sigma, Ipp32f* pKernel );
	int noAdaptiveUSM(const Ipp32f* pSrc, Ipp32f* pDst, IppiSize roiSize, float sigma, float lambda, int kernelLen);
private:

};
#endif /* NOADAPTIVEUSM_HPP_ */
