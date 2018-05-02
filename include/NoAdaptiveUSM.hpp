#ifndef NOADAPTIVEUSM_HPP_
#define NOADAPTIVEUSM_HPP_

#include <cstdio>
#include <opencv2/opencv.hpp>
#include <ipp.h>

using namespace cv;
using namespace std;

class NoAdaptiveUSM{

public:
	int generateLoGKernel(const int size, const float sigma, Ipp32f* pKernel );
	int noAdaptiveUSM(const Ipp32f* pSrc, int stepBytesSrc, Ipp32f* pDst, int stepByteDst, const IppiSize roiSize, const float sigma, const float lambda, const int kernelLen);
	void setNumberThreads(int num);
private:
	int threads;

};
#endif /* NOADAPTIVEUSM_HPP_ */
