#ifndef NOADAPTIVEUSM_HPP_
#define NOADAPTIVEUSM_HPP_

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ipp.h>

using namespace cv;
using namespace std;

class NoAdaptiveUSM{

public:
	Ipp32f* noAdaptiveUSM(const Ipp32f* pSrc, float lambda, IppiMaskSize mask);

private:
	
	
};
#endif /* NOADAPTIVEUSM_HPP_ */
