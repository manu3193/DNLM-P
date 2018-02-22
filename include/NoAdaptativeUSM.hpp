#ifndef NOADAPTATIVEUSM_HPP_
#define NOADAPTATIVEUSM_HPP_

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ipp.h>

using namespace cv;
using namespace std;

class NoAdaptativeUSM{

public:
	Ipp32f* noAdaptativeUSM(const Ipp32f* pSrc, float lambda, IppiMaskSize mask);

private:
	
	
};
#endif /* NOADAPTATIVEUSM_HPP_ */
