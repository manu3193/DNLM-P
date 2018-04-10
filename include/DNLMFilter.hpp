
#ifndef DNLMFILTER_HPP_
#define DNLMFILTER_HPP_

#include <cstdio>
#include <opencv2/opencv.hpp>
#include <ipp.h>

using namespace cv;
using namespace std;

class DNLMFilter{

public:
	int dnlmFilter(const Ipp32f* pSrc, int stepBytesSrc, int srcType, const Ipp32f* pUSMImage, int stepBytesUSM, Ipp32f* pDst, int stepBytesDst, IppiSize imageSize, int w, int w_n, float sigma_r);

private:
	int dnlmFilterBW(const Ipp32f* pSrc, int stepBytesSrc, const Ipp32f* pUSMImage, int stepByteUSM, Ipp32f* pDst, int stepBytesDst, IppiSize imageSize, int w, int w_n, float sigma_r);
};
#endif /* DNLMFILTER_HPP_ */
