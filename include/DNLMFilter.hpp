
#ifndef DNLMFILTER_HPP_
#define DNLMFILTER_HPP_

#include <cstdio>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <ipp.h>
#include <ippcv.h>

using namespace cv;
using namespace std;

class DNLMFilter{

public:
	int dnlmFilter(const Ipp32f* pSrcBorder, int stepBytesSrcBorder, int srcType, const Ipp32f* pUSMImage, int stepBytesUSM, Ipp32f* pDst, int stepBytesDst, IppiSize imageSize, int w, int w_n, float sigma_r);

private:
	int dnlmFilterBW(const Ipp32f* pSrcBorder, int stepBytesSrcBorder, const Ipp32f* pUSMImage, int stepByteUSM, Ipp32f* pDst, int stepBytesDst, IppiSize imageSize, int w, int w_n, float sigma_r);
};
#endif /* DNLMFILTER_HPP_ */
