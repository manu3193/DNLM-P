#ifndef DNLMFILTER_HPP_
#define DNLMFILTER_HPP_

#include <cstdio>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <npp.h>
#include <nppi.h>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;

class DNLMFilter{

public:
	int dnlmFilter(const Npp32f* pSrcBorder, int stepBytesSrcBorder, int srcType, const Npp32f* pUSMImage, int stepBytesUSM, Npp32f* pDst, int stepBytesDst, NppiSize imageSize, int w, int w_n, float sigma_r);
private:
	int dnlmFilterBW(const Npp32f* pSrcBorder, int stepBytesSrcBorder, const Npp32f* pUSMImage, int stepByteUSM, Npp32f* pDst, int stepBytesDst, NppiSize imageSize, int w, int w_n, float sigma_r);
	
};
#endif /* DNLMFILTER_HPP_ */
