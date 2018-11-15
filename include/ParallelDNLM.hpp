
#ifndef PARALLELDNLM_HPP_
#define PARALLELDNLM_HPP_

#include <string>
#include <iostream>
#include <regex>
#include <ippcv.h>
#include <DNLMFilter.hpp>
#include <NoAdaptiveUSM.hpp>
#include <ippcore.h>
#include <omp.h>

using namespace cv;
using namespace std;

/**
 * @brief      Class for parallel DNLM filter.
 */
class ParallelDNLM{
	
public:
	
	/**
	 * @brief      Constructs the object with default parameters
	 */
	ParallelDNLM();

	/**
	 * @brief      Constructs the object with the given parameters
	 *
	 * @param[in]  wSize      The window size S = wSize x wSize
	 * @param[in]  wSize_n    The neighborhood size W = wSize_n x wSize_n
	 * @param[in]  sigma_r    The filter bandwidth h = 2 * sigma_r
	 * @param[in]  lambda     The lambda gain of USM filter
	 * @param[in]  kernelLen  The kernel length of USM filter USM_kernel_len = kernelLen * kernelLen
	 * @param[in]  kernelStd  The kernel std of USM filter. 
	 * 
	 * For recomended parameters use 6 * kernelStd = kernelLen + 2
	 * 
	 */
	ParallelDNLM(int wSize, int wSize_n, float sigma_r, float lambda, int kernelLen, float kernelStd);
	
	/**
	 * @brief      Process the image
	 *
	 * @param[in]  U     Input image
	 *
	 * @return     Processed image as opencv Mat object
	 */
	Mat processImage(const Mat& U);
private:

	DNLMFilter dnlmFilter;
	NoAdaptiveUSM noAdaptiveUSM;

	int wSize;
    int wSize_n;
    float kernelStd;
    int kernelLen;
    float sigma_r; 
    float lambda;

	/**
	 * @brief      Filter the image with the given parameters
	 *
	 * @param[in]  U          Input image
	 * @param[in]  wSize      The window size S = wSize x wSize
	 * @param[in]  wSize_n    The neighborhood size W = wSize_n x wSize_n
	 * @param[in]  sigma_r    The filter bandwidth h = 2 * sigma_r
	 * @param[in]  lambda     The lambda gain of USM filter
	 * @param[in]  kernelLen  The kernel length of USM filter USM_kernel_len = kernelLen * kernelLen
	 * @param[in]  kernelStd  The kernel std of USM filter. 
	 *
	 * @return     Filtered image as OpenCV Mat object
	 */
	Mat filterDNLM(const Mat& U, int wSize, int wSize_n, float sigma_r, float lambda, int kernelLen, float kernelStd);
};
#endif /* PARALLELDNLM_HPP_ */
