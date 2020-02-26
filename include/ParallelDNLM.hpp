
#ifndef PARALLELDNLM_HPP_
#define PARALLELDNLM_HPP_

//#include <string>
#include <iostream>
#include <DNLMFilter.hpp>
#include <omp.h>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;

/**
 * @brief      Class for parallel DNLM filter.
 */
class ParallelDNLM{
	
public:
	Mat processImage(const Mat& U, int wSize, int nSize, float sigma);


private:

	DNLMFilter dnlmFilter;

	Mat filterDNLM(const Mat& U, int wSize, int wSize_n, float sigma_r);
};
#endif /* PARALLELDNLM_HPP_ */
