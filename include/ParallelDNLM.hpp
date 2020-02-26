
#ifndef PARALLELDNLM_HPP_
#define PARALLELDNLM_HPP_

#include <iostream>
#include <string>
#include <ippcv.h>
#include <DNLMFilter.hpp>
#include <omp.h>

using namespace cv;
using namespace std;

class ParallelDNLM{
	
public:
	Mat processImage(const Mat& U, int wSize, int nSize, float sigma);

private:
	DNLMFilter dnlmFilter;
	Mat filterDNLM(const Mat& U, int wSize, int wSize_n, float sigma_r);
};
#endif /* PARALLELDNLM_HPP_ */
