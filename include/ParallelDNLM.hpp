
#ifndef PARALLELDNLM_HPP_
#define PARALLELDNLM_HPP_

#include <iostream>
#include <string>
#include <ippcv.h>
#include <DNLMFilter.hpp>
#include <NoAdaptiveUSM.hpp>
#include <ippcore.h>
#include <timer.h>
#include <omp.h>

using namespace cv;
using namespace std;

class ParallelDNLM{
	
public:
	Mat processImage(const Mat& U);

private:
	DNLMFilter dnlmFilter;
	NoAdaptiveUSM noAdaptiveUSM;
	int threads;
	Mat filterDNLM(const Mat& U, int wSize, int wSize_n, float sigma_r, float lambda, int kernelLen, double kernelStd);
};
#endif /* PARALLELDNLM_HPP_ */
