#ifndef PARALLELDNLM_HPP_
#define PARALLELDNLM_HPP_

//#include <string>
#include <iostream>
#include <cuda.h>
#include <string>
#include <npp.h>
#include <omp.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;



/**
 * @brief      Class for parallel DNLM filter.
 */
class ParallelDNLM{
	
public:
	Mat processImage(const Mat& U, int wSize, int nSize, float sigma);

private:
	Mat filterDNLM(const Mat& U, int wSize, int wSize_n, float sigma_r);
};
#endif /* PARALLELDNLM_HPP_ */
