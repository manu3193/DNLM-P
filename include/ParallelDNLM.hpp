
#ifndef PARALLELDNLM_HPP_
#define PARALLELDNLM_HPP_

#include <string>
#include "DNLMFilter.hpp"
using namespace std;

class ParallelDNLM{
	
public:
	Mat processImage(const Mat& U);

private:
	DNLMFilter nlmfd;

	Mat filterDNLM(const Mat& U, int wSize, int wSize_n, double sigma_s, int sigma_r, int lambda);
};
#endif /* PARALLELDNLM_HPP_ */
