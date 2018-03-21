
#ifndef DNLMFILTER_HPP_
#define DNLMFILTER_HPP_

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class DNLMFilter{

public:
	Mat dnlmFilter(const Mat& A, const Mat& L, int w, int w_n, double sigma_s, int sigma_r);

private:
	Mat nlmfilBW_deceived(const Mat& A, const Mat& L, int w, int w_n, double sigma_d, int sigma_r);
	Mat nlmfltBWDeceived(const Mat& A, const Mat& L, int w, int w_n, double sigma_d, int sigma_r);
    Mat CalcEuclideanDistMat(const Mat& I, int w_n, int i, int j, int iMin, int jMin);
};
#endif /* DNLMFILTER_HPP_ */
