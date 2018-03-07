#include <stdio.h>
#include <opencv2/opencv.hpp>

#define WIDTH  1920  /* image width */
#define HEIGHT  1080  /* image height */
const float kernel[3*3] = {-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0}; // Define high pass filter


using namespace cv; 

int main(int, char**)
{
    Mat inputImage, outputImage;
    Mat<float> floatMat;
    //Loading image
    inputImage = imread("lena.bmp", IMREAD_GRAYSCALE);
    //The output image has the same shape of the input one
    outputImage = inputImage.clone();

    printf("Converting image to 32f\n"); 
    //The input image has to be normalized and single precission float type
    inputImage.convertTo(floatMat, CV_32F, 1.0/255.0);

    //aplying high pass filter
    
    Point anchor( -1 , -1 );
    double delta = 0;
    int ddepth = -1;

    Mat filterResMat = Mat(floatMat.size(), floatMat.type());
    filter2D(floatMat, filterResMat, ddepth , kernel, anchor, delta, BORDER_REPLICATE );
    filterResMat.convertTo(outputImage, CV_8U, 255);

    imwrite("lena_sharp.bmp", outputImage);
}
