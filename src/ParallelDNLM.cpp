
#include <ParallelDNLM.hpp>

using namespace std;
  


int main(int argc, char* argv[]){
    ParallelDNLM parallelDNLM;

    //Check input arguments
    if (argc != 2){
        cout << "ERROR: You must provide a valid image filename" << endl;
        return -1;
    }else{

    //Open input image
    const string inputFile = argv[1];
    // Find extension point
    string::size_type pAt = inputFile.find_last_of('.');

    // Form the new name with container
    const string outputFile = inputFile.substr(0, pAt) + "_DeNLM.png";

    Mat inputImage, outputImage;

    //This version only works with grayscale images
    inputImage = imread(inputFile, IMREAD_GRAYSCALE);
    //Check for errors when loading image
        if(!inputImage.data){
            cout << "Could not read image from file." << endl;
            return -1;
        }
    
    outputImage = parallelDNLM.processImage(inputImage);

    //Write image to output file.
    imwrite(outputFile, outputImage);
    
    return 0;
    }
}



Mat ParallelDNLM::processImage(const Mat& inputImage){
    //Set parameters for processing
    int wRSize = 7;
    int wSize_n=3;
    float kernelStd = 0.001f;
    int kernelLen = 19;
    float sigma_r = 12.0f; //13
    float lambda = 1.5f;
    
    Mat fDeceivedNLM = filterDNLM(inputImage, wRSize, wSize_n, sigma_r, lambda, kernelLen, kernelStd);

    return fDeceivedNLM;
}

//Input image must be from 0 to 255
Mat ParallelDNLM::filterDNLM(const Mat& srcImage, int wSize, int wSize_n, float sigma_r, float lambda, int kernelLen, double kernelStd){
    
    //Status variable helps to check for errors
    IppStatus status = ippStsNoErr;

    //Pointers to IPP type images 
    Ipp32f *pSrc32fImage = NULL, *pUSMImage = NULL, *pFilteredImage= NULL;

    //Variable to store image step size in bytes 
    int stepBytesSrc = 0;
    int stepBytesUSM = 0;
    int stepBytesFiltered = 0;

    //Scale factors to normalize and denormalize 32f image
    Ipp32f normFactor = 1.0/255.0; 
    Ipp32f scaleFactor = 255.0; 

    //cv::Mat output filtered image
    Mat outputImage = Mat(srcImage.size(), srcImage.type());

    //Variables used for image format conversion
    IppiSize roi;
    roi.width = srcImage.size().width;
    roi.height = srcImage.size().height;

    //Get pointer to src and dst data
    Ipp8u *pSrcImage = (Ipp8u*)&srcImage.data[0];                               
    Ipp8u *pDstImage = (Ipp8u*)&outputImage.data[0];                            

    //Allocate memory for images
    pSrc32fImage = ippiMalloc_32f_C1(roi.width, roi.height, &stepBytesSrc); 
    pUSMImage = ippiMalloc_32f_C1(roi.width, roi.height, &stepBytesUSM);
    pFilteredImage = ippiMalloc_32f_C1(roi.width, roi.height, &stepBytesFiltered);   
    
    //Convert input image to 32f format
    ippiConvert_8u32f_C1R(pSrcImage, srcImage.step[0], pSrc32fImage, stepBytesSrc, roi);
    //Normalize converted image
    ippiMulC_32f_C1IR(normFactor, pSrc32fImage, stepBytesSrc, roi);

    this->noAdaptiveUSM.noAdaptiveUSM(pSrc32fImage, stepBytesSrc, pUSMImage, stepBytesUSM, roi, kernelStd, lambda, kernelLen);
    //this->nlmfd.DNLMFilter(pSrc32fImage, pUSMImage, pFilteredImage, wSize, wSize_n, sigma_s, sigma_r);

    //putting back everything
    //ippiMulC_32f_C1IR(scaleFactor, pFilteredImage, stepBytesFiltered, roi);
    //ippiConvert_32f8u_C1R(pFilteredImage, stepBytesFiltered, pDstImage , outputImage.step[0], roi, ippRndFinancial);
    ippiMulC_32f_C1IR(scaleFactor, pUSMImage, stepBytesUSM, roi);
    ippiConvert_32f8u_C1R(pUSMImage, stepBytesUSM, pDstImage , outputImage.step[0], roi, ippRndFinancial);
    
    //Freeing memory
    ippiFree(pSrc32fImage);
    ippiFree(pUSMImage);
    ippiFree(pFilteredImage);

    return outputImage;
}
