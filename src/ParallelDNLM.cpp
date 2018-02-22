
#include <ParallelDeNLM.hpp>
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

    inputImage = imread(inputFile, CV_LOAD_IMAGE_COLOR);
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
    double sigma_s = wRSize/1.5;
    int sigma_r = 12; //13
    int lambda = 5;
    
    Mat fDeceivedNLM = filterDNLM(inputImage, wRSize, wSize_n, sigma_s, sigma_r, lambda);

    return fDeceivedNLM;
}

//Input image must be from 0 to 255
Mat ParallelDNLM::filterDNLM(const Mat& srcImage, int wSize, int wSize_n, double sigma_s, int sigma_r, int lambda){
    //Create output image
    Mat outputImage = srcImage.clone();
    //Variables used for image format conversion
    IppiSize roi;
    roi.width = srcImage.size().width;
    roi.height = srcImage.size().height;
    //Scale factor to normalize 32f image
    Ipp32f scaleFactor[3] = {1.0/255.0, 1.0/255.0, 1.0/255.0} 

    //Compute the step in bytes of the dst norm and dest data
    int dstStep = srcImage.size().width * srcImage.channels() * sizeof(Ipp32f); 
    int dstNormStep = srcImage.size().width * srcImage.channels() * sizeof(Ipp8u);

    Ipp8u *pIppInputImage = (Ipp8u*)&srcImage.data[0];  //Get pointer to image data
    Ipp32f *pIpp32fImage = NULL;                        //Pointer to 32f converted image  
    Ipp8u *pIpp8uNormImage = NULL;                  //Pointer to 8u normalized image 

    //The input image has to be normalized and single precission float type
    ippiConvert_8u32f_C3R(pIppInputImage, srcImage.step, pIpp32fImage, dstStep, roi);
    ippiMulC_32f_C3IR(scaleFactor, pIpp32fImage, dstStep, roi);

    //Mat L = this->nal.noAdaptiveLaplacian(Unorm, lambda);
    //Mat F = this->nlmfd.DNLMFilter(Unorm, L, wSize, wSize_n, sigma_s, sigma_r);

    //putting back everything
    ippiConvert_32f8u_C3R(pIpp32fImage, dstStep, pIpp8uNormImage, outputImage.step, roi, ippRndNear);
    ippiMulC_8u_C3RSfs(const Ipp<datatype>* pSrc, int srcStep, const Ipp<datatype> value[3], Ipp<datatype>* pDst, int dstStep, IppiSize roiSize, int scaleFactor);
    (Ipp8u*)&outputImage.data[0],
    return F;
}
