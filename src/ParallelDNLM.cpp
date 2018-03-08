
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
    double sigma_s = wRSize/1.5;
    int sigma_r = 12; //13
    int lambda = 5;
    
    Mat fDeceivedNLM = filterDNLM(inputImage, wRSize, wSize_n, sigma_s, sigma_r, lambda);

    return fDeceivedNLM;
}

//Input image must be from 0 to 255
Mat ParallelDNLM::filterDNLM(const Mat& srcImage, int wSize, int wSize_n, double sigma_s, int sigma_r, int lambda){
    
    //Variables used for image format conversion
    IppiSize roi;
    roi.width = srcImage.size().width;
    roi.height = srcImage.size().height;

    //Compute the step in bytes of the 8u and 32f image
    int 8uStep = roi.width * sizeof(Ipp8u); 
    int 32fStep = roi.width * sizeof(Ipp32f);

    //Allocate memory
    Ipp8u *pSrcImage = (Ipp8u*)&srcImage.data[0];                               //Get pointer to src image data
    Ipp32f *pIpp32fImage = ippiMalloc_32f_C1(roi.width, roi.height, &32fStep);  //Allocate buffer for converted image  
    Ipp8u *pDstImage = (Ipp8u*)&outputImage.data[0];                            //Get buffer for output image 

    //Create output image
    Mat outputImage = srcImage.clone();
    
    //Scale factor to normalize 32f image
    Ipp32f normFactor[3] = {1.0/255.0, 1.0/255.0, 1.0/255.0}; 
    Ipp32f scaleFactor[3] = {255.0, 255.0, 255.0}; 

    
    //The input image has to be normalized and single precission float type
    ippiConvert_8u32f_C3R(pSrcImage, srcImage.step, pIpp32fImage, 32fStep, roi);
    ippiMulC_32f_C3IR(normFactor, pIpp32fImage, 32fStep, roi);

    //Mat L = this->nal.noAdaptiveLaplacian(Unorm, lambda);
    //Mat F = this->nlmfd.DNLMFilter(Unorm, L, wSize, wSize_n, sigma_s, sigma_r);

    //putting back everything
    ippiMulC_32f_C3IR(scaleFactor, pIpp32fImage, 32fStep, roi);
    ippiConvert_32f8u_C1R(pIpp32fImage, 32fStep, pDstImage , outputImage.step, roi);
    
    ippiFree(pIpp32fImage);

    return F;
}
