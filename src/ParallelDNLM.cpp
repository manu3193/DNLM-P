
#include <ParallelDNLM.hpp>

using namespace std;
  

/**
 * @brief      Constructs the object with default parameters
 */
ParallelDNLM::ParallelDNLM(){
    this->wSize = 21;
    this->wSize_n = 7;
    this->kernelStd = 3;
    this->kernelLen = 16;
    this->sigma_r = 10; 
    this->lambda = 1;
}


/**
 * @brief      Constructs the object with the given parameters
 *
 * @param[in]  wSize      The window size S = wSize x wSize
 * @param[in]  wSize_n    The neighborhood size W = wSize_n x wSize_n
 * @param[in]  sigma_r    The filter bandwidth h = 2 * sigma_r
 * @param[in]  lambda     The lambda gain of USM filter
 * @param[in]  kernelLen  The kernel length of USM filter USM_kernel_len = kernelLen * kernelLen
 * @param[in]  kernelStd  The kernel std of USM filter. 
 * 
 * For recomended parameters use 6 * kernelStd = kernelLen + 2
 * 
 */
ParallelDNLM::ParallelDNLM(int wSize, int wSize_n, float sigma_r, float lambda, int kernelLen, float kernelStd){
    this->wSize = wSize;
    this->wSize_n = wSize_n;
    this->kernelStd = kernelStd;
    this->kernelLen = kernelLen;
    this->sigma_r = sigma_r; 
    this->lambda = lambda;
}



int main(int argc, char* argv[]){
    ParallelDNLM *parallelDNLM;
    regex intRegex = regex("[+]?[0-9]+");
    regex floatRegex = regex("[+]?([0-9]*[.])?[0-9]+");

    if (argc == 2){
        //cout << "Using default parameters W=21x21, W_n=7x7, sigma_r=10, lambda=1, USM_len=16,USM_std=3" << endl;
        parallelDNLM = new ParallelDNLM();
    } 
    //Check input arguments
    else if (argc !=8){
        cerr << "Error parsing parameters, please look at the documentation for correct use" <<endl;
        return -1;
    }
    else if (!regex_match(string(argv[2]), intRegex) & !regex_match(string(argv[3]), intRegex) & !regex_match(string(argv[6]), intRegex)){
        return -1;
    }
    else if (!regex_match(string(argv[4]), floatRegex) & !regex_match(string(argv[5]), floatRegex)  & !regex_match(string(argv[7]), floatRegex)){
        return -1;
    }
    else{
        parallelDNLM = new ParallelDNLM(stoi(string(argv[2])), stoi(string(argv[3])), stof(string(argv[4])), stof(string(argv[5])), stoi(string(argv[6])), stof(string(argv[7])));
    }

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

    //Process image   
    outputImage = parallelDNLM->processImage(inputImage);

    //Write image to output file.
    imwrite(outputFile, outputImage);

    //Release object memory
    inputImage.release();
    outputImage.release();
    
    return 0;
}




Mat ParallelDNLM::processImage(const Mat& inputImage){
    //Set parameters for processing
    Mat fDeceivedNLM = filterDNLM(inputImage, wSize, wSize_n, sigma_r, lambda, kernelLen, kernelStd);

    return fDeceivedNLM;
}

//Input image must be from 0 to 255
Mat ParallelDNLM::filterDNLM(const Mat& srcImage, int wSize, int wSize_n, float sigma_r, float lambda, int kernelLen, float kernelStd){
    //Create CUDA events to meausre execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //Status variable helps to check for errors
    NppStatus status =  NPP_NO_ERROR;
    //Pointers to NPP type images 
    Npp32f *pSrcImage32f = NULL, *pSrcwBorderImage = NULL, *pUSMImage = NULL, *pFilteredImage32f = NULL;
    Npp8u *pUSMBuffer = NULL, *pSrcImage8u = NULL, *pFilteredImage8u = NULL;
    //Variable to store image step size in bytes 
    int stepBytesSrc32f = 0;
    int stepBytesSrc8u = 0;
    int stepBytesSrcwBorder = 0;
    int stepBytesUSM = 0;
    int stepBytesFiltered32f = 0;
    int stepBytesFiltered8u = 0;
    //Scale factors to normalize and denormalize 32f image
    Npp32f normFactor = 1.0/255.0; 
    Npp32f scaleFactor = 255.0; 

    //cv::Mat output filtered image
    Mat outputImage = Mat(srcImage.size(), srcImage.type());

    //Variables used for image format conversion
    NppiSize imageROISize, imageROIwBorderSize;
    imageROISize.width = srcImage.size().width;
    imageROISize.height = srcImage.size().height;

    //Compute border offset for border replicated image
    const int imageTopLeftOffset = floor(wSize_n/2);

    imageROIwBorderSize = {imageROISize.width + 2*imageTopLeftOffset, imageROISize.height + 2*imageTopLeftOffset};     
    //Variable to store USM buffer size
    int usmBufferSize = 0;
    //Compute USM radius
    const Npp32f usmRadius = (kernelLen-1)/2;  
    //Compute USM buffer size
    status = nppiFilterUnsharpGetBufferSize_32f_C1R(usmRadius, kernelStd, &usmBufferSize);
    //Allocate memory for gpu images
    pSrcImage8u = nppiMalloc_8u_C1(imageROISize.width, imageROISize.height, &stepBytesSrc8u);
    pSrcImage32f = nppiMalloc_32f_C1(imageROISize.width, imageROISize.height, &stepBytesSrc32f); 
    pSrcwBorderImage = nppiMalloc_32f_C1(imageROIwBorderSize.width, imageROIwBorderSize.height, &stepBytesSrcwBorder);
    pUSMImage =  nppiMalloc_32f_C1(imageROIwBorderSize.width, imageROIwBorderSize.height, &stepBytesUSM);
    pFilteredImage32f = nppiMalloc_32f_C1(imageROIwBorderSize.width, imageROIwBorderSize.height, &stepBytesFiltered32f);   
    pFilteredImage8u = nppiMalloc_8u_C1(imageROIwBorderSize.width, imageROIwBorderSize.height, &stepBytesFiltered8u);
    //Allocate memory for usm bufer
    pUSMBuffer = nppsMalloc_8u(usmBufferSize);
    //Set result gpu image to 0
    nppiSet_32f_C1R((Npp32f) 0.0f, pFilteredImage32f, stepBytesFiltered32f, {imageROISize.width, imageROISize.height});
    //Copy images to gpu
    cudaMemcpy2D(pSrcImage8u, stepBytesSrc8u, &srcImage.data[0], srcImage.step[0], imageROISize.width, imageROISize.height, cudaMemcpyHostToDevice);    

    //Convert input image to 32f format
    nppiConvert_8u32f_C1R(pSrcImage8u, stepBytesSrc8u, pSrcImage32f, stepBytesSrc32f, imageROISize);
    //Normalize converted image
    //ippiMulC_32f_C1IR(normFactor, pSrc32fImage, stepBytesSrc, imageROISize);

    // Mirror border for full image filtering
    status = nppiCopyWrapBorder_32f_C1R(pSrcImage32f, stepBytesSrc32f, imageROISize, pSrcwBorderImage, stepBytesSrcwBorder, imageROIwBorderSize, imageTopLeftOffset, imageTopLeftOffset);
    //timer start
    cudaEventRecord(start); 
    //Applying USM Filter
    status =nppiFilterUnsharpBorder_32f_C1R(pSrcwBorderImage, stepBytesSrcwBorder, {imageTopLeftOffset,imageTopLeftOffset}, pUSMImage, stepBytesUSM, {imageROIwBorderSize.width, imageROIwBorderSize.height}, usmRadius, kernelStd, lambda, 2, NPP_BORDER_REPLICATE, pUSMBuffer);
    //Gossens version doesnt works with normalized images
    //ippiMulC_32f_C1IR(scaleFactor, pSrcwBorderImage, stepBytesSrcwBorder, imageROIwBorderSize);
    //ippiMulC_32f_C1IR(scaleFactor, pUSMImage, stepBytesUSM, imageROIwBorderSize);
    //Aplying DNLM filter
    this->dnlmFilter.dnlmFilter(pSrcwBorderImage, stepBytesSrcwBorder, CV_32FC1, pUSMImage, stepBytesUSM, pFilteredImage32f, stepBytesFiltered32f,  {imageROIwBorderSize.width, imageROIwBorderSize.height}, wSize, wSize_n, sigma_r);
    //Measure slapsed time
    cudaEventRecord(stop);
    
    //Convert back to uchar, add offset to pointer to remove border
    nppiConvert_32f8u_C1R(pFilteredImage32f, stepBytesFiltered32f, pFilteredImage8u, stepBytesFiltered8u, imageROIwBorderSize, NPP_RND_FINANCIAL);

    cudaMemcpy2D((void *) &outputImage.data[0], outputImage.step[0], (void *) (pFilteredImage8u + imageTopLeftOffset*stepBytesFiltered8u/sizeof(Npp8u)+imageTopLeftOffset), 
                 stepBytesFiltered8u, imageROISize.width, imageROISize.height, cudaMemcpyDeviceToHost);
    
    cudaEventSynchronize(stop);
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, stop);
    //Freeing memory
    nppiFree(pSrcImage32f); 
    nppiFree(pSrcImage8u);
    nppiFree(pSrcwBorderImage);
    nppiFree(pFilteredImage32f);
    nppiFree(pUSMImage);
    nppiFree(pFilteredImage8u);    
    nppsFree(pUSMBuffer);
    cout << elapsed/1000 <<endl;

    return outputImage;
}
