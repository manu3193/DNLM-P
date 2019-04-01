
#include <ParallelDNLM.hpp>

using namespace std;
  

/**
 *  * @brief      Constructs the object with default parameters
 *   */
ParallelDNLM::ParallelDNLM(){
    this->wSize = 21;
    this->wSize_n = 7;
    this->kernelStd = 3;
    this->kernelLen = 16;
    this->sigma_r = 10; 
    this->lambda = 1;
}

/**
 *  * @brief      Constructs the object with the given parameters
 *   *
 *    * @param[in]  wSize      The window size S = wSize x wSize
 *     * @param[in]  wSize_n    The neighborhood size W = wSize_n x wSize_n
 *      * @param[in]  sigma_r    The filter bandwidth h = 2 * sigma_r
 *       * @param[in]  lambda     The lambda gain of USM filter
 *        * @param[in]  kernelLen  The kernel length of USM filter USM_kernel_len = kernelLen * kernelLen
 *         * @param[in]  kernelStd  The kernel std of USM filter. 
 *          * 
 *           * For recomended parameters use 6 * kernelStd = kernelLen + 2
 *            * 
 *             */
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
    else { 
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
                                                                                                                                                                                             return 0;
                                                                                                                                                                                          }


Mat ParallelDNLM::processImage(const Mat& inputImage){
   //Set Parameters for processing 
    Mat fDeceivedNLM = filterDNLM(inputImage, wSize, wSize_n, sigma_r, lambda, kernelLen, kernelStd);

    return fDeceivedNLM;
}

//Input image must be from 0 to 255
Mat ParallelDNLM::filterDNLM(const Mat& srcImage, int wSize, int wSize_n, float sigma_r, float lambda, int kernelLen, float kernelStd){
    
    //Get cuda device properties
    int device;
    cudaError_t cudaStatus = cudaSuccess;
    //Create CUDA events to measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //Status variable helps to check for errors
    NppStatus status =  NPP_NO_ERROR;
    
    cudaStatus = cudaGetDevice(&device);
    if (cudaStatus == cudaSuccess)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        cout <<"Device Number: " <<device <<endl;
        cout <<"  Device name: " <<prop.name <<endl;
        cout <<"  Memory Clock Rate (KHz): "<<prop.memoryClockRate<<endl;
        cout <<"  Memory Bus Width (bits): "<<prop.memoryBusWidth<<endl;
        cout <<"  Peak Memory Bandwidth (GB/s): "<<2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 <<endl;
        cout <<"  Compute Capability: "<<prop.major<<"."<<prop.minor<<endl;

    }    
    //Pointers to NPP type images 
    //OLD Npp32f *pSrcImage32f = NULL, *pSrcwBorderImage = NULL, *pUSMImage = NULL, *pFilteredImage32f = NULL;
    Npp32f *pSrcImage32f = NULL, *pSrcwBorderImage = NULL, *pFilteredImage32f = NULL;
    //OLD Npp8u *pUSMBuffer = NULL, *pSrcImage8u = NULL, *pFilteredImage8u = NULL;
    Npp8u *pSrcImage8u = NULL, *pFilteredImage8u = NULL;
    //Variable to store image step size in bytes 
    int stepBytesSrc32f = 0;
    int stepBytesSrc8u = 0;
    int stepBytesSrcwBorder = 0;
    //OLD int stepBytesUSM = 0;
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
    //OLD Variable to store USM buffer size
    //OLD int usmBufferSize = 0;
    //OLD Compute USM radius
    //OLD const Npp32f usmRadius = (kernelLen-1)/2;  
    //OLD Compute USM buffer size
    //OLD status = nppiFilterUnsharpGetBufferSize_32f_C1R(usmRadius, kernelStd, &usmBufferSize);
    //Allocate memory for gpu images
    pSrcImage8u = nppiMalloc_8u_C1(imageROISize.width, imageROISize.height, &stepBytesSrc8u);
    pSrcImage32f = nppiMalloc_32f_C1(imageROISize.width, imageROISize.height, &stepBytesSrc32f); 
    pSrcwBorderImage = nppiMalloc_32f_C1(imageROIwBorderSize.width, imageROIwBorderSize.height, &stepBytesSrcwBorder);
    //OLD pUSMImage =  nppiMalloc_32f_C1(imageROIwBorderSize.width, imageROIwBorderSize.height, &stepBytesUSM);
    pFilteredImage32f = nppiMalloc_32f_C1(imageROISize.width, imageROISize.height, &stepBytesFiltered32f);   
    pFilteredImage8u = nppiMalloc_8u_C1(imageROISize.width, imageROISize.height, &stepBytesFiltered8u);
    
    if (pSrcImage8u ==NULL) cout << "error alocating pSrcImage8u" << endl;
    if (pSrcImage32f ==NULL) cout << "error alocating pSrcImage32f" << endl;
    if (pSrcwBorderImage ==NULL) cout << "error alocating pSrcwBorderImage" << endl;
    //OLD if (pUSMImage ==NULL) cout << "error alocating pUSMImage" << endl;
    if (pFilteredImage32f ==NULL) cout << "error alocating pFilteredImage32f" << endl;
    if (pFilteredImage8u ==NULL) cout << "error alocating pFilteredImage8u" << endl;
    
    //Allocate memory for usm bufer
    //OLD pUSMBuffer = nppsMalloc_8u(usmBufferSize);
    //Copy images to gpu
    cudaStatus = cudaMemcpy2D(pSrcImage8u, stepBytesSrc8u, srcImage.data, srcImage.step[0], imageROISize.width, imageROISize.height, cudaMemcpyHostToDevice);
    // OLD cudaStatus = cudaMemcpy2D((void*) pSrcImage8u, stepBytesSrc8u, (void*) srcImage.data, srcImage.step[0], imageROISize.width, imageROISize.height, cudaMemcpyHostToDevice);
    if (cudaStatus !=cudaSuccess) cout << " error 2 " << cudaStatus  << " " << stepBytesSrc8u << " " << srcImage.step[0] << endl;
    //Convert input image to 32f format
    status = nppiConvert_8u32f_C1R(pSrcImage8u, stepBytesSrc8u, pSrcImage32f, stepBytesSrc32f, imageROISize);
    if (status !=0) cout << " error 3 " << status << endl;
    //Normalize converted image
    //ippiMulC_32f_C1IR(normFactor, pSrc32fImage, stepBytesSrc, imageROISize);

    // Mirror border for full image filtering
    status = nppiCopyWrapBorder_32f_C1R(pSrcImage32f, stepBytesSrc32f, imageROISize, pSrcwBorderImage, stepBytesSrcwBorder, imageROIwBorderSize, imageTopLeftOffset, imageTopLeftOffset);
    if (status !=0) cout << " error 4 " << status << endl;
    //timer start
    cudaEventRecord(start); 
    //OLD Applying USM Filter
    //OLD status =nppiFilterUnsharpBorder_32f_C1R(pSrcwBorderImage, stepBytesSrcwBorder, {mageTopLeftOffset, imageTopLeftOffset}, pUSMImage, stepBytesUSM, imageROIwBorderSize, usmRadius, kernelStd, lambda, 2, NPP_BORDER_REPLICATE, pUSMBuffer);
    //OLD if (status !=0) cout << " error 5 " << status << endl;
    //Unormalize
    //Here
    //Aplying DNLM filter
    //OLD this->dnlmFilter.dnlmFilter(pSrcwBorderImage, stepBytesSrcwBorder, CV_32FC1, pUSMImage, stepBytesUSM, pFilteredImage32f, stepBytesFiltered32f,  imageROISize, wSize, wSize_n, sigma_r);
    this->dnlmFilter.dnlmFilter(pSrcwBorderImage, stepBytesSrcwBorder, CV_32FC1, pFilteredImage32f, stepBytesFiltered32f,  imageROISize, wSize, wSize_n, sigma_r);
    //Measure slapsed time
    cudaEventRecord(stop);
    
    //Convert back to uchar, add offset to pointer to remove border
    status = nppiConvert_32f8u_C1R(pFilteredImage32f, stepBytesFiltered32f, pFilteredImage8u, stepBytesFiltered8u, imageROISize, NPP_RND_FINANCIAL);
    if (status !=0) cout << " error converting" << status << endl;

    cudaStatus = cudaMemcpy2D(outputImage.data, outputImage.step[0], pFilteredImage8u,
                 stepBytesFiltered8u, imageROISize.width, imageROISize.height, cudaMemcpyDeviceToHost);
    if (cudaStatus !=cudaSuccess) cout << " error copying back to host" << cudaStatus << endl;
    cudaEventSynchronize(stop);
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, stop);
    //Freeing memory
    nppiFree(pSrcImage32f); 
    nppiFree(pSrcImage8u);
    nppiFree(pSrcwBorderImage);
    nppiFree(pFilteredImage32f);
    //OLD nppiFree(pUSMImage);
    nppiFree(pFilteredImage8u);    
    //OLD nppsFree(pUSMBuffer);
    cout << elapsed/1000 <<endl;

    return outputImage;
}
