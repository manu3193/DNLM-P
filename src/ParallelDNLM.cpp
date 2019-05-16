#include <ParallelDNLM.hpp>

using namespace std;
  
void DNLM_OpenACC(const float*, int, const float*, int, float*, int, int, int, int, int, int, int, int, int, float);

/**
 *  * @brief      Constructs the object with default parameters
 *   */
ParallelDNLM::ParallelDNLM(){
    this->wSize = 21;
    this->wSize_n = 7;
    this->kernelStd = 3;
    this->kernelLen = 16;
    this->sigma_r = 1; 
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

    //regex intRegex = regex("[+]?[0-9]+");
    //regex floatRegex = regex("[+]?([0-9]*[.])?[0-9]+");

    if (argc == 2){
        cout << "Using default parameters W=21x21, W_n=7x7, sigma_r=10, lambda=1, USM_len=16,USM_std=3" << endl;
        parallelDNLM = new ParallelDNLM();
    } 
    //Check input arguments
    else if (argc !=8){
        cerr << "Error parsing parameters, please look at the documentation for correct use" <<endl;
        return -1;
    }
    //else if (!regex_match(string(argv[2]), intRegex) & !regex_match(string(argv[3]), intRegex) & !regex_match(string(argv[6]), intRegex)){
    //    return -1;
    //}
    //else if (!regex_match(string(argv[4]), floatRegex) & !regex_match(string(argv[5]), floatRegex)  & !regex_match(string(argv[7]), floatRegex)){
    //    return -1;
    //}
    else { 
        parallelDNLM = new ParallelDNLM(stoi(string(argv[2])), stoi(string(argv[3])), stof(string(argv[4])), stof(string(argv[5])), stoi(string(argv[6])), stof(string(argv[7])));
    } 

    //Open input image
    const string inputFile = argv[1];
    // Find extension point
    string::size_type pAt = inputFile.find_last_of('.');
    // Form the new name with container
    const string outputFile = inputFile.substr(0, pAt) + "_DeNLM.png";
    //This version only works with grayscale images
    Mat inputImage = imread(inputFile, IMREAD_GRAYSCALE);
    //Check for errors when loading image
    if(!inputImage.data){
        cout << "Could not read image from file." << endl;
        return -1;
    }
    //Process image 
    Mat outputImage  = parallelDNLM->processImage(inputImage);
    //Write image to output file.
    imwrite(outputFile, outputImage);
    //Release object memory
    inputImage.release();
    outputImage.release();

    return 0;
                                                                                                                                                                                              }


Mat ParallelDNLM::processImage(const Mat inputImage){
   //Set Parameters for processing 
    Mat fDeceivedNLM = filterDNLM(inputImage, wSize, wSize_n, sigma_r, lambda, kernelLen, kernelStd);

    return fDeceivedNLM;
}

//Input image must be from 0 to 255
Mat ParallelDNLM::filterDNLM(const Mat inputImage, int wSize, int wSize_n, float sigma_r, float lambda, int kernelLen, float kernelStd){
    
    //Get cuda device properties
    //int device;
    cudaError_t cudaStatus = cudaSuccess;
    //Create CUDA events to measure execution time
    cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    //Status variable helps to check for errors
    NppStatus status =  NPP_NO_ERROR;
    //Pointers to NPP type images 
    Npp32f *pIntegralImage32f = NULL, *pSrcwBorderImage32f = NULL, *pFilteredImage32f = NULL;
    Npp64f *pSqrIntegralImage64f = NULL;
    Npp8u  *pSrcImage8u = NULL, *pSrcwBorderImage8u = NULL, *pFilteredImage8u = NULL;
    //Variable to store image step size in bytes 
    int stepBytesSrcwBorder8u = 0;
    int stepBytesSrcwBorder32f = 0;
    int stepBytesSrc8u = 0;
    int stepBytesSqrIntegral64f = 0;
    int stepBytesFiltered32f = 0;
    int stepBytesFiltered8u = 0;
    int stepBytesIntegral = 0;
    //Scale factors to normalize and denormalize 32f image
    Npp64f normFactor64f = 1.0/255.0;
    Npp32f normFactor32f = 1.0/255.0; 
    Npp32f scaleFactor = 255.0; 

    //cv::Mat output filtered image
    Mat outputImage = Mat(inputImage.size(), inputImage.type());

    //Variables used for image format conversion
    NppiSize imageROISize, imageROIwBorderSize, integralImageROISize;
    imageROISize.width = inputImage.size().width;
    imageROISize.height = inputImage.size().height;

    //Compute border offset for border replicated image
    const int windowRadius = floor(wSize/2);
    const int neighborRadius = floor(wSize_n/2);
    const int imageTopLeftOffset = windowRadius + neighborRadius;

    imageROIwBorderSize = {imageROISize.width + 2*imageTopLeftOffset, imageROISize.height + 2*imageTopLeftOffset};     
    integralImageROISize = {imageROIwBorderSize.width+1, imageROIwBorderSize.height+1};
    //Allocate memory for gpu images
    pSrcImage8u = nppiMalloc_8u_C1(imageROISize.width, imageROISize.height, &stepBytesSrc8u);
    pSrcwBorderImage32f = nppiMalloc_32f_C1(imageROIwBorderSize.width, imageROIwBorderSize.height, &stepBytesSrcwBorder32f); 
    pIntegralImage32f = nppiMalloc_32f_C1(integralImageROISize.width, integralImageROISize.height, &stepBytesIntegral);
    pSrcwBorderImage8u = nppiMalloc_8u_C1(imageROIwBorderSize.width, imageROIwBorderSize.height, &stepBytesSrcwBorder8u);
    pFilteredImage32f = nppiMalloc_32f_C1(imageROISize.width, imageROISize.height, &stepBytesFiltered32f);   
    pFilteredImage8u = nppiMalloc_8u_C1(imageROISize.width, imageROISize.height, &stepBytesFiltered8u);
    
    cudaMemcpy2D(pSrcImage8u, stepBytesSrc8u, &inputImage.data[0], inputImage.step[0], imageROISize.width, imageROISize.height, cudaMemcpyHostToDevice);
 
    cudaStatus = cudaMallocPitch(&pSqrIntegralImage64f, (size_t*) &stepBytesSqrIntegral64f, integralImageROISize.width * sizeof(Npp64f), integralImageROISize.height);
    //Set result gpu image to 0
    //nppiSet_32f_C1R((Npp32f) 0.0f, pFilteredImage32f, stepBytesFiltered32f, {imageROISize.width, imageROISize.height});
    // Mirror border for full image filtering
    status = nppiCopyWrapBorder_8u_C1R(pSrcImage8u, stepBytesSrc8u, imageROISize, pSrcwBorderImage8u, stepBytesSrcwBorder8u, imageROIwBorderSize, imageTopLeftOffset, imageTopLeftOffset);
    // Compute Sqr Integral Image
    status = nppiSqrIntegral_8u32f64f_C1R(pSrcwBorderImage8u, stepBytesSrcwBorder8u, pIntegralImage32f, stepBytesIntegral, pSqrIntegralImage64f, stepBytesSqrIntegral64f, imageROIwBorderSize, 0, 0);
    //Convert input image to 32f format
    status = nppiConvert_8u32f_C1R(pSrcwBorderImage8u, stepBytesSrcwBorder8u, pSrcwBorderImage32f, stepBytesSrcwBorder32f, imageROIwBorderSize);
    // Normalize input
    status = nppiMulC_32f_C1IR(normFactor32f, pSrcwBorderImage32f, stepBytesSrcwBorder32f, imageROIwBorderSize);
    status = nppsMulC_64f_I(normFactor64f, pSqrIntegralImage64f, integralImageROISize.height*stepBytesSqrIntegral64f/sizeof(Npp64f));   
    status = nppsConvert_64f32f(pSqrIntegralImage64f, pIntegralImage32f, integralImageROISize.height*stepBytesSqrIntegral64f/sizeof(Npp64f));
    //Normalize converted image
    //ippiMulC_32f_C1IR(normFactor, pSrc32fImage, stepBytesSrc, imageROISize);

    //timer start
    //cudaEventRecord(start); 
    //Applying USM Filter
    //OLD status =nppiFilterUnsharpBorder_32f_C1R(pSrcwBorderImage, stepBytesSrcwBorder, {imageTopLeftOffset,imageTopLeftOffset}, pUSMImage, stepBytesUSM, {imageROIwBorderSize.width, imageROIwBorderSize.height}, usmRadius, kernelStd, lambda, 2, NPP_BORDER_REPLICATE, pUSMBuffer);
    //Gossens version doesnt works with normalized images
    //ippiMulC_32f_C1IR(scaleFactor, pSrcwBorderImage, stepBytesSrcwBorder, imageROIwBorderSize);
    //ippiMulC_32f_C1IR(scaleFactor, pUSMImage, stepBytesUSM, imageROIwBorderSize);
    //Aplying DNLM filter
    //OLD this->dnlmFilter.dnlmFilter(pSrcwBorderImage, stepBytesSrcwBorder, CV_32FC1, pUSMImage, stepBytesUSM, pFilteredImage32f, stepBytesFiltered32f,  {imageROIwBorderSize.width, imageROIwBorderSize.height}, wSize, wSize_n, sigma_r);
    DNLM_OpenACC(pSrcwBorderImage32f, stepBytesSrcwBorder32f, pIntegralImage32f, stepBytesIntegral, pFilteredImage32f, stepBytesFiltered32f, windowRadius, neighborRadius, imageROISize.width, imageROISize.height, wSize , wSize, wSize_n, wSize_n, sigma_r);

    //Measure slapsed time
    //cudaEventRecord(stop);
    //Unormalize
    status = nppiMulC_32f_C1IR(scaleFactor, pFilteredImage32f, stepBytesFiltered32f, imageROISize);
    //Convert back to uchar, add offset to pointer to remove border
    nppiConvert_32f8u_C1R(pFilteredImage32f, stepBytesFiltered32f, pFilteredImage8u, stepBytesFiltered8u, imageROISize, NPP_RND_FINANCIAL);

    cudaMemcpy2D(&outputImage.data[0], outputImage.step[0], pFilteredImage8u ,stepBytesFiltered8u, imageROISize.width, imageROISize.height, cudaMemcpyDeviceToHost);
    
    //cudaEventSynchronize(stop);
    //float elapsed = 0;
   // cudaEventElapsedTime(&elapsed, start, stop);
    //Freeing memory
    nppiFree(pSrcImage8u); 
    nppiFree(pSrcwBorderImage8u);
    nppiFree(pSrcwBorderImage32f);
    nppiFree(pFilteredImage32f);
    nppiFree(pIntegralImage32f);
    nppsFree(pSqrIntegralImage64f);
    //nppiFree(pUSMImage);
    nppiFree(pFilteredImage8u);    
    //nppsFree(pUSMBuffer);
    //cout << elapsed/1000 <<endl;

    return outputImage;
}
