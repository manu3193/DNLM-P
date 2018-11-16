
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
        cout << "Using default parameters W=21x21, W_n=7x7, sigma_r=10, lambda=1, USM_len=16,USM_std=3" << endl;
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

    //Init IPP library
    IppStatus status = ippStsNoErr;
    status = ippInit();
    if(status != ippStsNoErr)
        cerr <<"Processor not identified"<<endl;
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
    
    //Status variable helps to check for errors
    IppStatus status = ippStsNoErr;

    //Pointers to IPP type images 
    Ipp32f *pSrc32fImage = NULL, *pSrcwBorderImage = NULL, *pUSMImage = NULL, *pFilteredImage= NULL;
    //Pointers to NPP type images
    Npp32f *pSrcwBorderImageGpu = NULL, *pUSMImageGpu = NULL, *pFilteredImageGpu = NULL;
    //Variable to store image step size in bytes 
    int stepBytesSrc = 0;
    int stepBytesSrcwBorder = 0;
    int stepBytesUSM = 0;
    int stepBytesFiltered = 0;
    int stepBytesSrcwBorderGpu = 0;
    int stepBytesUSMGpu = 0;
    int stepBytesFilteredGpu = 0;

    //Scale factors to normalize and denormalize 32f image
    Ipp32f normFactor = 1.0/255.0; 
    Ipp32f scaleFactor = 255.0; 

    //cv::Mat output filtered image
    Mat outputImage = Mat(srcImage.size(), srcImage.type());

    //Variables used for image format conversion
    IppiSize imageROISize, imageROIwBorderSize;
    imageROISize.width = srcImage.size().width;
    imageROISize.height = srcImage.size().height;

    //Get pointer to src and dst data
    Ipp8u *pSrcImage = (Ipp8u*)&srcImage.data[0];                               
    Ipp8u *pDstImage = (Ipp8u*)&outputImage.data[0];   

    //Compute border offset for border replicated image
    const int imageTopLeftOffset = floor(wSize_n/2);

    imageROIwBorderSize = {imageROISize.width + 2*imageTopLeftOffset, imageROISize.height + 2*imageTopLeftOffset};     
    
    //Allocate memory for cpu images
    pSrc32fImage = ippiMalloc_32f_C1(imageROISize.width, imageROISize.height, &stepBytesSrc); 
    pSrcwBorderImage = ippiMalloc_32f_C1(imageROIwBorderSize.width, imageROIwBorderSize.height, &stepBytesSrcwBorder);
    pUSMImage = ippiMalloc_32f_C1(imageROIwBorderSize.width, imageROIwBorderSize.height, &stepBytesUSM);
    pFilteredImage = ippiMalloc_32f_C1(imageROIwBorderSize.width, imageROIwBorderSize.height, &stepBytesFiltered);   
    //Allocate memory for gpu images
    pSrcwBorderImageGpu = nppiMalloc_32f_C1(imageROIwBorderSize.width, imageROIwBorderSize.height, &stepBytesSrcwBorderGpu);
    pUSMImageGpu = nppiMalloc_32f_C1(imageROIwBorderSize.width, imageROIwBorderSize.height, &stepBytesUSMGpu);
    pFilteredImageGpu = nppiMalloc_32f_C1(imageROIwBorderSize.width, imageROIwBorderSize.height, &stepBytesFilteredGpu);

    //Set result gpu image to 0
    nppiSet_32f_C1R((Npp32f) 0.0f, pFilteredImageGpu, stepBytesFilteredGpu, {imageROIwBorderSize.width, imageROIwBorderSize.height});
    
    //Convert input image to 32f format
    ippiConvert_8u32f_C1R(pSrcImage, srcImage.step[0], pSrc32fImage, stepBytesSrc, imageROISize);
    //Normalize converted image
    //ippiMulC_32f_C1IR(normFactor, pSrc32fImage, stepBytesSrc, imageROISize);

    // Mirror border for full image filtering
    status = ippiCopyMirrorBorder_32f_C1R(pSrc32fImage, stepBytesSrc, imageROISize, pSrcwBorderImage, stepBytesSrcwBorder, imageROIwBorderSize, imageTopLeftOffset, imageTopLeftOffset);
    //timer start
    double start = omp_get_wtime();
    //Applying USM Filter
    this->noAdaptiveUSM.noAdaptiveUSM(pSrcwBorderImage, stepBytesSrcwBorder, pUSMImage, stepBytesUSM, imageROIwBorderSize, kernelStd, lambda, kernelLen);
    //Gossens version doesnt works with normalized images
    //ippiMulC_32f_C1IR(scaleFactor, pSrcwBorderImage, stepBytesSrcwBorder, imageROIwBorderSize);
    //ippiMulC_32f_C1IR(scaleFactor, pUSMImage, stepBytesUSM, imageROIwBorderSize);
    //Copy images to GPU
    cudaMemcpy2D((void *) pSrcwBorderImageGpu, stepBytesSrcwBorderGpu, (void *) pSrcwBorderImage, stepBytesSrcwBorder, 
                 imageROIwBorderSize.width, imageROIwBorderSize.height, cudaMemcpyHostToDevice);
    cudaMemcpy2D((void *) pUSMImageGpu, stepBytesUSM, (void *) pUSMImage, stepBytesUSM, 
                 imageROIwBorderSize.width, imageROIwBorderSize.height, cudaMemcpyHostToDevice);
    //Aplying DNLM filter
    this->dnlmFilter.dnlmFilter(pSrcwBorderImageGpu, stepBytesSrcwBorderGpu, CV_32FC1, pUSMImageGpu, stepBytesUSM, pFilteredImageGpu, stepBytesFilteredGpu,  {imageROIwBorderSize.width, imageROIwBorderSize.height}, wSize, wSize_n, sigma_r);
    //Measure slapsed time
    double elapsed = omp_get_wtime() - start;
    cudaMemcpy2D((void *) pFilteredImage, stepBytesFiltered, (void *) pFilteredImageGpu, stepBytesFilteredGpu,
                 imageROIwBorderSize.width, imageROIwBorderSize.height, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    //Convert back to uchar, add offset to pointer to remove border
    ippiConvert_32f8u_C1R((Ipp32f*) (pFilteredImage + imageTopLeftOffset*stepBytesFiltered/sizeof(Ipp32f)+imageTopLeftOffset), stepBytesFiltered, pDstImage , outputImage.step[0], imageROISize, ippRndFinancial);
    //ippiConvert_32f8u_C1R(pUSMImage, stepBytesUSM, pDstImage , outputImage.step[0], imageROISize, ippRndFinancial);
    
    //Freeing memory
    ippiFree(pSrc32fImage);
    ippiFree(pSrcwBorderImage);
    ippiFree(pUSMImage);
    ippiFree(pFilteredImage);

    cout <<"Elapsed time: "<< elapsed << " s"<< endl;

    return outputImage;
}
