#include <ParallelDNLM.hpp>

using namespace std;
  
extern "C" void DNLM_OpenACC(const float*, int, const float*, int, float*, int, int, int, int, int, int, int, int, int, float);


int main(int argc, char* argv[]){
    ParallelDNLM parallelDNLM;

    //Parameters to read    
    string inputFileStr;
    int windowSize, neighborhoodSize;
    float sigma;
   
    string programName = argv[0];

    for(int i=1; i<argc; i++){
        string cur(argv[i]);
        if(cur == "-w"){
            if((i+1) < argc){
                i++;
                if(!(istringstream(argv[i]) >> windowSize)){
		    cerr << "[error] Expected valid windowSize value after '" << cur << "'." << endl;
	            exit(1);
		}
	    }else{
	        cerr << "[error] Expected search window size value after '" << cur << "'." << endl;
	        exit(1);
	    }
        }
        else if(cur == "-n"){
            if((i+1) < argc){
                i++;
                if(!(istringstream(argv[i]) >> neighborhoodSize)){
	            cerr << "[error] Expected valid neighborhood size value after '" << cur << "'." << endl;
		    exit(1);
		}	
            }else{
                cerr << "[error] Expected neighborhood size value after '" << cur << "'." << endl;
                exit(1);
            }
        }
        else if(cur == "-s"){
            if((i+1) < argc){
                i++;
                if(!(istringstream(argv[i]) >> sigma)){
                    cerr << "[error] Expected valid sigma value after '" << cur << "'." << endl;
                    exit(1);
                }
            }else{
                cerr << "[error] Expected sigma value after '" << cur << "'." << endl;
                exit(1);
            }
        }
        else{
            inputFileStr = cur;
        }
    }
    
    if(argc<2){
        cerr << "Usage: "<< programName << " [OPTION]... FILE"<<endl<<endl;
        cerr << "OPTION:"<<endl;
        cerr << "  -w          Length in pixels of the squared search window"<<endl;
        cerr << "  -n          Length in pixels of the squared pixel neighborhood"<<endl;
        cerr << "  -s          Smooth parameter" <<endl;
        return 0;
    }  

    //Check input arguments
    if (argc < 2){
        cerr << "ERROR: You must provide a valid image filename" << endl;
        exit(1);
    }else{
    
    //Open input image
    const string inputFile = inputFileStr;
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
    double start = omp_get_wtime();
    Mat outputImage = parallelDNLM.processImage(inputImage, windowSize, neighborhoodSize, sigma);
    double elapsed = omp_get_wtime()-start;
    cout << elapsed<<endl;
    //Write image to output file.
    imwrite(outputFile, outputImage);
    pAt = outputFile.find_last_of('.'); 
    cout <<outputFile.substr(0, pAt) + "_base.png"<<endl;
    Mat baseOutput = imread(outputFile.substr(0, pAt) + "_base.png", IMREAD_GRAYSCALE);
    double msd = norm(baseOutput, outputImage);
    double pixels = (double) baseOutput.total();
    msd = msd * msd / pixels;
    cout << "MSE: "<< msd<<endl;
    //Release object memory
    inputImage.release();
    outputImage.release();

    return 0;
    }
}


Mat ParallelDNLM::processImage(const Mat& inputImage, int wSize, int nSize, float sigma){
    //Set parameters for processing
    
    Mat fDeceivedNLM = filterDNLM(inputImage, wSize, nSize, sigma);

    return fDeceivedNLM;
}

//Input image must be from 0 to 255
Mat ParallelDNLM::filterDNLM(const Mat& inputImage, int wSize, int wSize_n, float sigma_r){
    
    //Get cuda device properties
    //int device;
    cudaError_t cudaStatus = cudaSuccess;
    //Create CUDA events to measure execution time
    //cudaEvent_t start, stop;
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
    size_t stepBytesSqrIntegral64f = 0;
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
    cudaStatus = cudaMallocPitch(&pSqrIntegralImage64f, &stepBytesSqrIntegral64f, integralImageROISize.width * sizeof(Npp64f), integralImageROISize.height);
    pSrcImage8u = nppiMalloc_8u_C1(imageROISize.width, imageROISize.height, &stepBytesSrc8u);
    pSrcwBorderImage32f = nppiMalloc_32f_C1(imageROIwBorderSize.width, imageROIwBorderSize.height, &stepBytesSrcwBorder32f); 
    pIntegralImage32f = nppiMalloc_32f_C1(integralImageROISize.width, integralImageROISize.height, &stepBytesIntegral);
    pSrcwBorderImage8u = nppiMalloc_8u_C1(imageROIwBorderSize.width, imageROIwBorderSize.height, &stepBytesSrcwBorder8u);
    pFilteredImage32f = nppiMalloc_32f_C1(imageROISize.width, imageROISize.height, &stepBytesFiltered32f);   
    pFilteredImage8u = nppiMalloc_8u_C1(imageROISize.width, imageROISize.height, &stepBytesFiltered8u);
    
    cudaMemcpy2D(pSrcImage8u, stepBytesSrc8u, &inputImage.data[0], inputImage.step[0], imageROISize.width, imageROISize.height, cudaMemcpyHostToDevice);
 
    //Set result gpu image to 0
    //nppiSet_32f_C1R((Npp32f) 0.0f, pFilteredImage32f, stepBytesFiltered32f, {imageROISize.width, imageROISize.height});
    // Mirror border for full image filtering
    status = nppiCopyWrapBorder_8u_C1R(pSrcImage8u, stepBytesSrc8u, imageROISize, pSrcwBorderImage8u, stepBytesSrcwBorder8u, imageROIwBorderSize, imageTopLeftOffset, imageTopLeftOffset);
    // Compute Sqr Integral Image
    status = nppiSqrIntegral_8u32f64f_C1R(pSrcwBorderImage8u, stepBytesSrcwBorder8u, pIntegralImage32f, stepBytesIntegral, pSqrIntegralImage64f, (int) stepBytesSqrIntegral64f, imageROIwBorderSize, 0, 0);
    //Convert input image to 32f format
    status = nppiConvert_8u32f_C1R(pSrcwBorderImage8u, stepBytesSrcwBorder8u, pSrcwBorderImage32f, stepBytesSrcwBorder32f, imageROIwBorderSize);
    // Normalize input
    status = nppiMulC_32f_C1IR(normFactor32f, pSrcwBorderImage32f, stepBytesSrcwBorder32f, imageROIwBorderSize);
    status = nppsConvert_64f32f(pSqrIntegralImage64f, pIntegralImage32f, stepBytesSqrIntegral64f/sizeof(Npp64f));
    status = nppiMulC_32f_C1IR(normFactor32f, pIntegralImage32f, stepBytesIntegral, integralImageROISize);
    //timer start
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
