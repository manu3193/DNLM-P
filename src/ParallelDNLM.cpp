
#include <ParallelDNLM.hpp>

using namespace std;
  
extern "C" void DNLM_OpenACC(const float* pSrcBorder, int stepBytesSrcBorder, float* pDst, int stepBytesDst, int windowRadius, int neighborRadius, int imageWidth, int imageHeight, int windowWidth, int windowHeight, int neighborWidth, int neighborHeight, float sigma_r);


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
    int device;
    cudaError_t cudaStatus = cudaSuccess;
    //Create CUDA events to measure execution time
    cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    //Status variable helps to check for errors
    NppStatus status =  NPP_NO_ERROR;
    
    cudaStatus = cudaGetDevice(&device);
    /*if (cudaStatus == cudaSuccess)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        cout <<"Device Number: " <<device <<endl;
        cout <<"  Device name: " <<prop.name <<endl;
        cout <<"  Memory Clock Rate (KHz): "<<prop.memoryClockRate<<endl;
        cout <<"  Memory Bus Width (bits): "<<prop.memoryBusWidth<<endl;
        cout <<"  Peak Memory Bandwidth (GB/s): "<<2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 <<endl;
        cout <<"  Compute Capability: "<<prop.major<<"."<<prop.minor<<endl;
    } */   
    //Pointers to NPP type images 
    Npp32f *pSrcImage32f = NULL, *pSrcwBorderImage = NULL, *pFilteredImage32f = NULL;
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

    //Create output Mat
    Mat outputImage = Mat(inputImage.size(), inputImage.type());

    //Variables used for image format conversion
    NppiSize imageROISize, imageROIwBorderSize;
    imageROISize.width = inputImage.size().width;
    imageROISize.height = inputImage.size().height;

    //Compute border offset for border replicated image
    const int windowRadius = floor(wSize/2);
    const int neighborRadius = floor(wSize_n/2);
    const int imageTopLeftOffset = windowRadius + neighborRadius;
    imageROIwBorderSize.width = imageROISize.width + 2*imageTopLeftOffset;
    imageROIwBorderSize.height = imageROISize.height + 2*imageTopLeftOffset;
    //Allocate memory for gpu images
    pSrcImage8u = nppiMalloc_8u_C1(imageROISize.width, imageROISize.height, &stepBytesSrc8u);
    pSrcImage32f = nppiMalloc_32f_C1(imageROISize.width, imageROISize.height, &stepBytesSrc32f); 
    pSrcwBorderImage = nppiMalloc_32f_C1(imageROIwBorderSize.width, imageROIwBorderSize.height, &stepBytesSrcwBorder);
    pFilteredImage32f = nppiMalloc_32f_C1(imageROISize.width, imageROISize.height, &stepBytesFiltered32f);   
    pFilteredImage8u = nppiMalloc_8u_C1(imageROISize.width, imageROISize.height, &stepBytesFiltered8u);
    
    if (pSrcImage8u ==NULL) cout << "error alocating pSrcImage8u" << endl;
    if (pSrcImage32f ==NULL) cout << "error alocating pSrcImage32f" << endl;
    if (pSrcwBorderImage ==NULL) cout << "error alocating pSrcwBorderImage" << endl;
    if (pFilteredImage32f ==NULL) cout << "error alocating pFilteredImage32f" << endl;
    if (pFilteredImage8u ==NULL) cout << "error alocating pFilteredImage8u" << endl;
    
    //Copy images to gpu
    cudaStatus = cudaMemcpy2D(pSrcImage8u, stepBytesSrc8u, inputImage.data, inputImage.step[0], imageROISize.width, imageROISize.height, cudaMemcpyHostToDevice);
    //Convert input image to 32f format
    status = nppiConvert_8u32f_C1R(pSrcImage8u, stepBytesSrc8u, pSrcImage32f, stepBytesSrc32f, imageROISize);
    //Normalize converted image
    status = nppiMulC_32f_C1IR(normFactor, pSrcImage32f, stepBytesSrc32f, imageROISize);

    // Mirror border for full image filtering
    status = nppiCopyWrapBorder_32f_C1R(pSrcImage32f, stepBytesSrc32f, imageROISize, pSrcwBorderImage, stepBytesSrcwBorder, imageROIwBorderSize, imageTopLeftOffset, imageTopLeftOffset);
    if (status !=0) cout << " error 4 " << status << endl;
    //timer start
    //cudaEventRecord(start); 
    //this->dnlmFilter.dnlmFilter(pSrcwBorderImage, stepBytesSrcwBorder, CV_32FC1, pFilteredImage32f, stepBytesFiltered32f,  imageROISize, wSize, wSize_n, sigma_r);
    //Measure slapsed time
    DNLM_OpenACC(pSrcwBorderImage, stepBytesSrcwBorder, pFilteredImage32f, stepBytesFiltered32f, windowRadius, neighborRadius, imageROISize.width, imageROISize.height, wSize , wSize, wSize_n, wSize_n, sigma_r);

    //cudaEventRecord(stop);
    //Normalize converted image
    status = nppiMulC_32f_C1IR(scaleFactor, pFilteredImage32f, stepBytesFiltered32f, imageROISize); 
    //Convert back to uchar, add offset to pointer to remove border
    status = nppiConvert_32f8u_C1R(pFilteredImage32f, stepBytesFiltered32f, pFilteredImage8u, stepBytesFiltered8u, imageROISize, NPP_RND_FINANCIAL);
    if (status !=0) cout << " error converting" << status << endl;
    cudaStatus = cudaMemcpy2D(outputImage.data, outputImage.step[0], pFilteredImage8u,
                 stepBytesFiltered8u, imageROISize.width, imageROISize.height, cudaMemcpyDeviceToHost);
    if (cudaStatus !=cudaSuccess) cout << " error copying back to host" << cudaStatus << endl;
    //cudaEventSynchronize(stop);
    //float elapsed = 0;
    //cudaEventElapsedTime(&elapsed, start, stop);
    //Freeing memory
    nppiFree(pSrcImage32f); 
    nppiFree(pSrcImage8u);
    nppiFree(pSrcwBorderImage);
    nppiFree(pFilteredImage32f);
    nppiFree(pFilteredImage8u);    
    //cout << elapsed/1000 <<endl;

    return outputImage;
}
