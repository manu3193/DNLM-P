
#include <ParallelDNLM.hpp>

using namespace std;
  


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

    double start = omp_get_wtime();
    outputImage = parallelDNLM.processImage(inputImage, windowSize, neighborhoodSize, sigma);
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
Mat ParallelDNLM::filterDNLM(const Mat& srcImage, int wSize, int wSize_n, float sigma_r){
    
    //Status variable helps to check for errors
    IppStatus status = ippStsNoErr;

    //Pointers to IPP type images 
    Ipp32f *pSrc32fImage __attribute__((aligned(64)));
    Ipp32f *pSrcwBorderImage __attribute__((aligned(64)));
    //Ipp32f *pUSMImage __attribute__((aligned(64)));
    Ipp32f *pFilteredImage __attribute__((aligned(64)));

    //Variable to store image step size in bytes 
    int stepBytesSrc = 0;
    int stepBytesSrcwBorder = 0;
    //int stepBytesUSM = 0;
    int stepBytesFiltered = 0;

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
    const int windowTopLeftOffset = floor(wSize_n/2);
    const int imageTopLeftOffset = floor(wSize/2) + windowTopLeftOffset;

    imageROIwBorderSize = {imageROISize.width + 2*imageTopLeftOffset, imageROISize.height + 2*imageTopLeftOffset};     

    //Allocate memory for images
    pSrc32fImage = ippiMalloc_32f_C1(imageROISize.width, imageROISize.height, &stepBytesSrc); 
    pSrcwBorderImage = ippiMalloc_32f_C1(imageROIwBorderSize.width, imageROIwBorderSize.height, &stepBytesSrcwBorder);
    //pUSMImage = ippiMalloc_32f_C1(imageROIwBorderSize.width, imageROIwBorderSize.height, &stepBytesUSM);
    pFilteredImage = ippiMalloc_32f_C1(imageROISize.width, imageROISize.height, &stepBytesFiltered);   
    
    //Convert input image to 32f format
    ippiConvert_8u32f_C1R(pSrcImage, srcImage.step[0], pSrc32fImage, stepBytesSrc, imageROISize);
    //Normalize converted image
    ippiMulC_32f_C1IR(normFactor, pSrc32fImage, stepBytesSrc, imageROISize);

    // Mirror border for full image filtering
    status = ippiCopyMirrorBorder_32f_C1R(pSrc32fImage, stepBytesSrc, imageROISize, pSrcwBorderImage, stepBytesSrcwBorder, imageROIwBorderSize, imageTopLeftOffset, imageTopLeftOffset);
    //timer start
    //timerStart();
    
    //this->noAdaptiveUSM.noAdaptiveUSM(pSrcwBorderImage, stepBytesSrcwBorder, pSrcwBorderImage, stepBytesSrcwBorder, imageROIwBorderSize, kernelStd, lambda, kernelLen);
    this->dnlmFilter.dnlmFilter(pSrcwBorderImage, stepBytesSrcwBorder, CV_32FC1, pSrcwBorderImage, stepBytesSrcwBorder, pFilteredImage, stepBytesFiltered, imageROISize, wSize, wSize_n, sigma_r);

    //double time = timerStop();

    //putting back everything
    ippiMulC_32f_C1IR(scaleFactor, pFilteredImage, stepBytesFiltered, imageROISize);
    ippiConvert_32f8u_C1R(pFilteredImage, stepBytesFiltered, pDstImage , outputImage.step[0], imageROISize, ippRndFinancial);
    
    //Freeing memory
    ippiFree(pSrc32fImage);
    ippiFree(pSrcwBorderImage);
    //ippiFree(pUSMImage);
    ippiFree(pFilteredImage);

    //cout << time << endl;

    return outputImage;
}
