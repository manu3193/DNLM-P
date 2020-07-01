#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace std;
using namespace cv;
  
void DNLM_OpenACC(const float*, int, const float*, int, float*, int, int, int, int, int, int, int, int, int, float);


int main(int argc, char* argv[]){

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
    
    if(argc<=2){
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

    int windowRadius = floor(windowSize/2);
    int neighborRadius = floor(neighborhoodSize/2);
    int imageWidth = inputImage.cols;
    int imageHeight = inputImage.rows;
    int borderOffset = windowRadius+neighborRadius;
    cout << "border offset: "<<borderOffset<<endl;;
    Mat inputImageBorder;
    copyMakeBorder(inputImage, inputImageBorder, borderOffset, borderOffset, borderOffset, borderOffset, BORDER_REFLECT);

    Mat inputImage32f = Mat(inputImage.size(), CV_32FC1);
    inputImageBorder.convertTo(inputImage32f, CV_32FC1, 1/255.0);
    //Process image 
    Mat outputImage  = Mat(inputImage.size(), inputImage.type());
    Mat outputImage32f = Mat(inputImage.size(), inputImage32f.type());
    //Compute Integral Image
    Mat integralImage, sqrIntegralImage32f;
    integral(inputImage32f, integralImage, sqrIntegralImage32f, CV_32F, CV_32F);
 
    if ( ! inputImage32f.isContinuous() )
    { 
        inputImage32f = inputImage32f.clone();
    }
    
    double start = omp_get_wtime();
    DNLM_OpenACC(inputImage32f.ptr<float>(0), inputImage32f.step[0], sqrIntegralImage32f.ptr<float>(0), sqrIntegralImage32f.step[0], outputImage32f.ptr<float>(0), outputImage32f.step[0], windowRadius, neighborRadius, imageWidth, imageHeight, windowSize , windowSize, neighborhoodSize, neighborhoodSize, sigma);
    double elapsed = omp_get_wtime()-start;
    cout << elapsed<<endl;

    outputImage32f.convertTo(outputImage, CV_8UC1, 255.0);
    //Write image to output file.
    imwrite(outputFile, outputImage);
    pAt = outputFile.find_last_of('.'); 

    Mat baseOutput = imread(outputFile.substr(0, pAt) + "_base.png", IMREAD_GRAYSCALE);
  
    if (baseOutput.data!=NULL)
    {
        baseOutput.convertTo(baseOutput, CV_8UC1, 1.0);
        double msd = norm(baseOutput, outputImage);
        double pixels = (double) baseOutput.total();
        msd = msd * msd / pixels;
        cout << "MSE: "<< msd<<endl;
    }
    //Release object memory
    inputImage.release();
    outputImage.release();

    return 0;
    }
}



