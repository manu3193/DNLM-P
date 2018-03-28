
#include "DNLMFilter.hpp"

// Pre-process input and select appropriate filter.
int DNLMFilter::dnlmFilter(const Ipp32f* pSrc, int stepBytesSrc, int srcType, const Ipp32f* pUSMImage, int stepByteUSM, Ipp32f* pDst, int stepBytesDst, IppiSize imageSize, int w, int w_n, float sigma_r){

    int status;

     if (srcType == CV_32FC3){
    
        //apply DNLM for color images   
        //status = this->dnlmFilter(pSrc, stepBytesSrc, pUSMImage, stepByteUSM, pDst, stepBytesDst, imageSize, w, w_n, sigma_r);

     }

    else if (srcType == CV_32FC1 ){
        //apply DNLM for grayscale images   
        status = this->dnlmFilterBW(pSrc, stepBytesSrc, pUSMImage, stepByteUSM, pDst, stepBytesDst, imageSize, w, w_n, sigma_r);
    }
    
    else
         cout<< "Error in image format" << endl;
    return status;
}

//Implements dnlm filter for grayscale images.
int DNLMFilter::dnlmFilterBW(const Ipp32f* pSrc, int stepBytesSrc, const Ipp32f* pUSMImage, int stepByteUSM, Ipp32f* pDst, int stepBytesDst, IppiSize imageSize, int w, int w_n, float sigma_r){
    //Variable to store status
    int status;
    Ipp32f* pSrcBorder = NULL;
    int stepBytesSrcBorder = 0;

    //Compute border offset for border replicated image
    int borderOffset = floor(w/2)+floor(w_n/2);
    IppiSize imageBorderSize = {imageSize.width + borderOffset, imageSize.height + borderOffset};


    //Allocate memory for image with borders
    pSrcBorder = ippiMalloc_32f_C1(roi.width, roi.height, &stepBytesSrcBorder);
    //
    // Replicate border for full image filtering
    status = ippiCopyReplicateBorder_32f_C1R(pSrc, stepBytesSrc, imageSize, pSrcBorder, stepBytesSrcBorder, imageBorderSize, borderOffset, borderOffset);

    
}

Mat DNLMFilter::nlmfilBW_deceived(const Mat& A, const Mat& Laplacian, int w, int w_n, double sigma_d, int sigma_r){
    int iMin, iMax, jMin, jMax;
    Mat B, F, G, H, I, L, S, E;

    // Mat1i X,Y;
    // vector<Mat> channels(3);
    // double norm_F;

    // //Pre-compute Gaussian domain weights.
    // Tools::meshgrid(Range(-w,w),Range(-w,w),X,Y);
    // pow(X,2,X);
    // pow(Y,2,Y);
    // S = X+Y;
    // S.convertTo(S,CV_32F);
    // S /= (-2*pow(sigma_d,2));

    // exp(S,G);

    // //Apply nlm filter.
    // omp_set_num_threads(1);
    // B = Mat::zeros(A.size(),A.type());
    // cout << "Applying the deceived nlm filter..." << endl;
    
    // #pragma omp parallel for private(I,iMin,iMax,jMin,jMax/*,pixel*/,E,channels,H,F,norm_F,L) shared(A,B,G,Laplacian,w,sigma_d,sigma_r)
    // for(int i = 0; i < A.rows-1; i++){
    //    for(int j = 0; j < A.cols-1; j++){

    //        float val; //result value
    //        //Extract local region.
    //        iMin = max(i - w,0);
    //        iMax = min(i + w,A.rows-1);
    //        jMin = max(j - w,0);
    //        jMax = min(j + w,A.cols-1);


    //        //Exclude borders
    //        if((i>=w+w_n) && (j>=w+w_n) && (i<=A.rows-1-w-w_n) && (j<=A.cols-1-w-w_n)){

    //            I = A(Range(iMin-w_n,iMax+w_n+1), Range(jMin-w_n,jMax+w_n+1)); /*********************************************************/
    //            //Compute Gaussian range weights.
    //            //done in the three layers
    //            split(I,channels);
               
    //            E= CalcEuclideanDistMat(channels[0], w_n, i, j, iMin, jMin);
    //            exp(E / (-2 * pow(sigma_r,2)),H);


    //            //Calculate NLM filter response.
    //            F = H.mul(G(Range(iMin-i+w, iMax-i+w), Range(jMin-j+w, jMax-j+w)));
    //            norm_F = sum(F).val[0];

    //            //The laplacian deceive consists on weighting the gaussian function with
    //            //the original image, and using the image values of the laplacian image.
    //            L = Laplacian(Range(iMin,iMax), Range(jMin,jMax));
    //            split(L,channels);
    //            val =(sum(sum(F.mul(channels[0])))/norm_F).val[0];

    //          //Create black border
    //          }else{
    //              val=0;
    //          }            
    //              B.at<Vec3f>(i,j)[0] = val;
    //    }
    // }
    return B;

}

Mat DNLMFilter::CalcEuclideanDistMat(const Mat& I, int w_n, int i, int j, int iMin, int jMin){
    //int nMin_w, nMax_w, mMin_w, mMax_w, nMin_z, nMax_z, mMin_z, mMax_z;
    Mat Z, W, O;
    /*int size_x,size_y; //Window dimensions

    //NLM------------------------------------------------------------------------
    size_x=I.rows;
    size_y=I.cols;

    //Extract pixel z neighborhood local region.
    nMin_z = max(i - iMin - w_n,0);
    nMax_z = min(i - iMin + w_n,I.rows-1);
    mMin_z = max(j - jMin - w_n,0);
    mMax_z = min(j - jMin + w_n,I.cols-1);


    //Current Pixel z neighborhood
    Z= I(Range(nMin_z,nMax_z+1),Range(mMin_z,mMax_z+1));

    cout << Z.size() << endl;

    //Create output Mat
    O.create(size_x-1-2*w_n,size_y-1-2*w_n, DataType<float>::type);

     //Visit each pixel in the window I
     for(int n=w_n; n<size_x-w_n-1; n++){
         for(int m=w_n; m<size_y-w_n-1; m++){

            //Extract pixel w neighborhood local region.
            nMin_w = max(n - w_n,0);
            nMax_w = min(n + w_n,I.rows-1);
            mMin_w = max(m - w_n,0);
            mMax_w = min(m + w_n,I.cols-1);
            //Get pixel mini-window
            W = I(Range(nMin_w,nMax_w+1), Range(mMin_w,mMax_w+1));
            cout << W.size() << endl;
            float res=(float) norm(W,Z,NORM_L2);
            O.at<float>(n-w_n,m-w_n) = res;
         }
     }*/
     return O;

}
