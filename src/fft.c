#include "fft.h"

void computeCorr(double _Complex * restrict signal, double _Complex * restrict kernel, double _Complex * restrict res, int width, int height){
        #pragma acc loop vector collapse(2) 
        for ( int row = 0; row < height; row++){
                for(int col = 0; col < width; col++){
                        res[col + (row * width)] = signal[col + (row * width)] * conj(kernel[col + (row * width)]);
                }
        }
}

void zeroPadding(float * restrict image, int step, double * restrict paddedImage, int height, int width, int padded_size){
      #pragma acc  loop vector collapse(2)
        for(int row = 0; row < height; row++){
                for(int col = 0; col < width; col++){
                        paddedImage[col + (row * padded_size)] = image[col + (row * step)];
                }
        }
}

void compute2D_C2RInvFFT(double _Complex * restrict inputImage, int step, double * restrict outputImage, int width, int height, double _Complex * restrict pBuffer) {
        #pragma acc loop vector 
        for (int row = 0; row < height; row++) {
                compute1D_C2CInvFFT(&inputImage[row*step], 1, width, &pBuffer[row*step]);
        }
        #pragma acc  loop vector 
        for(int col = 0; col < width; col++){
                compute1D_C2RInvFFT(&pBuffer[col], width, height, &outputImage[col]);
        }
}

void compute1D_C2RInvFFT(double _Complex * restrict image, int step, int N, double * restrict outputImage){
	#pragma acc loop seq
        for ( int k = 0; k < N; k++){
            double _Complex sumEven = 0.0 + 0.0 * I;
            double _Complex sumOdd = 0.0 + 0.0 * I;
            for ( int n = 0; n <= (N / 2) - 1; n++){
                double _Complex comp = image[2 * n * step];
                double _Complex eThetta = eIThettaInv(k, N, n, 0);
                double _Complex resultEven = comp * eThetta;
                sumEven += resultEven;

                double _Complex compOdd = image[(2 * n * step) + step];
                double _Complex eThettaOdd = eIThettaInv(k, N, n, 1);
                double _Complex resultOdd = compOdd * eThettaOdd;
                sumOdd = resultOdd + sumOdd;
            }
            outputImage[k*step] = creal((sumEven + sumOdd) / N);
        }
}

void compute1D_C2CInvFFT(double _Complex * restrict image, int step, int N, double _Complex * restrict outputImage){
	#pragma acc loop seq
        for ( int k = 0; k < N; k++){
            double _Complex sumEven = 0.0 + 0.0 * I;
            double _Complex sumOdd = 0.0 + 0.0 * I;
            for ( int n = 0; n <= (N / 2) - 1; n++){
                double _Complex comp = image[2 * n * step];
                double _Complex eThetta = eIThettaInv(k, N, n, 0);
                double _Complex resultEven = comp * eThetta;
                sumEven += resultEven;

                double _Complex compOdd = image[(2 * n * step) + step];
                double _Complex eThettaOdd = eIThettaInv(k, N, n, 1);
                double _Complex resultOdd = compOdd * eThettaOdd;
                sumOdd = resultOdd + sumOdd;
            }
            outputImage[k*step] = (sumEven + sumOdd) / N;
        }
}


void compute2D_R2CFFT(double * restrict inputImage, int step, double _Complex * restrict outputImage, int width, int height, double _Complex * restrict pBuffer) {
        #pragma acc loop vector    
        for (int row = 0; row < height; row++) {
                compute1D_R2CFFT(&inputImage[row*step], 1, width, &pBuffer[row*step]);
        }
        #pragma acc loop vector
        for(int col = 0; col < width; col++){
                compute1D_C2CFFT(&pBuffer[col], width, height, &outputImage[col]);
        }
}

void compute1D_R2CFFT(double * restrict inputSignal, int step, int N, double _Complex * restrict outputSignal){
        #pragma acc loop seq 
        for (int k = 0; k < N; k++) {
               double _Complex sumEven = 0.0 + 0.0 * I;
               double _Complex sumOdd = 0.0 + 0.0 * I;
               for (int n = 0; n <= (N / 2) - 1; n++) {

                       double comp = inputSignal[2*n*step];
                       double _Complex eThetta = eIThetta(k, N, n, 0);
                       double _Complex resultEven = comp * eThetta;
                       sumEven += resultEven;

                       double compOdd = inputSignal[2*n*step + step];
                       double _Complex eThettaOdd = eIThetta(k, N, n, 1);
                       double _Complex resultOdd = compOdd * eThettaOdd;
                       sumOdd += resultOdd;
               }
               outputSignal[k*step] = sumEven + sumOdd;
       }
}

void compute1D_C2CFFT( double _Complex * restrict inputSignal, int step, int N, double _Complex * restrict outputSignal){
	#pragma acc loop seq
        for (int k = 0; k < N; k++) {
               double _Complex sumEven = 0.0 + 0.0 * I;
               double _Complex sumOdd = 0.0 + 0.0 * I;
               for (int n = 0; n <= (N / 2) - 1; n++) {

                       double _Complex comp = inputSignal[2*n*step];
                       double _Complex eThetta = eIThetta(k, N, n, 0);
                       double _Complex resultEven = comp * eThetta;
                       sumEven += resultEven;

                       double _Complex compOdd = inputSignal[2*n*step + step];
                       double _Complex eThettaOdd = eIThetta(k, N, n, 1);
                       double _Complex resultOdd = compOdd * eThettaOdd;
                       sumOdd += resultOdd;
               }
               outputSignal[k*step] = sumEven + sumOdd;
       }
}

//computes the spin of the signal around a circle at its frequency
double _Complex eIThetta(int k, int N, int n, int offset) {
        // compute real part
        double realPart = cos((2 * PI * (2 * n + offset) * k) / N);

        // compute imaginary part
        double imaginaryPart = -1 *sin((2 * PI * (2 * n + offset) * k) / N);

        // create a _Complex number out of them and return it
        double _Complex result = realPart + imaginaryPart * I;
        return result;
}

double _Complex eIThettaInv(int k, int N, int n, int offset) {
        // compute real part
        double realPart = cos((2 * PI * (2 * n + offset) * k) / N);

        // compute imaginary part
        double imaginaryPart = sin((2 * PI * (2 * n + offset) * k) / N);

        // create a _Complex number out of them and return it
        double _Complex result = realPart + imaginaryPart * I;
        return result;
}



