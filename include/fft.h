#ifndef FFT_H_
#define FFT_H_

#include <stdlib.h>
#include <complex.h>
#include <accelmath.h>

#define PI 3.14159265

#pragma acc routine seq 
extern void compute2D_R2CFFT(double * restrict, int, double _Complex * restrict, int, int, double _Complex * restrict);
#pragma acc routine seq
extern void compute2D_C2RInvFFT(double _Complex * restrict, int, double * restrict, int, int, double _Complex * restrict);
#pragma acc routine seq
extern void computeCorr(double _Complex * restrict, double _Complex * restrict, double _Complex * restrict, int, int);
#pragma acc routine seq
extern void zeroPadding(float * restrict, int, double * restrict, int, int, int);
#pragma acc routine seq
extern void compute1D_R2CFFT(double * restrict, int, int, double _Complex * restrict);
#pragma acc routine seq
extern void compute1D_C2CFFT(double _Complex * restrict, int, int, double _Complex * restrict);
#pragma acc routine seq
extern void compute1D_C2CInvFFT(double _Complex * restrict, int, int, double _Complex * restrict);
#pragma acc routine seq
extern void compute1D_C2RInvFFT(double _Complex * restrict, int, int, double * restrict);

#pragma acc routine seq
extern double _Complex eIThetta(int, int, int, int);
#pragma acc routine seq
extern double _Complex eIThettaInv(int, int, int, int);

#endif 
