#ifndef FGLT_CUH_
#define FGLT_CUH_

#include "fglt.hpp"

#define BLOCK_SIZE 256

//used to calculate p1, c3
__global__ void row_sum_red(double * res, mwIndex *ii, mwIndex *jStart, mwSize n, mwSize m, double *vals);

//calculating matrix C3
__global__ void C3_calc(double *C3_vals, mwIndex *ii, mwIndex *jStart, mwSize n, mwSize m);

//calculate p2
__global__ void p2(double * res, double * p1, mwIndex *ii, mwIndex *jStart, mwSize n, mwSize m);

//calculate d3
__global__ void d3(double * f, double * p2, mwSize n);

//raw2net 
__global__ void raw2net(double * f, mwSize n);

#endif