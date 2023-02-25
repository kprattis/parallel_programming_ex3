#include "fglt.hpp" //mwIndex, mwSize
#include "fglt_cuda.cuh" //kernels

#include <iostream> //cout 
#include <cuda_runtime.h>

#define DEBUG_PRINT(a, id) std::cout << id  << ", " << toc(a) << std::endl

struct timeval tic(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv;
  }
    
static double toc(struct timeval begin){
    
    struct timeval end;
    gettimeofday(&end, NULL);

    double stime = ((double) (end.tv_sec - begin.tv_sec) * 1000 ) +
        ((double) (end.tv_usec - begin.tv_usec) / 1000 );
    stime = stime / 1000;

    return(stime);
}

int compute_gpu(double ** const f, double ** const fn, mwIndex *ii, mwIndex *jStart, mwSize n, mwSize m){

    struct timeval begin;
    begin = tic();

    //Init device
    cudaFree(0);
    
    #ifdef DEBUG 
        DEBUG_PRINT(begin, "GPU-INIT");
    #endif

    //device memory declaration
    double *dev_f, *C3;
    mwIndex *dev_ii, *dev_jStart;

    //allocate memory for GPU matrices

    cudaMalloc( (void **) &dev_ii, sizeof(mwIndex) * m);
    cudaMemcpyAsync( dev_ii, ii, sizeof(mwIndex) * m, cudaMemcpyHostToDevice);

    cudaMalloc( (void **) &dev_jStart, sizeof(mwIndex) * (n + 1));
    cudaMemcpyAsync( dev_jStart, jStart, sizeof(mwIndex) * (n + 1), cudaMemcpyHostToDevice);

    cudaMalloc( (void **) &dev_f, sizeof(double) * NGRAPHLET * n);
    cudaMalloc( (void **) &C3, sizeof(double) * m);

    //define grid and block dimensions
    dim3 BlockDim(BLOCK_SIZE);
    dim3 GridDim((n - BLOCK_SIZE + 1)/BLOCK_SIZE);
    
    dim3 GridDimMat((m - BlockDim.x + 1)/BlockDim.x);

    dim3 GridDimMatNET((n * NGRAPHLET - BLOCK_SIZE + 1)/BLOCK_SIZE);

    cudaDeviceSynchronize();

    #ifdef DEBUG 
        DEBUG_PRINT(begin, "DEVICE MEMORY ALLOCATION");
    #endif

    //kernel for p1 
    row_sum_red <<<GridDim, BlockDim>>> (dev_f + n, dev_ii, dev_jStart, n, m, NULL);
    
    //kernel for C3 matrix
    C3_calc <<<GridDimMat, BlockDim>>> (C3, dev_ii, dev_jStart, n, m);

    cudaDeviceSynchronize();

    #ifdef DEBUG 
        DEBUG_PRINT(begin, "Calculations of p1, C3");
    #endif


    //kernel for c3
    row_sum_red<<<GridDim, BlockDim>>>(dev_f + 4 * n, dev_ii, dev_jStart, n, m, C3);

    //kernel for p2
    p2 <<<GridDim, BlockDim>>> (dev_f + 2 * n, dev_f + n, dev_ii, dev_jStart, n, m);

    //kernel for d3
    d3 <<<GridDim, BlockDim>>> (dev_f + 3 * n, dev_f + n, n);

    cudaDeviceSynchronize();

    #ifdef DEBUG 
        DEBUG_PRINT(begin, "Calculations of p2, d3, d4");
    #endif
    
    //transfer back results to host
    for(int i = 1; i < NGRAPHLET; i++){
        cudaMemcpyAsync( f[i], dev_f + i * n, n * sizeof(double), cudaMemcpyDeviceToHost);
    }
    
    cudaDeviceSynchronize();
    
    #ifdef DEBUG 
        DEBUG_PRINT(begin, "Transfer dev_f -> f");
    #endif

    //raw to net frequencies
    raw2net<<<GridDimMatNET, BlockDim>>>(dev_f, n);

    cudaDeviceSynchronize();

    #ifdef DEBUG 
        DEBUG_PRINT(begin, "Calculation of net frequncies");
    #endif
    
    //Transfer back to memory the results

    cudaMemcpyAsync( f[0], dev_f, n * sizeof(double), cudaMemcpyDeviceToHost);
    for(int i = 0; i < NGRAPHLET; i++){
        cudaMemcpyAsync( fn[i], dev_f + i * n, n * sizeof(double), cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();

    #ifdef DEBUG 
        DEBUG_PRINT(begin, "Transfer dev_f -> fn");
    #endif

    cudaDeviceSynchronize();

    //free GPU memory
    cudaFree(dev_f);
    cudaFree(C3);
    
    cudaFree(dev_ii);
    cudaFree(dev_jStart);

    cudaDeviceSynchronize();
    
    #ifdef DEBUG 
        DEBUG_PRINT(begin, "Free device memory");
    #endif

    std::cout << "Time to execute GPU kernel, " << toc(begin) << std::endl;
    
    return 1;
}
