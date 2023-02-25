#include "fglt_cuda.cuh"

__global__ void row_sum_red(double * res, mwIndex *ii, mwIndex *jStart, mwSize n, mwSize m, double *vals){
    int row = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < n){
        mwIndex start = jStart[row];
        mwIndex end = jStart[row + 1];

        if(vals == NULL){
            res[row] = (end - start);
        }
        else{
            double sum = 0.0;

            for(int i = start; i < end; i++){
                sum += vals[i];
            }

            res[row] = sum / 2;
        }
    }
}

__global__ void C3_calc(double *C3_vals, mwIndex *ii, mwIndex *jStart, mwSize n, mwSize m){
    
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    //Calculate C3
    if(id < m){

        int sum;

        mwIndex start_r, end_r, start_c, end_c; 

        int row = ii[id];

        start_c = jStart[row];
        end_c = jStart[row + 1];

        //find column
        int j = start_c;
        for(j = start_c; j < end_c; j++){
            if(id < jStart[ii[j]]){
                break;
            }
        }

        start_r = jStart[ii[j - 1]];
        end_r = jStart[ii[j - 1] + 1];


        sum = 0;
        int k1 = start_r;
        int k2 = start_c;

        //int id1, id2;

        while(k1 < end_r && k2 < end_c){

            if(ii[k1] == ii[k2]){
                sum ++;
                k1++;
                k2++;
            }
            else if(ii[k1] > ii[k2]){
                k2++;
            }
            else{
                k1++;
            }
        }

        C3_vals[id] = sum;
    }

}

__global__ void p2(double *res, double * p1, mwIndex *ii, mwIndex *jStart, mwSize n, mwSize m){
    int row = threadIdx.x + blockIdx.x * blockDim.x;

    double temp = 0.0;

    if(row < n){

        mwSize start = jStart[row];
        mwSize end = jStart[row + 1];

        for(mwSize i = start; i < end; i++){
            temp += p1[ii[i]];
        }
    }

    if(temp < p1[row])
        res[row] = 0;
    else{
        res[row] = temp - p1[row];   
    }
    
}

__global__ void d3(double *f, double *p1, mwSize n){
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    double temp = 0.0;

    if(row < n){
        if(p1[row] > 1)
            temp = p1[row] - 1;
        
        temp = temp * p1[row] / 2;
    
        f[row] = temp;
    }

}

__global__ void raw2net(double * f, mwSize n){

    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id < n){
        f[id] = 1;
    }
    else if( 2 * n <= id && id < 3 * n){
        f[id] = f[id] - 2 * f[2 * n + id];
    }
    else if(3 * n <= id && id < 4 * n){
        f[id] = f[id] - f[id + n];
    }

}