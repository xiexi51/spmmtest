#include "spmm_ref.h"
#include <iostream>
using namespace std;
__global__ void spmm_kernel_ref(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_v) return;
    int begin = ptr[tid], end = ptr[tid + 1];
    for (int j = 0; j < INFEATURE; ++j)
    {
        float result = 0.0f;
        for (int i = begin; i < end; ++i)
        {
            result += vin[idx[i] * INFEATURE + j] * val[i];
        }
        vout[tid * INFEATURE + j] = result;
    }
}

void spmm_ref(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int num_e, int dim){
    int BLOCK_SIZE = 128;
    int grid = (num_v + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int block = BLOCK_SIZE;
    spmm_kernel_ref<<<grid, block>>>(ptr, idx, val, vin, vout, num_v, dim);
    
}



