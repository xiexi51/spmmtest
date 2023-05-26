#include "spmm_opt2.h"
#include "data.h"
#include <string>
#include <iostream>
#define CONSTINT const int

using namespace std;

extern string base_dir, graph;

const int WARPS_PER_BLOCK = 12;

__global__ void spmm_kernel_opt2(const int *_warp4, const int *idx, const float *val, const float *vin, float *vout, const int num_v, const int num_e, const int feat_in, const int num_warps)
{
    const int4 *warp4 = reinterpret_cast<const int4 *>(_warp4);
    // extern __shared__ float out_cache[];

    CONSTINT dim_mul = (feat_in + 31) / 32;
    CONSTINT round_dim = dim_mul * 32;

#pragma unroll
    for (int ext = 0; ext < dim_mul; ext++)
    {
        const int tid = blockIdx.x * blockDim.x * dim_mul + threadIdx.x + ext * blockDim.x; 
        const int warpid = tid / round_dim;                           
        const int block_warpid = threadIdx.x / round_dim;            
        const int laneid = threadIdx.x % round_dim;   
        if (warpid >= num_warps || laneid >= feat_in)
            return; 
        const int4 w_info = warp4[warpid];
        CONSTINT warp_row = w_info.x;
        CONSTINT warp_loc = w_info.y;
        CONSTINT warp_len = w_info.z;

        float tmp = 0;
#pragma unroll
        for (int i = 0; i < warp_len; i++)
        {
            const int nz_loc = warp_loc + i;
            const float left_val = __ldg(val + nz_loc);
            const float right_val = vin[__ldg(idx + nz_loc) * feat_in + laneid];
            tmp += left_val * right_val;
        }
        atomicAdd(&vout[warp_row * feat_in + laneid], tmp);
    }
}

void SPMM_OPT2::run(int dim)
{
    spmm_kernel_opt2<<<grid, block>>>(_warp4, idx, val, vin, vout, num_v, num_e, dim, num_warps);
}

double SPMM_OPT2::do_test(bool timing, int dim)
{
    this->num_warps = cuda_read_array(&this->_warp4, "/home/xix22010/py_projects/graph_preprocess/warp_4/" + this->_graph + ".warp4") / 4;
    int block_num = (num_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (!timing)
    {
        cout << "block num = " << block_num << endl;
    }

    grid.x = block_num;
    block.x = WARPS_PER_BLOCK * 32;

    double ret = timing_body(timing, dim);

    cudaFree(this->_warp4);
    return ret;
}