#include "spmm_opt2_sparse.h"
#include "data.h"
#include <string>
#include <iostream>
#define CONSTINT const int

using namespace std;

extern string base_dir, graph;

const int WARPS_PER_BLOCK = 12;

__global__ void spmm_kernel_opt2_sparse(const int *_warp4, const int *idx, const float *val, const float *vin_data, const int *vin_selector, float *vout, const int num_v, const int num_e, const int feat_in, const int dim_sparse, const int num_warps)
{
    const int4 *warp4 = reinterpret_cast<const int4 *>(_warp4);
    // extern __shared__ float out_cache[];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    const int warpid = tid / dim_sparse; 
    const int laneid = threadIdx.x % dim_sparse;   
    if (warpid >= num_warps )
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
        const int right_loc = __ldg(idx + nz_loc) * dim_sparse + laneid;
        const float right_val = vin_data[right_loc];
        atomicAdd(&vout[warp_row * feat_in + __ldg(vin_selector + right_loc)], left_val * right_val);
    }
        
    
}

void SPMM_OPT2_SPARSE::run(int dim)
{
    spmm_kernel_opt2_sparse<<<grid, block>>>(_warp4, idx, val, vin, vin_sparse_selector, vout, num_v, num_e, dim, dim_sparse, num_warps);
}

double SPMM_OPT2_SPARSE::do_test(bool timing, int dim)
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