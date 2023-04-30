#include "spmm_opt_innerloop.h"
#include "data.h"
#include <string>
#include <iostream>
#define CONSTINT const int

using namespace std;

extern string base_dir, graph;

const int WARPS_PER_BLOCK = 12;

__global__ void spmm_kernel_opt_innerloop(const int *_warp4, const int *idx, const float *val, const float *vin, float *vout, const int num_v, const int num_e, const int feat_in, const int num_warps)
{
    const int4 *warp4 = reinterpret_cast<const int4 *>(_warp4);
    extern __shared__ float out_cache[];

    CONSTINT dim_mul = (feat_in + 31) / 32;
    CONSTINT round_dim = dim_mul * 32;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warpid = tid / 32;
    const int block_warpid = threadIdx.x / 32;
    const int laneid = threadIdx.x % 32;
    if (warpid >= num_warps)
        return;

    const int4 w_info = warp4[warpid];
    CONSTINT warp_row = w_info.x;
    CONSTINT warp_loc = w_info.y;
    CONSTINT warp_len = w_info.z;

#pragma unroll
    for (int i = 0; i < warp_len; i++)
    {
        if (i == 0)
        {
            for (int d = laneid; d < feat_in; d += 32)
            {
                out_cache[block_warpid * round_dim + d] = 0;
            }
        }
        const int nz_loc = warp_loc + i;
        const float left_val = __ldg(val + nz_loc);

        for (int d = laneid; d < feat_in; d += 32)
        {
            const float right_val = vin[__ldg(idx + nz_loc) * feat_in + d];
            out_cache[block_warpid * round_dim + d] += left_val * right_val;
        }
    }
#pragma unroll
    for (int d = laneid; d < feat_in; d += 32)
    {
        atomicAdd(&vout[warp_row * feat_in + d], out_cache[block_warpid * round_dim + d]);
    }
}

void SPMM_OPT_INNERLOOP::run(int dim)
{
    spmm_kernel_opt_innerloop<<<grid, block, WARPS_PER_BLOCK *((dim + 31) / 32) * 32 * sizeof(float)>>>(_warp4, idx, val, vin, vout, num_v, num_e, dim, num_warps);
}

double SPMM_OPT_INNERLOOP::do_test(bool timing, int dim)
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