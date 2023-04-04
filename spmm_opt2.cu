#include "spmm_opt2.h"
#include "data.h"
#include <string>
#include <iostream>
#define CONSTINT const int

using namespace std;

extern string base_dir, graph;

const int WARPS_PER_BLOCK = 12;

__global__ void spmm_kernel_opt2(int *_warp4, int *idx, float *val, float *vin, float *vout, int num_v, int num_e, int feat_in, int num_warps)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x; // global thread-id
    const int warpid = tid / 32;                     // global warp-id
    const int block_warpid = threadIdx.x / 32;       // block warp-id
    const int laneid = threadIdx.x % 32;             // warp thread-id -- laneid

    if(warpid >= num_warps)
        return;

    int4 *warp4 = reinterpret_cast<int4 *>(_warp4);
    const int4 w_info = warp4[warpid];

    CONSTINT warp_row = w_info.x;
    CONSTINT warp_loc = w_info.y;
    CONSTINT warp_len = w_info.z;

    extern __shared__ float out_cache[];

#pragma unroll
    for (int i = 0; i < warp_len; i++)
    {
        if (i == 0)
        {
            for (int j = 0; j < feat_in / 32; j++){
                out_cache[block_warpid * feat_in + j * 32 + laneid] = 0;
            }
            __syncwarp();
        }
        const int nz_loc = warp_loc + i;
        const float left_val = val[nz_loc];
#pragma unroll
        for (int j = 0; j < feat_in / 32; j++){
            const float right_val = vin[idx[nz_loc] * feat_in + j * 32 + laneid];
            out_cache[block_warpid * feat_in + j * 32 + laneid] += left_val * right_val;
        }
    }
    #pragma unroll
    for (int j = 0; j < feat_in / 32; j++){
        atomicAdd(&vout[warp_row * feat_in + j * 32 + laneid], out_cache[block_warpid * feat_in + j * 32 + laneid]);
    }

}

void SPMM_OPT2::run()
{
    spmm_kernel_opt2<<<grid, block, WARPS_PER_BLOCK * dim * sizeof(float)>>>(_warp4, idx, val, vin, vout, num_v, num_e, dim, num_warps);
}

double SPMM_OPT2::do_test(bool timing)
{
    this->num_warps = cuda_read_array(&this->_warp4, "/home/xix22010/py_projects/graph_preprocess/warp_4/" + graph + ".warp4") / 4;
    int block_num = (num_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (!timing)
    {
        cout << "block num = " << block_num << endl;
    }

    grid.x = block_num;
    block.x = WARPS_PER_BLOCK * 32;

    double ret = timing_body(timing);

    cudaFree(this->_warp4);
    return ret;
}