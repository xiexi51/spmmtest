#include "spmm_opt2_sparse_backward_v3.h"
#include "data.h"
#include <string>
#include <iostream>
#include <assert.h>
#define CONSTINT const int

using namespace std;

extern string base_dir, graph;

const int WARPS_PER_BLOCK = 12;
const int EXT_WARP_DIM = 32;

__global__ void spmm_kernel_opt2_sparse_backward_v3(const int *_warp4, const int *idx, const float *val, const float *vin_data, const int *vin_selector, float *vout, const int num_v, const int num_e, const int feat_in, const int dim_sparse, const int num_warps)
{
    extern __shared__ float out_cache[];

    const int4 *warp4 = reinterpret_cast<const int4 *>(_warp4);

    const int total_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_warpid = total_tid / EXT_WARP_DIM;
    const int laneid = threadIdx.x % EXT_WARP_DIM;
    const int wid = threadIdx.x / EXT_WARP_DIM;

    int4 w_info;
    int warp_row, warp_loc, warp_len;

    if (total_warpid < num_warps)
    {
        w_info = warp4[total_warpid];
        warp_row = w_info.x;
        warp_loc = w_info.y;
        warp_len = w_info.z;
    }
    else
    {
        return;
    }

#pragma unroll
    for (int ext = 0; ext < (feat_in + EXT_WARP_DIM - 1) / EXT_WARP_DIM; ext++)
    {
        out_cache[wid * feat_in + laneid + ext * EXT_WARP_DIM] = vin_data[warp_row * feat_in + laneid + ext * EXT_WARP_DIM];
        // out_cache[(wid + WARPS_PER_BLOCK) * feat_in + laneid + ext * EXT_WARP_DIM] = 0;
    }
    __syncthreads();

    for (int sp_ext = 0; sp_ext < (dim_sparse + EXT_WARP_DIM - 1) / EXT_WARP_DIM; sp_ext++)
    {

        int sp_laneid = (threadIdx.x + sp_ext * blockDim.x) % dim_sparse;
        int sp_wid = (threadIdx.x + sp_ext * blockDim.x) / dim_sparse;
        if (sp_wid >= WARPS_PER_BLOCK || sp_wid + blockIdx.x * WARPS_PER_BLOCK >= num_warps)
            return;
        int4 sp_w_info;
        int sp_warp_row, sp_warp_loc, sp_warp_len;
        sp_w_info = warp4[sp_wid + blockIdx.x * WARPS_PER_BLOCK];
        sp_warp_row = sp_w_info.x;
        sp_warp_loc = sp_w_info.y;
        sp_warp_len = sp_w_info.z;

        for (int i = 0; i < sp_warp_len; i++)
        {

            int col_idx = __ldg(idx + sp_warp_loc + i);

            int selector_ = __ldg(vin_selector + col_idx * dim_sparse + sp_laneid);

            atomicAdd(&vout[col_idx * dim_sparse + sp_laneid], out_cache[(sp_wid)*feat_in + selector_]);
        }
    }
}

void SPMM_OPT2_SPARSE_BACKWARD_V3::run(int dim)
{
    int shared_size = WARPS_PER_BLOCK * dim * sizeof(float);
    spmm_kernel_opt2_sparse_backward_v3<<<grid, block, shared_size>>>(_warp4, idx, val, vin, vin_sparse_selector, vout, num_v, num_e, dim, dim_sparse, num_warps);
}

double SPMM_OPT2_SPARSE_BACKWARD_V3::do_test(bool timing, int dim)
{
    this->num_warps = cuda_read_array(&this->_warp4, "/home/xix22010/py_projects/graph_preprocess/w" + to_string(WARPS_PER_BLOCK) + "_nz" + "32_warp_4/" + this->_graph + ".warp4") / 4;
    int block_num = (num_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (!timing)
    {
        cout << "block num = " << block_num << endl;
    }

    grid.x = block_num;
    block.x = WARPS_PER_BLOCK * EXT_WARP_DIM;

    double ret = timing_body(timing, dim);

    cudaFree(this->_warp4);
    return ret;
}