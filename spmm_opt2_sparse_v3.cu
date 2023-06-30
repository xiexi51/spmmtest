#include "spmm_opt2_sparse_v3.h"
#include "data.h"
#include <string>
#include <iostream>
#define CONSTINT const int

using namespace std;

extern string base_dir, graph;

const int WARPS_PER_BLOCK = 12;
const int EXT_WARP_DIM = 32;

// #define DIM_MUL(x) ((x + 31) / 32) * 32

__global__ void spmm_kernel_opt2_sparse_v3(const int *_warp4, const int *idx, const float *val, const float *vin_data, const int *vin_selector, float *vout, const int num_v, const int num_e, const int feat_in, const int dim_sparse, const int num_warps)
{
    const int4 *warp4 = reinterpret_cast<const int4 *>(_warp4);
    extern __shared__ float out_cache[];

    const int total_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_warpid = total_tid / EXT_WARP_DIM;
    const int laneid = threadIdx.x % EXT_WARP_DIM;
    const int wid = threadIdx.x / EXT_WARP_DIM;

    int res_dim_sparse = dim_sparse % 32;

    int res_sparse_wid, res_sparse_laneid;
    
    if(res_dim_sparse > 0){
        res_sparse_wid = wid * (EXT_WARP_DIM / res_dim_sparse) + laneid / res_dim_sparse;
        res_sparse_laneid = laneid % res_dim_sparse;
    }

    int4 sparse_w_info, w_info;
    int sparse_warp_row, sparse_warp_loc, sparse_warp_len;
    int warp_row, warp_loc, warp_len;

    if (total_warpid < num_warps)
    {
        w_info = warp4[total_warpid];
        warp_row = w_info.x;
        warp_loc = w_info.y;
        warp_len = w_info.z;

        if (res_dim_sparse > 0 && res_sparse_wid < blockDim.x / EXT_WARP_DIM)
        {
            sparse_w_info = warp4[blockIdx.x * blockDim.x / EXT_WARP_DIM + res_sparse_wid];
            sparse_warp_row = sparse_w_info.x;
            sparse_warp_loc = sparse_w_info.y;
            sparse_warp_len = sparse_w_info.z;
        }
    }

#pragma unroll
    for (int ext = 0; ext < (feat_in + EXT_WARP_DIM - 1) / EXT_WARP_DIM; ext++)
    {
        out_cache[threadIdx.x + ext * blockDim.x] = 0;
    }
    if (total_warpid >= num_warps)
        return;

    __syncthreads();

    if (res_dim_sparse > 0 && res_sparse_wid < blockDim.x / EXT_WARP_DIM && laneid / res_dim_sparse < EXT_WARP_DIM / res_dim_sparse)
    {
        for (int i = 0; i < sparse_warp_len; i++)
        {

            int nz_loc = sparse_warp_loc + i;
            float left_val = __ldg(val + nz_loc);
            int right_loc = __ldg(idx + nz_loc) * dim_sparse + (dim_sparse / 32) * 32 + res_sparse_laneid;
            float right_val = vin_data[right_loc];
            out_cache[res_sparse_wid * feat_in + vin_selector[right_loc]] += left_val * right_val;

            //  atomicAdd_block(&out_cache[sparse_wid * feat_in + __ldg(vin_selector + right_loc)], left_val * right_val);
        }
    }

    __syncthreads();

#pragma unroll
    for (int base = 0; base < dim_sparse / 32; base++){
        for(int i = 0; i < warp_len; i++){
            int nz_loc = warp_loc + i;
            float left_val = __ldg(val + nz_loc);
            int right_loc = __ldg(idx + nz_loc) * dim_sparse + base * 32 + laneid;
            float right_val = vin_data[right_loc];
            out_cache[wid * feat_in + vin_selector[right_loc]] += left_val * right_val;
        }
    }
    __syncthreads();

#pragma unroll
    for (int ext = 0; ext < (feat_in + EXT_WARP_DIM - 1) / EXT_WARP_DIM; ext++)
    {
        atomicAdd(&vout[warp_row * feat_in + laneid + ext * EXT_WARP_DIM], out_cache[wid * feat_in + laneid + ext * EXT_WARP_DIM]);
    }
}

void SPMM_OPT2_SPARSE_V3::run(int dim)
{
    int shared_size = WARPS_PER_BLOCK * dim * sizeof(float);

    spmm_kernel_opt2_sparse_v3<<<grid, block, shared_size>>>(_warp4, idx, val, vin, vin_sparse_selector, vout, num_v, num_e, dim, dim_sparse, num_warps);
}

double SPMM_OPT2_SPARSE_V3::do_test(bool timing, int dim)
{
    this->num_warps = cuda_read_array(&this->_warp4, "/home/xix22010/py_projects/graph_preprocess/warp_4/" + this->_graph + ".warp4") / 4;
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