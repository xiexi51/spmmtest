#include "spmm_opt_sort_innerloop.h"
#include "data.h"
#include <string>
#include <iostream>
#define CONSTINT const int

using namespace std;

extern string base_dir, graph;

const int DEG_BOUND = 12 * 32;
const int WARPS_PER_BLOCK = 12;

#define DIM_MUL(x) ((x + 31) / 32) * 32

__global__ void spmm_kernel_opt_sort_innerloop(const int *_block4, const int *coo_row, const int *idx, const float *val, const float *vin, float *vout, const int num_v, const int num_e, const int feat_in, const float *vout_ref)
{
    const int4 *block4 = reinterpret_cast<const int4 *>(_block4);
    const int4 b_info = block4[blockIdx.x];

    CONSTINT block_degree = b_info.x;
    CONSTINT block_row_begin = b_info.y;
    CONSTINT block_loc_begin = b_info.z;
    CONSTINT block_info = b_info.w;

    CONSTINT n_rows = block_degree <= DEG_BOUND ? block_info & 65535 : 1;
    CONSTINT w_nz = block_degree <= DEG_BOUND ? block_info >> 16 : DEG_BOUND / WARPS_PER_BLOCK;
    CONSTINT row_nz = block_degree <= DEG_BOUND ? block_degree : block_info;

    extern __shared__ float out_cache[];

    CONSTINT round_dim = DIM_MUL(feat_in);

    CONSTINT warps_per_row = (row_nz + w_nz - 1) / w_nz;

    CONSTINT lane_id = threadIdx.x % 32;

    CONSTINT wid = threadIdx.x / 32;

    CONSTINT warp_loc_row = wid / warps_per_row;
    CONSTINT warp_loc_col = wid % warps_per_row * w_nz;

    if (warp_loc_row >= n_rows)
    {
        return;
    }

#pragma unroll
    for (int i = 0; i < w_nz; i++)
    {
        if (i + warp_loc_col >= row_nz)
        {
            break;
        }
        if (i == 0)
        {
            for (int d = lane_id; d < feat_in; d += 32)
            {
                out_cache[wid * round_dim + d] = 0;
#ifdef BLOCK_SUMUP
                if (warps_per_row > 1 && wid < n_rows)
                {
                    out_cache[(wid + WARPS_PER_BLOCK) * round_dim + d] = 0;
                }
#endif
            }
#ifdef BLOCK_SUMUP
            __syncwarp();
#endif
        }
        const int nz_loc = block_loc_begin + warp_loc_row * row_nz + i + warp_loc_col;
        const float left_val = __ldg(val + nz_loc);

        for (int d = lane_id; d < feat_in; d += 32)
        {
            float right_val = vin[__ldg(idx + nz_loc) * feat_in + d];
            out_cache[wid * round_dim + d] += left_val * right_val;
        }
        // out_cache[wid * feat_in + lane_id + j * 32] += right_val;
    }

    // atomicAdd(&vout[(block_row_begin + warp_loc_row) * feat_in + lane_id], out_cache[wid * round_dim + lane_id]);

    if (warps_per_row > 1)
    {

#ifdef BLOCK_SUMUP
        for (int d = lane_id; d < feat_in; d += 32)
        {
            atomicAdd_block(&out_cache[(warp_loc_row + WARPS_PER_BLOCK) * round_dim + d], out_cache[wid * round_dim + d]);
        }

        __syncthreads();
        if (wid < n_rows)
        {

            // if(vout[(block_row_begin + wid) * feat_in + lane_id] - vout_ref[(block_row_begin + wid) * feat_in + lane_id] > 0.01){
            //     ;
            // }

            if (block_degree <= DEG_BOUND)
            {
                for (int d = lane_id; d < feat_in; d += 32)
                {
                    vout[(block_row_begin + wid) * feat_in + d] = out_cache[(wid + WARPS_PER_BLOCK) * round_dim + d];
                }
            }
            else
            {
                for (int d = lane_id; d < feat_in; d += 32)
                {
                    atomicAdd(&vout[(block_row_begin + wid) * feat_in + d], out_cache[(wid + WARPS_PER_BLOCK) * round_dim + d]);
                }
            }
        }
#else
        for (int d = lane_id; d < feat_in; d += 32)
        {
            atomicAdd(&vout[(block_row_begin + warp_loc_row) * feat_in + d], out_cache[wid * round_dim + d]);
        }
#endif
    }
    else
    {
        if (block_degree <= DEG_BOUND)
        {
            for (int d = lane_id; d < feat_in; d += 32)
            {
                vout[(block_row_begin + wid) * feat_in + d] = out_cache[wid * round_dim + d];
            }
        }

        else
        {
            for (int d = lane_id; d < feat_in; d += 32)
            {
                atomicAdd(&vout[(block_row_begin + wid) * feat_in + d], out_cache[wid * round_dim + d]);
            }
        }
    }
}

void SPMM_OPT_SORT_INNERLOOP::run()
{
#ifdef BLOCK_SUMUP
    int shared_size = (WARPS_PER_BLOCK + WARPS_PER_BLOCK / 2) * DIM_MUL(dim) * sizeof(float);
#else
    int shared_size = WARPS_PER_BLOCK * DIM_MUL(dim) * sizeof(float);
#endif
    spmm_kernel_opt_sort_innerloop<<<grid, block, shared_size>>>(_block4, 0, idx, val, vin, vout, num_v, num_e, dim, 0);
}

double SPMM_OPT_SORT_INNERLOOP::do_test(bool timing)
{
    // cudaMallocManaged(&coo_row, num_e * sizeof(int));
    // int k = 0;
    // for (int i = 0; i < num_v; i++)
    // {
    //     for (int j = 0; j < ptr[i + 1] - ptr[i]; j++)
    //     {
    //         coo_row[k++] = i;
    //     }
    // }

    int block_num = cuda_read_array(&this->_block4, "/home/xix22010/py_projects/graph_preprocess/block_4/" + graph + ".block4") / 4;
    if (!timing)
    {
        cout << "block num = " << block_num << endl;
    }

    grid.x = block_num;

    // block.x = DIM_MUL(dim);
    // block.y = WARPS_PER_BLOCK;
    block.x = WARPS_PER_BLOCK * 32;

    double ret = timing_body(timing);

    // cudaFree(coo_row);
    cudaFree(this->_block4);
    return ret;
}