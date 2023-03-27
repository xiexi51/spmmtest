#include "spmm_opt.h"
#include "data.h"
#include <string>
#include <iostream>
#define CONSTINT const int

using namespace std;

extern string base_dir, graph;

const int DEG_BOUND = 12*32;
const int WARPS_PER_BLOCK = 12;

__global__ void spmm_kernel_opt(int *_block4, int *coo_row, int *idx, float *val, float *vin, float *vout, int num_v, int num_e, int feat_in, float *vout_ref)
{
    int4 *block4 = reinterpret_cast<int4 *>(_block4);
    const int4 b_info = block4[blockIdx.x];

    CONSTINT block_degree = b_info.x;
    CONSTINT block_row_begin = b_info.y;
    CONSTINT block_loc_begin = b_info.z;
    CONSTINT block_info = b_info.w;

    CONSTINT wid = threadIdx.x / 32;
    CONSTINT lane_id = threadIdx.x & 31;

    extern __shared__ float out_cache[];

    CONSTINT n_rows = block_degree <= DEG_BOUND ? block_info & 65535 : 1;
    CONSTINT w_nz = block_degree <= DEG_BOUND ? block_info >> 16 : DEG_BOUND / WARPS_PER_BLOCK;
    CONSTINT row_nz = block_degree <= DEG_BOUND ? block_degree : block_info;

    CONSTINT warps_per_row = (row_nz + w_nz - 1) / w_nz;
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
            for (int j = 0; j < feat_in / 32; j++){
                out_cache[wid * feat_in + j * 32 + lane_id] = 0;
                if (warps_per_row > 1 && wid < n_rows)
                {
                    out_cache[(wid + WARPS_PER_BLOCK) * feat_in + j * 32 + lane_id] = 0;
                }
            }
            __syncwarp();
        }
        const int nz_loc = block_loc_begin + warp_loc_row * row_nz + i + warp_loc_col;
        const float left_val = __ldg(val + nz_loc);
        
        for (int j = 0; j < feat_in / 32; j++){
            const float right_val =  vin[__ldg(idx + nz_loc) * feat_in + j * 32 + lane_id];
            out_cache[wid * feat_in + j * 32 + lane_id] += left_val * right_val;
            // out_cache[wid * feat_in + j * 32 + lane_id] += right_val;
        }
    }

    // atomicAdd(&vout[(block_row_begin + warp_loc_row) * feat_in + lane_id], out_cache[wid * 32 + lane_id]);

    
    if (warps_per_row > 1)
    {
        for (int j = 0; j < feat_in / 32; j++){
            atomicAdd_block(&out_cache[(warp_loc_row + WARPS_PER_BLOCK) * feat_in + j * 32 + lane_id], out_cache[wid * feat_in + j * 32 + lane_id]);
        }
        __syncthreads();
        if (wid < n_rows)
        {
            // if(vout[(block_row_begin + wid) * feat_in + lane_id] - vout_ref[(block_row_begin + wid) * feat_in + lane_id] > 0.01){
            //     ;
            // }
            
            for (int j = 0; j < feat_in / 32; j++){
            if (block_degree <= DEG_BOUND)
            {
                    vout[(block_row_begin + wid) * feat_in + j * 32 + lane_id] = out_cache[(wid + WARPS_PER_BLOCK) * feat_in + j * 32 + lane_id];
                
            }
            else
            {
                    atomicAdd(&vout[(block_row_begin + wid) * feat_in + j * 32 + lane_id], out_cache[(wid + WARPS_PER_BLOCK) * feat_in + j * 32 + lane_id]);
                
            }
            }
        }
    }
    else
    {
    #pragma unroll
    for (int j = 0; j < feat_in / 32; j++){
    
        if (block_degree <= DEG_BOUND)
        {
        
                vout[(block_row_begin + wid) * feat_in + j * 32 + lane_id] = out_cache[wid * feat_in + j * 32 + lane_id];
        }
            
        else
        {
            
                atomicAdd(&vout[(block_row_begin + wid) * feat_in + j * 32 + lane_id], out_cache[wid * feat_in + j * 32 + lane_id]);
            
        }
    }
    }
}

void SPMM_OPT::run()
{
    spmm_kernel_opt<<<grid, block, (WARPS_PER_BLOCK + WARPS_PER_BLOCK / 2) * dim * sizeof(float)>>>(_block4, 0, idx, val, vin, vout, num_v, num_e, dim, 0);
}

double SPMM_OPT::do_test(bool timing)
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

    int block_num = cuda_read_array(&this->_block4, "/home/xiexi/PycharmProjects/pythonProject/block_4/" + graph + ".block4") / 4;
    if (!timing)
    {
        cout << "block num = " << block_num << endl;
    }

    grid.x = block_num;
    block.x = WARPS_PER_BLOCK * 32;

    double ret = timing_body(timing);

    // cudaFree(coo_row);
    cudaFree(this->_block4);
    return ret;
}