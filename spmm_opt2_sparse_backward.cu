#include "spmm_opt2_sparse_backward.h"
#include "data.h"
#include <string>
#include <iostream>
#include <assert.h>
#define CONSTINT const int

using namespace std;

extern string base_dir, graph;

const int WARPS_PER_BLOCK = 8;
const int EXT_WARP_DIM = 32;

__global__ void spmm_kernel_opt2_sparse_backward(const int *_warp4, const int *idx, const float *val, const float *vin_data, const int *vin_selector, float *vout, const int num_v, const int num_e, const int feat_in, const int dim_sparse, const int num_warps)
{
    extern __shared__ float out_cache[];

    const int4 *warp4 = reinterpret_cast<const int4 *>(_warp4);

    const int total_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_warpid = total_tid / EXT_WARP_DIM;
    const int laneid = threadIdx.x % EXT_WARP_DIM;
    const int wid = threadIdx.x / EXT_WARP_DIM;

    const int sparse_wid = wid * (EXT_WARP_DIM / dim_sparse) + laneid / dim_sparse;

    const int sparse_laneid = laneid % dim_sparse;

    int4 sparse_w_info, w_info;
    int sparse_warp_row, sparse_warp_loc, sparse_warp_len;
    int warp_row, warp_loc, warp_len;

    if (total_warpid < num_warps)
    {
        w_info = warp4[total_warpid];
        warp_row = w_info.x;
        warp_loc = w_info.y;
        warp_len = w_info.z;

        if (dim_sparse < 32 && sparse_wid < blockDim.x / EXT_WARP_DIM)
        {
            sparse_w_info = warp4[blockIdx.x * blockDim.x / EXT_WARP_DIM + sparse_wid];
            sparse_warp_row = sparse_w_info.x;
            sparse_warp_loc = sparse_w_info.y;
            sparse_warp_len = sparse_w_info.z;
        }
    }



    // if (total_warpid >= num_warps)
    //     return;

    // if(laneid < sparse_warp_len){
    //     int right_base = idx[warp_loc + laneid] * feat_in;
    //     for(int i = 0; i < dim_sparse; i++){
    //         int sparse_loc = vin_selector[warp_row * dim_sparse + i];
    //         float right_val = vin_data[right_base + sparse_loc];
    //         out_cache[wid * feat_in + sparse_loc] += right_val; 
    //     }
    // }

#define ROW_ACCUM
#define WRITE_DIRECT
//#define COL_WISE


#if 1

#if defined(ROW_ACCUM) && !defined(WRITE_DIRECT)

#pragma unroll
    for (int ext = 0; ext < (feat_in + EXT_WARP_DIM - 1) / EXT_WARP_DIM; ext++)
    {
        // out_cache[wid * feat_in + laneid + ext * EXT_WARP_DIM] = vin_data[warp_row * feat_in + laneid + ext * EXT_WARP_DIM];
        out_cache[wid * feat_in + laneid + ext * EXT_WARP_DIM] = 0;
    }
#elif defined(COL_WISE)
#pragma unroll
    for (int ext = 0; ext < (feat_in + EXT_WARP_DIM - 1) / EXT_WARP_DIM; ext++)
    {
        out_cache[wid * feat_in + laneid + ext * EXT_WARP_DIM] = vin_data[warp_row * feat_in + laneid + ext * EXT_WARP_DIM];
        out_cache[(wid + WARPS_PER_BLOCK) * feat_in + laneid + ext * EXT_WARP_DIM] = 0;
    }


#endif

    if (total_warpid >= num_warps)
        return;

    __syncthreads();


    if (dim_sparse < 32)
    {
        if (sparse_wid < blockDim.x / EXT_WARP_DIM && laneid / dim_sparse < EXT_WARP_DIM / dim_sparse)
        {
            float tmp = 0;

#ifdef ROW_ACCUM
            int selector_loc = vin_selector[sparse_warp_row * dim_sparse + sparse_laneid];
            
            for (int i = 0; i < sparse_warp_len; i++)
            {
                int nz_loc = sparse_warp_loc + i;
            //    float left_val = __ldg(val + nz_loc);   
                // int right_loc = __ldg(idx + nz_loc) * feat_in + selector_loc;
                float right_val = vin_data[__ldg(idx + nz_loc) * feat_in + selector_loc];
                tmp += right_val;
            }

#ifdef WRITE_DIRECT
            atomicAdd(&vout[sparse_warp_row * feat_in + selector_loc], tmp);
#else
            out_cache[sparse_wid * feat_in + selector_loc] += tmp;
#endif
#endif
        
#ifdef COL_WISE
            for (int i = 0; i < sparse_warp_len; i++)
            {
                
                int col_idx = __ldg(idx + sparse_warp_loc + i);

            //    assert(col_idx * dim_sparse + sparse_laneid < num_v * dim_sparse);
                int selector_ = __ldg(vin_selector + col_idx * dim_sparse + sparse_laneid);

                out_cache[(sparse_wid + WARPS_PER_BLOCK) * feat_in + selector_] = out_cache[(sparse_wid) * feat_in + selector_];
                // if(col_idx * feat_in + selector_ >= num_v * feat_in){
                //     printf("wrong");
                // }
                // if(sparse_wid * feat_in + selector_ >= WARPS_PER_BLOCK * feat_in){
                //     printf("wrong 2");
                // }
            //   atomicAdd(&vout[col_idx * feat_in + selector_], out_cache[sparse_wid * feat_in + selector_]);
            //    vout[col_idx * feat_in + selector_ ] += out_cache[sparse_wid * feat_in + selector_];

                #pragma unroll
                for (int ext = 0; ext < (feat_in + EXT_WARP_DIM - 1) / EXT_WARP_DIM; ext++)
                {
                    atomicAdd(&vout[col_idx * feat_in + laneid + ext * EXT_WARP_DIM], out_cache[(wid + WARPS_PER_BLOCK) * feat_in + laneid + ext * EXT_WARP_DIM]);
                }

            //    __syncthreads();
            
            }
        

#endif


        }

        __syncthreads();
    }
    else
    {
        for (int i = 0; i < warp_len; i++)
        {
            for (int l = laneid; l < dim_sparse; l += 32)
            {
                int selector_loc = warp_row * dim_sparse + l;
                int nz_loc = warp_loc + i;
                float left_val = __ldg(val + nz_loc);
                int right_loc = __ldg(idx + nz_loc) * feat_in + vin_selector[selector_loc];
                float right_val = vin_data[right_loc];
                out_cache[wid * feat_in + vin_selector[selector_loc]] += left_val * right_val;
            }
        }

        __syncthreads();
        
    }

#if defined(ROW_ACCUM) && !defined(WRITE_DIRECT)
#pragma unroll
    for (int ext = 0; ext < (feat_in + EXT_WARP_DIM - 1) / EXT_WARP_DIM; ext++)
    {
        atomicAdd(&vout[warp_row * feat_in + laneid + ext * EXT_WARP_DIM], out_cache[wid * feat_in + laneid + ext * EXT_WARP_DIM]);
    }
#endif

#endif
        
}

void SPMM_OPT2_SPARSE_BACKWARD::run(int dim)
{
    int shared_size = WARPS_PER_BLOCK * dim * sizeof(float) * 2;
    spmm_kernel_opt2_sparse_backward<<<grid, block, shared_size>>>(_warp4, idx, val, vin, vin_sparse_selector, vout, num_v, num_e, dim, dim_sparse, num_warps);
}

double SPMM_OPT2_SPARSE_BACKWARD::do_test(bool timing, int dim)
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