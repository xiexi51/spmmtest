#include "spmm_mergepath.h"
#include "data.h"
#include <string>
#include <vector>
#include <iostream>
using namespace std;

extern string base_dir, graph;

const int WARP_SIZE = 32;
const int WARPS_PER_BLOCK = 8;
const int BLOCK = WARPS_PER_BLOCK * WARP_SIZE; 


SPMM_MERGEPATH::CoordinateT SPMM_MERGEPATH::MergePathSearch( int diagonal, volatile int* RP, int* NZ_INDICES, int num_rows, int nnz)
{
    int x_min = max(diagonal - nnz, 0);
    int x_max = min(diagonal, num_rows);

    while (x_min < x_max) {
        // so this is div by 2
        int pivot = (x_min + x_max) >> 1;
        if (RP[pivot] <= NZ_INDICES[diagonal - pivot - 1]) {
            x_min = pivot + 1;
        } 
        else {
            x_max = pivot;
        }
    }
    return CoordinateT{min(x_min, num_rows), diagonal - x_min};
}


std::vector<int *> SPMM_MERGEPATH::generate_mp_sched(int num_threads, int* row_ptr) {

    feature_start_all = new int[num_threads];
    feature_end_all   = new int[num_threads];
    feature_start_num = new int[num_threads];
    feature_end_num   = new int[num_threads];
    start_row_all     = new int[num_threads];
    end_row_all       = new int[num_threads];
    NZ_INDICES        = new int[FEATURE_TOTAL];
    
    for (int i = 0; i < FEATURE_TOTAL; i++) {
        NZ_INDICES[i] = i;
    }

    for (int i = 0; i < num_threads; i++) {
        feature_start_all[i] = 0;
        feature_end_all[i]   = 0;
        feature_start_num[i] = 0;
        feature_end_num[i]   = 0;
        start_row_all[i]     = 0;
        end_row_all[i]       = 0;
    }

    

    for (int i = 0; i < num_threads; i++) {
        int core_id = i;

        int num_merge_items = NODE_ACT_NUM + FEATURE_TOTAL; 
        int items_per_thread = (num_merge_items + num_threads - 1) / num_threads;

        int diagonal = min(items_per_thread * core_id, num_merge_items);
        int diagonal_end = min(diagonal + items_per_thread, num_merge_items);

        
                                                                
        CoordinateT thread_coord = MergePathSearch(diagonal, row_ptr, NZ_INDICES, NODE_ACT_NUM, FEATURE_TOTAL);
        CoordinateT thread_coord_end = MergePathSearch(diagonal_end, row_ptr, NZ_INDICES, NODE_ACT_NUM, FEATURE_TOTAL);
    
        int start = thread_coord.x - 1;
        int end = thread_coord_end.x - 1;
        if (start < 0) start = 0;

        int num_features = 0;

        int feature_start = thread_coord.y;
        if (row_ptr[start] == feature_start) {
            feature_start = 0;
        }
        if (core_id == 0) {
            feature_start = 0;
        }

        int feature_end = thread_coord_end.y;
        if (row_ptr[end] == feature_end) {
            feature_end = 0;
        }

        if (feature_start != 0) {
            if (start == end && feature_end != 0) {
                num_features = feature_end - feature_start;
                feature_end = 0;
            }
            else {
                num_features = row_ptr[start + 1] - feature_start;
            }
            
        }
        int num_features_end = 0;
        if (feature_end != 0) num_features_end = feature_end - row_ptr[end];

        feature_start_all[core_id] = feature_start;
        feature_end_all[core_id]   = feature_end; 
        feature_start_num[core_id] = num_features;
        feature_end_num[core_id]   = num_features_end;   
        start_row_all[core_id]     = start;     
        end_row_all[core_id]       = end;       

    }

    

    return {feature_start_all, feature_start_num, 
            feature_end_all,
            feature_end_num,
            start_row_all, end_row_all};
    
}


// #define DEGREE_NORM


__global__ void spmm_merge_path(
    float *output,
    float *input, 
    int *row_pointers, 
    int *column_index, 
    int *degrees, 
    int *feature_start,
    int *feature_end,
    int *feature_start_num,
    int *feature_end_num,    
    int *start_row,
    int *end_row,
    int num_nodes, 
    int dim,
    int dimWorker,
    int warpPerBlock,
    int num_warps,
    int sched_to_process
) {

    int tid =  blockIdx.x * blockDim.x + threadIdx.x;  // global thread-id
    int warpId = tid / WARP_SIZE;                             // global warp-id
    int laneid = threadIdx.x % WARP_SIZE;                     // warp thread-id -- laneid
    
    if (warpId < num_warps) {
        int sched_id = laneid / dim;
        laneid =  laneid % dim;
        if (sched_id >= sched_to_process) { 
            return;
        }
        
        int start = start_row[sched_to_process * warpId + sched_id];
        int end = end_row[sched_to_process * warpId + sched_id];
        int fstart = feature_start[sched_to_process * warpId + sched_id];
        int fstart_num = feature_start_num[sched_to_process * warpId + sched_id];
        int fend = feature_end[sched_to_process * warpId + sched_id];
        int fend_num = feature_end_num[sched_to_process * warpId + sched_id];

        float partial_results_start = 0;
        float  partial_results_end = 0;
        float output_temp = 0; 
        float degree_norm_inv = 0;
        float src_norm = 0;
        int index = 0;
        int num_features = 0;
        int features_start = 0;

        if (fstart != 0) {
            src_norm = degrees[start];  
            
            for (int j = 0; j < fstart_num; j++) {
                index = column_index[fstart++];
#ifdef DEGREE_NORM
                degree_norm_inv =  __fmaf_rn(src_norm, degrees[index], 0);
                partial_results_start +=  __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0); 
#else
                partial_results_start +=  __fmaf_rn(1, input[index * dim + laneid], 0); 
#endif
                            
            }
            atomicAdd(&output[start * dim + laneid], partial_results_start);         
            start = start + 1;

        }

        for (int i = start; i < end; i++) {
            src_norm = degrees[i];
            output_temp = 0.0f;

            num_features = row_pointers[i + 1] - row_pointers[i];
            features_start = row_pointers[i]; 
    
            #pragma unroll
            for (int j = 0; j < num_features; j++) {
                index = column_index[features_start];
#ifdef DEGREE_NORM
                degree_norm_inv =  __fmaf_rn(src_norm, degrees[index], 0);
                output_temp += __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0);
#else
                output_temp += __fmaf_rn(1, input[index * dim + laneid], 0);
#endif
                features_start++;
            }

            output[i * dim + laneid] = output_temp;
        }             

        if (fend != 0) {
            src_norm = 1;  
         
            #pragma unroll
            for (int j = 0; j < fend_num; j++) {
                index = column_index[fend++];
#ifdef DEGREE_NORM
                degree_norm_inv =  __fmaf_rn(src_norm, degrees[index], 0);
                partial_results_end += __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0); 
#else
                partial_results_end += __fmaf_rn(1, input[index * dim + laneid], 0); 
#endif
            } 
            atomicAdd(&output[end * dim + laneid], partial_results_end);
        }
        return;
    }
}


__global__ void spmm_merge_path_64(
    float *output,
    float *input, 
    int *row_pointers, 
    int *column_index, 
    int *degrees, 
    int *feature_start,
    int *feature_end,
    int *feature_start_num,
    int *feature_end_num,    
    int *start_row,
    int *end_row,
    int num_nodes, 
    int dim,
    int dimWorker,
    int warpPerBlock,
    int num_warps,
    int factor
) {

    int tid =  blockIdx.x * blockDim.x + threadIdx.x;  // global thread-id
    int warpId = tid / WARP_SIZE;                             // global warp-id
    int laneid = threadIdx.x % WARP_SIZE;                     // warp thread-id -- laneid
    
    if (warpId < num_warps) {
        
        laneid += (warpId % factor) * 32;
        if (laneid > dim)  return;
        warpId = warpId / factor;
       
        int start = start_row[warpId];
        int end = end_row[warpId];
        int fstart = feature_start[warpId];
        int fstart_num = feature_start_num[warpId];
        int fend = feature_end[warpId];
        int fend_num = feature_end_num[warpId];

        float partial_results_start = 0;
        float  partial_results_end = 0;
        float output_temp = 0; 
        float degree_norm_inv = 0;
        float src_norm = 0;
        int index = 0;
        int num_features = 0;
        int features_start = 0;

        if (fstart != 0) {
            src_norm = degrees[start];  
            
            for (int j = 0; j < fstart_num; j++) {
                index = column_index[fstart++];
#ifdef DEGREE_NORM
                degree_norm_inv =  __fmaf_rn(src_norm, degrees[index], 0);
                partial_results_start +=  __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0); 
#else
                partial_results_start +=  __fmaf_rn(1, input[index * dim + laneid], 0); 
#endif
                            
            }
            atomicAdd(&output[start * dim + laneid], partial_results_start);         
            start = start + 1;

        }

        for (int i = start; i < end; i++) {
            src_norm = degrees[i];
            output_temp = 0.0f;

            num_features = row_pointers[i + 1] - row_pointers[i];
            features_start = row_pointers[i]; 
    
            #pragma unroll
            for (int j = 0; j < num_features; j++) {
                index = column_index[features_start];
#ifdef DEGREE_NORM
                degree_norm_inv =  __fmaf_rn(src_norm, degrees[index], 0);
                output_temp += __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0);
#else
                output_temp += __fmaf_rn(1, input[index * dim + laneid], 0);
#endif
                features_start++;
            }

            output[i * dim + laneid] = output_temp;
        }             

        if (fend != 0) {
            src_norm = 1;  
         
            #pragma unroll
            for (int j = 0; j < fend_num; j++) {
                index = column_index[fend++];
#ifdef DEGREE_NORM
                degree_norm_inv =  __fmaf_rn(src_norm, degrees[index], 0);
                partial_results_end +=  __fmaf_rn(degree_norm_inv, input[index * dim + laneid], 0); 
#else
                partial_results_end +=  __fmaf_rn(1, input[index * dim + laneid], 0); 
#endif
            } 
            atomicAdd(&output[end * dim + laneid], partial_results_end);
        }
        return;
    }
}


void SPMM_MERGEPATH::run(int dim)
{
    if (dim <= WARP_SIZE) {
        int threads_per_warp = WARP_SIZE / dim;
        int grid = (num_threads * WARP_SIZE + BLOCK - 1) / (BLOCK * threads_per_warp); 
        spmm_merge_path<<<grid, BLOCK>>>(
                (float *) this->vout, (float *) this->vin, 
                (int *) this->ptr, (int *) this->idx, (int *) d_degrees, 
                (int *) d_feature_start,
                (int *) d_feature_end,
                (int *) d_feature_start_num,
                (int *) d_feature_end_num,
                (int *) d_row_start,
                (int *) d_row_end,
                NODE_NUM, dim, WARP_SIZE, WARPS_PER_BLOCK, num_threads, threads_per_warp);
    }
    else {
        int factor = ceil(dim / WARP_SIZE);
        int num_threads_gpu = num_threads * factor; 
        int grid = (num_threads_gpu * WARP_SIZE + BLOCK  - 1) / (BLOCK);
        spmm_merge_path_64<<<grid, BLOCK>>>(
            (float *) this->vout, (float *) this->vin, 
            (int *) this->ptr, (int *) this->idx, (int *) d_degrees, 
            (int *) d_feature_start,
            (int *) d_feature_end,
            (int *) d_feature_start_num,
            (int *) d_feature_end_num,
            (int *) d_row_start,
            (int *) d_row_end,
            NODE_NUM, dim, WARP_SIZE, WARPS_PER_BLOCK, num_threads_gpu, factor);
    }
}

double SPMM_MERGEPATH::do_test(bool timing, int dim)
{
    NODE_ACT_NUM = num_v + 1;
    NODE_NUM = NODE_ACT_NUM - 1;
    FEATURE_TOTAL = num_e;

    
    int cost = 20;
    num_threads = (NODE_ACT_NUM + FEATURE_TOTAL - 1) / cost;
    if (num_threads < 1024){
        num_threads = 1024;
    }
    auto mp_sched = generate_mp_sched(num_threads, this->ptr);

    cudaMallocManaged(&d_degrees, NODE_NUM * sizeof(int));
    for (int i = 0; i < NODE_NUM; i++) {
        d_degrees[i] = this->ptr[i + 1] - this->ptr[i];
    }

    
    
    cudaMallocManaged(&d_feature_start, num_threads * sizeof(int));
    cudaMallocManaged(&d_feature_start_num, num_threads * sizeof(int));
    cudaMallocManaged(&d_feature_end, num_threads * sizeof(int));
    cudaMallocManaged(&d_feature_end_num, num_threads * sizeof(int));
    cudaMallocManaged(&d_row_start, num_threads * sizeof(int));
    cudaMallocManaged(&d_row_end, num_threads * sizeof(int));

    

    cudaMemcpy(d_feature_start, mp_sched[0], num_threads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_feature_start_num, mp_sched[1], num_threads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_feature_end, mp_sched[2], num_threads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_feature_end_num, mp_sched[3], num_threads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_start, mp_sched[4], num_threads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_end, mp_sched[5], num_threads * sizeof(int), cudaMemcpyHostToDevice);

    double ret = timing_body(timing, dim);

    cudaFree(d_degrees);
    cudaFree(d_feature_start);
    cudaFree(d_feature_start_num);
    cudaFree(d_feature_end);
    cudaFree(d_feature_end_num);
    cudaFree(d_row_start);
    cudaFree(d_row_end);

    return ret;
}