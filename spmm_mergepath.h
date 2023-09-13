#pragma once
#include "spmm_base.h"
#include <vector>

class SPMM_MERGEPATH : public SPMM_BASE
{
public:
    using SPMM_BASE::SPMM_BASE;

protected:
    struct CoordinateT {
        int x;
        int y;
    };
    
    CoordinateT MergePathSearch( int diagonal, volatile int* RP, int* NZ_INDICES, int num_rows, int nnz);
    std::vector<int *> generate_mp_sched(int num_threads, int* row_ptr);

    /* Total number of nodes */
    int NODE_NUM = 0;
    /* Total number of nodes in CSR */
    int NODE_ACT_NUM = 0;
    /* Total number of non-zeros */
    int FEATURE_TOTAL = 0;

    int *feature_start_all = 0;
    int *feature_end_all   = 0;
    int *feature_start_num = 0;
    int *feature_end_num   = 0;
    int *start_row_all     = 0;
    int *end_row_all       = 0;
    int *NZ_INDICES        = 0;

    int *d_feature_start, *d_feature_start_num, *d_feature_end, *d_feature_end_num;
    int *d_row_start, *d_row_end;
    int num_threads;
    int *d_degrees;

public:
    double do_test(bool timing, int dim);
    ~SPMM_MERGEPATH(){
        if(feature_start_all != 0){
            delete[] feature_start_all;
        }
        if(feature_end_all != 0){
            delete[] feature_end_all;
        }
        if(feature_start_num != 0){
            delete[] feature_start_num;
        }
        if(feature_end_num != 0){
            delete[] feature_end_num;
        }
        if(start_row_all != 0){
            delete[] start_row_all;
        }
        if(end_row_all != 0){
            delete[] end_row_all;
        }
        if(NZ_INDICES != 0){
            delete[] NZ_INDICES;
        }
    }

protected:
    void run(int dim);

};