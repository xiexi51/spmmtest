#pragma once
#include "spmm_base.h"

class SPMM_OPT_SORT_INNERLOOP : public SPMM_BASE
{
public:
    using SPMM_BASE::SPMM_BASE;

    float *vout_ref;

protected:
    int *coo_row, *_block4;
    

public:
    double do_test(bool timing);
protected:
    void run();

};

// __global__ void spmm_kernel_opt(int *ptr, int *coo_row, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in);
// double spmm_opt(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int num_e, int dim, int times);
