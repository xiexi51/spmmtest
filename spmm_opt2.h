#pragma once
#include "spmm_base.h"

class SPMM_OPT2 : public SPMM_BASE
{
public:
    using SPMM_BASE::SPMM_BASE;

    float *vout_ref;

protected:
    int *_warp4;
    int num_warps;
    

public:
    double do_test(bool timing);
protected:
    void run();

};
