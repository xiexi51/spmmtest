#pragma once
#include "spmm_base.h"

class SPMM_COL : public SPMM_BASE
{
public:
    using SPMM_BASE::SPMM_BASE;    

public:
    double do_test(bool timing, int dim);
protected:
    void run(int dim);

};
