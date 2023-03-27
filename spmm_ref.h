#pragma once

__global__ void spmm_kernel(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE);
void spmm_ref(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int num_e, int dim);

