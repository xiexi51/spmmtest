#include <iostream>
#include "data.h"
#include "spmm_ref.h"
#include "spmm_opt.h"
#include "spmm_opt2.h"
#include "spmm_gnna.h"
#include "spmm_cusparse.h"
#include <random>
#include <algorithm>

string base_dir = "/home/xi/cuda_projects/hpc_data/";
// string base_dir = "/home/xiexi/PycharmProjects/pythonProject/graphs/";
string graph = "artist";

using namespace std;

double check_err(float *out, float *out_ref, int len, bool &has_err)
{
    double err_sum = 0;
    bool show = 1;

    for (int i = 0; i < len; i++)
    {
        double err = abs(out[i] - out_ref[i]);
        err_sum += err;
        // if (err_sum / (v_num * dim) >= 0.001 && show)
        // {
        //     show = 0;
        //     cout << "fail begin at " << i/32 << endl;
        // }
        if (err > 0.01)
        {
            has_err = 1;
        }
    }
    cout << "err sum = " << err_sum << "  ";
    if (err_sum / len < 0.001)
    // if(!has_err)
    {
        cout << "validation pass!" << endl;
    }
    else
    {
        cout << "validation fail!" << endl;
    }
    return err_sum;
}

int main(int argc, char *argv[])
{
    int dim = 32;
    if (argc > 2)
    {
        string arg_graph(argv[1]);
        graph = arg_graph;
        dim = atoi(argv[2]);
    }
    else if (argc > 1)
    {
        string arg_graph(argv[1]);
        graph = arg_graph;
    }
    cout << "dir = " << base_dir << endl;
    cout << "dim = " << dim << endl;
    int *cu_indptr, *cu_indices, *cu_indptr_new, *cu_indices_new;

    int v_num = cuda_read_array(&cu_indptr_new, base_dir + graph + ".new_indptr") - 1;
    int e_num = cuda_read_array(&cu_indices_new, base_dir + graph + ".new_indices");
    cuda_read_array(&cu_indptr, base_dir + graph + ".graph.ptrdump");
    cuda_read_array(&cu_indices, base_dir + graph + ".graph.edgedump");
    cout << "graph = " << graph << " v_num = " << v_num << " e_num = " << e_num << endl;
    float *cu_val;
    cudaMallocManaged(&cu_val, e_num * sizeof(float));

    float *cu_vin, *cu_vout, *cu_vout_ref, *cu_vout2, *cu_vout_gnna, *cu_vout_ref_new;
    cudaMallocManaged(&cu_vin, v_num * dim * sizeof(float));
    cudaMallocManaged(&cu_vout, v_num * dim * sizeof(float));
    cudaMallocManaged(&cu_vout2, v_num * dim * sizeof(float));
    cudaMallocManaged(&cu_vout_gnna, v_num * dim * sizeof(float));
    cudaMallocManaged(&cu_vout_ref, v_num * dim * sizeof(float));
    cudaMallocManaged(&cu_vout_ref_new, v_num * dim * sizeof(float));

    default_random_engine engine;
    engine.seed(123);

    uniform_real_distribution<float> rd(0, 1);

    if (0)
    {
        generate(cu_val, cu_val + e_num, [&]()
                 { return rd(engine); });
        generate(cu_vin, cu_vin + v_num * dim, [&]()
                 { return rd(engine); });
    }
    else if (0)
    {
        for (int i = 0; i < e_num; i++)
        {
            cu_val[i] = 1;
        }

        for (int i = 0; i < v_num * dim; i++)
        {
            cu_vin[i] = 0.01 * i;
        }
    }
    else if (1)
    {
        for (int i = 0; i < e_num; i++)
        {
            cu_val[i] = 1;
        }

        generate(cu_vin, cu_vin + v_num * dim, [&]()
                 { return rd(engine); });
    }

    // fill(cu_vin, cu_vin + v_num * dim, 1);
    fill(cu_vout, cu_vout + v_num * dim, 0);
    fill(cu_vout2, cu_vout2 + v_num * dim, 0);
    fill(cu_vout_gnna, cu_vout_gnna + v_num * dim, 0);
    fill(cu_vout_ref, cu_vout_ref + v_num * dim, 0);
    fill(cu_vout_ref_new, cu_vout_ref_new + v_num * dim, 0);

    SPMM_OPT opt(cu_indptr_new, cu_indices_new, cu_val, cu_vin, cu_vout, v_num, e_num, dim);
    SPMM_OPT2 opt2(cu_indptr, cu_indices, cu_val, cu_vin, cu_vout2, v_num, e_num, dim);
    SPMM_GNNA gnna(cu_indptr, cu_indices, cu_val, cu_vin, cu_vout_gnna, v_num, e_num, dim);
    opt.do_test(false);
    // for (int i = 0; i < v_num; i++)
    // {
    //     cout << i << endl;
    //     for (int j = 0; j < dim; j++)
    //     {
    //         cout << cu_vout[i * dim + j] << " ";
    //     }
    //     cout << endl;
    // }
    opt2.do_test(false);
    gnna.do_test(false);

    spmm_cusparse(cu_indptr_new, cu_indices_new, cu_val, cu_vin, cu_vout_ref_new, v_num, e_num, dim, 0);
    spmm_cusparse(cu_indptr, cu_indices, cu_val, cu_vin, cu_vout_ref, v_num, e_num, dim, 0);

    bool has_err = 0;
    cout << "checking opt" << endl;
    check_err(cu_vout, cu_vout_ref_new, v_num * dim, has_err);
    cout << "checking opt2" << endl;
    check_err(cu_vout2, cu_vout_ref, v_num * dim, has_err);
    cout << "checking gnna" << endl;
    check_err(cu_vout_gnna, cu_vout_ref, v_num * dim, has_err);
    

    if (true)
    {
        double t_cusparse = spmm_cusparse(cu_indptr, cu_indices, cu_val, cu_vin, cu_vout_ref, v_num, e_num, dim, 10);
        cout << "cusparse time = " << t_cusparse * 1000 << endl;

        double t_opt = opt.do_test(true);
        cout << "opt time = " << t_opt * 1000 << endl;

        double t_opt2 = opt2.do_test(true);
        cout << "opt2 time = " << t_opt2 * 1000 << endl;

        double t_gnna = gnna.do_test(true);
        cout << "gnna time = " << t_gnna * 1000 << endl;
    }

    cudaFree(cu_indptr);
    cudaFree(cu_indices);
    cudaFree(cu_indptr_new);
    cudaFree(cu_indices_new);
    cudaFree(cu_val);
    cudaFree(cu_vin);
    cudaFree(cu_vout);
    cudaFree(cu_vout2);
    cudaFree(cu_vout_gnna);
    cudaFree(cu_vout_ref);
    cudaFree(cu_vout_ref_new);

    return 0;
}