#include <iostream>
#include "data.h"
#include "spmm_ref.h"
#include "spmm_opt_sort.h"
#include "spmm_opt_sort_innerloop.h"
#include "spmm_opt.h"
#include "spmm_opt_innerloop.h"
#include "spmm_gnna.h"
#include "spmm_cusparse.h"
#include "spmm_col.h"
#include <random>
#include <algorithm>
#include <filesystem>

string base_dir = "/home/xix22010/cuda_projects/hpc_data/";
// string base_dir = "/home/xiexi/PycharmProjects/pythonProject/graphs/";
// string _graph = "cora";

using namespace std;

double check_err(float *out, float *out_ref, int len, bool &has_err)
{
    double err_sum = 0;
    bool show = 1;

    has_err = 0;

    for (int i = 0; i < len; i++)
    {
        double err = abs(out[i] - out_ref[i]);
        err_sum += err;
        // if (err_sum / (v_num * dim) >= 0.001 && show)
        // {
        //     show = 0;
        //     cout << "fail begin at " << i/32 << endl;
        // }
        if (err > 0.1 && has_err == 0)
        {
            has_err = 1;
            cout << "err at " << i << endl;
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

void test_graph(string graph)
{
    int dim_min = 16, dim_max = 128;
    int *cu_indptr, *cu_indices, *cu_indptr_new, *cu_indices_new, *cu_coo_row;
    int v_num = cuda_read_array(&cu_indptr_new, base_dir + graph + ".new_indptr") - 1;
    int e_num = cuda_read_array(&cu_indices_new, base_dir + graph + ".new_indices");
    cuda_read_array(&cu_indptr, base_dir + graph + ".graph.ptrdump");
    cuda_read_array(&cu_indices, base_dir + graph + ".graph.edgedump");

    cudaMallocManaged(&cu_coo_row, e_num * sizeof(int));
    {
        int k = 0;
        for (int i = 0; i < v_num; i++)
        {
            for (int j = 0; j < cu_indptr[i + 1] - cu_indptr[i]; j++)
            {
                cu_coo_row[k++] = i;
            }
        }
    }

    // cout << "graph = " << graph << " v_num = " << v_num << " e_num = " << e_num << endl;
    float *cu_val;
    cudaMallocManaged(&cu_val, e_num * sizeof(float));

    float *cu_vin, *cu_vout, *cu_vout2, *cu_vout_new, *cu_vout2_new, *cu_vout_ref, *cu_vout_gnna, *cu_vout_ref_new, *cu_vout_ref_coo, *cu_vout_col;
    cudaMallocManaged(&cu_vin, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout2, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout_new, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout2_new, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout_gnna, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout_ref, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout_ref_new, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout_ref_coo, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout_col, v_num * dim_max * sizeof(float));

    default_random_engine engine;
    engine.seed(123);

    uniform_real_distribution<float> rd(0, 1);

    int input_mode = 3;
    switch (input_mode)
    {
    case 1:
        generate(cu_val, cu_val + e_num, [&]()
                 { return rd(engine); });
        generate(cu_vin, cu_vin + v_num * dim_max, [&]()
                 { return rd(engine); });
        break;
    case 2:
        for (int i = 0; i < e_num; i++)
        {
            cu_val[i] = 1;
        }
        for (int i = 0; i < v_num * dim_max; i++)
        {
            cu_vin[i] = 0.01 * i;
        }
        break;
    case 3:
        for (int i = 0; i < e_num; i++)
        {
            cu_val[i] = 1;
        }
        generate(cu_vin, cu_vin + v_num * dim_max, [&]()
                 { return rd(engine); });
        break;

    default:
        break;
    }

    // fill(cu_vin, cu_vin + v_num * dim, 1);
    fill(cu_vout, cu_vout + v_num * dim_max, 0);
    fill(cu_vout2, cu_vout2 + v_num * dim_max, 0);
    fill(cu_vout_gnna, cu_vout_gnna + v_num * dim_max, 0);
    fill(cu_vout_ref, cu_vout_ref + v_num * dim_max, 0);
    fill(cu_vout_ref_new, cu_vout_ref_new + v_num * dim_max, 0);

    SPMM_OPT_SORT opt_sort(graph, cu_indptr_new, cu_indices_new, cu_val, cu_vin, cu_vout_new, v_num, e_num, dim_max);
    SPMM_OPT_SORT_INNERLOOP opt_sort_innerloop(graph, cu_indptr_new, cu_indices_new, cu_val, cu_vin, cu_vout2_new, v_num, e_num, dim_max);
    SPMM_OPT opt(graph, cu_indptr, cu_indices, cu_val, cu_vin, cu_vout, v_num, e_num, dim_max);
    SPMM_OPT_INNERLOOP opt_innerloop(graph, cu_indptr, cu_indices, cu_val, cu_vin, cu_vout2, v_num, e_num, dim_max);
    SPMM_GNNA gnna(graph, cu_indptr, cu_indices, cu_val, cu_vin, cu_vout_gnna, v_num, e_num, dim_max);
    SPMM_COL spmm_col(graph, cu_indptr, cu_indices, cu_val, cu_vin, cu_vout_col, v_num, e_num, dim_max);

// #define CHECK
#define TIMING

    for (int dim = dim_min; dim <= dim_max; dim++)
    {
        // cout << "dim = " << dim << endl;

#ifdef CHECK
        spmm_cusparse(cu_indptr, cu_indices, cu_val, cu_vin, cu_vout_ref, v_num, e_num, dim, 0);
        spmm_cusparse(cu_indptr_new, cu_indices_new, cu_val, cu_vin, cu_vout_ref_new, v_num, e_num, dim, 0);
        spmm_cusparse_coo(cu_coo_row, cu_indices, cu_val, cu_vin, cu_vout_ref_coo, v_num, e_num, dim, 0);
        opt_sort.do_test(false, dim);
        opt_sort_innerloop.do_test(false, dim);
        opt.do_test(false, dim);
        opt_innerloop.do_test(false, dim);
        gnna.do_test(false, dim);
        spmm_col.do_test(false, dim);

        bool has_err = 0;
        cout << "checking cusparse_coo" << endl;
        check_err(cu_vout_ref_coo, cu_vout_ref, v_num * dim, has_err);
        cout << "checking opt_sort" << endl;
        check_err(cu_vout_new, cu_vout_ref_new, v_num * dim, has_err);
        cout << "checking opt_sort_innerloop" << endl;
        check_err(cu_vout2_new, cu_vout_ref_new, v_num * dim, has_err);
        cout << "checking opt" << endl;
        check_err(cu_vout, cu_vout_ref, v_num * dim, has_err);
        cout << "checking opt_innerloop" << endl;
        check_err(cu_vout2, cu_vout_ref, v_num * dim, has_err);
        cout << "checking gnna" << endl;
        check_err(cu_vout_gnna, cu_vout_ref, v_num * dim, has_err);
        cout << "checking spmm_col" << endl;
        check_err(cu_vout_col, cu_vout_ref, v_num * dim, has_err);

#endif

#ifdef TIMING
        double t_cusparse = spmm_cusparse(cu_indptr, cu_indices, cu_val, cu_vin, cu_vout_ref, v_num, e_num, dim, 10);
        cout << graph << " " << dim << " cusparse " << t_cusparse * 1000 << endl;

        double t_cusparse_coo = spmm_cusparse_coo(cu_coo_row, cu_indices, cu_val, cu_vin, cu_vout_ref_coo, v_num, e_num, dim, 10);
        cout << graph << " " << dim << " cusparse_coo " << t_cusparse_coo * 1000 << endl;

        double t_opt_sort = opt_sort.do_test(true, dim);
        cout << graph << " " << dim << " opt_sort " << t_opt_sort * 1000 << endl;

        double t_opt_sort_innerloop = opt_sort_innerloop.do_test(true, dim);
        cout << graph << " " << dim << " opt_sort_innerloop " << t_opt_sort_innerloop * 1000 << endl;

        double t_opt = opt.do_test(true, dim);
        cout << graph << " " << dim << " opt " << t_opt * 1000 << endl;

        double t_opt_innerloop = opt_innerloop.do_test(true, dim);
        cout << graph << " " << dim << " opt_innerloop " << t_opt_innerloop * 1000 << endl;

        double t_gnna = gnna.do_test(true, dim);
        cout << graph << " " << dim << " gnna " << t_gnna * 1000 << endl;

        double t_spmm_col = spmm_col.do_test(true, dim);
        cout << graph << " " << dim << " spmm_col " << t_spmm_col * 1000 << endl;

#endif
    }

    cudaFree(cu_indptr);
    cudaFree(cu_indices);
    cudaFree(cu_coo_row);
    cudaFree(cu_indptr_new);
    cudaFree(cu_indices_new);
    cudaFree(cu_val);
    cudaFree(cu_vin);
    cudaFree(cu_vout);
    cudaFree(cu_vout2);
    cudaFree(cu_vout_new);
    cudaFree(cu_vout2_new);
    cudaFree(cu_vout_gnna);
    cudaFree(cu_vout_ref);
    cudaFree(cu_vout_ref_new);
    cudaFree(cu_vout_ref_coo);
    cudaFree(cu_vout_col);
}

int main(int argc, char *argv[])
{
    if (argc > 1)
    {
        string arg_graph(argv[1]);
        cout << "dir = " << base_dir << endl;
        test_graph(arg_graph);
    }
    else
    {
        string folder_path = "/home/xix22010/cuda_projects/hpc_data/";
        string extension = ".config";
        filesystem::directory_iterator iter(folder_path);
        int n = 0;
        for (const auto &file : iter)
        {
            if (file.path().extension() == extension)
            {
                test_graph(file.path().stem().string());
                cudaDeviceSynchronize();
                n++;
                // if (n >= 2)
                // {
                //     break;
                // }
            }
        }
    }

    return 0;
}