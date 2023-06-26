#include <iostream>
#include "data.h"
#include "spmm_ref.h"
#include "spmm_opt_sort.h"
#include "spmm_opt_sort2.h"
#include "spmm_opt_sort_innerloop.h"
#include "spmm_opt.h"
#include "spmm_opt2.h"
#include "spmm_opt2_sparse.h"
#include "spmm_opt2_sparse_shared.h"
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

int total_file_cnt, current_file_cnt;

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

void test_graph(string graph, int spec_dim)
{
    int dim_min = 256, dim_max = 256, interval = 1;
    int dim_sparse = 16;
    cout << "dim sparse = " << dim_sparse << endl;

    if (spec_dim > 0)
    {
        dim_min = spec_dim;
        dim_max = spec_dim;
    }

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

    float *cu_vin, *cu_vout, *cu_vout2, *cu_vout_inner, *cu_vout_new, *cu_vout_new2, *cu_vout_inner_new, *cu_vout_ref, *cu_vout_gnna, *cu_vout_ref_new, *cu_vout_ref_coo, *cu_vout_col;
    float *cu_vin_sparse, *cu_vin_sparse_data, *cu_vout2_sparse, *cu_vout2_sparse_shared;
    int *cu_vin_sparse_selector;
    cudaMallocManaged(&cu_vin, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout2, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout_inner, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout_new, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout_new2, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout_inner_new, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout_gnna, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout_ref, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout_ref_new, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout_ref_coo, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout_col, v_num * dim_max * sizeof(float));

    cudaMallocManaged(&cu_vin_sparse, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vin_sparse_data, v_num * dim_sparse * sizeof(float));
    cudaMallocManaged(&cu_vin_sparse_selector, v_num * dim_sparse * sizeof(int));
    cudaMallocManaged(&cu_vout2_sparse, v_num * dim_max * sizeof(float));
    cudaMallocManaged(&cu_vout2_sparse_shared, v_num * dim_max * sizeof(float));


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

    vector<int> sequence(dim_max);
    iota(sequence.begin(), sequence.end(), 0);  // Fill with 0, 1, ..., dim_max - 1

    vector<int> sample(dim_sparse);

    for (int i = 0; i < v_num; ++i) {
        std::sample(sequence.begin(), sequence.end(), sample.begin(), dim_sparse, engine);

        for (int j = 0; j < dim_sparse; ++j) {
            cu_vin_sparse_data[i * dim_sparse + j] = rd(engine);
            cu_vin_sparse_selector[i * dim_sparse + j] = sample[j];
        }
    }

    for (int i = 0; i < v_num; ++i) {
        for (int j = 0; j < dim_max; ++j) {
            cu_vin_sparse[i * dim_max + j] = 0.0;
        }
        for (int j = 0; j < dim_sparse; ++j) {
            int col = cu_vin_sparse_selector[i * dim_sparse + j];
            cu_vin_sparse[i * dim_max + col] = cu_vin_sparse_data[i * dim_sparse + j];
        }
    }

    // for(int i = 0; i < 100; i++){
    //     cout<<cu_vin_sparse_data[i]<<" ";
    // }
    // cout<<endl<<endl;
    // for(int i = 0; i < 100; i++){
    //     cout<<cu_vin_sparse_selector[i]<<" ";
    // }
    // cout<<endl<<endl;
    // for(int i = 0; i < 100; i++){
    //     cout<<cu_vin_sparse[i]<<" ";
    // }
    // cout<<endl<<endl;


    // fill(cu_vin, cu_vin + v_num * dim, 1);
    fill(cu_vout, cu_vout + v_num * dim_max, 0);
    fill(cu_vout_inner, cu_vout_inner + v_num * dim_max, 0);
    fill(cu_vout_gnna, cu_vout_gnna + v_num * dim_max, 0);
    fill(cu_vout_ref, cu_vout_ref + v_num * dim_max, 0);
    fill(cu_vout_ref_new, cu_vout_ref_new + v_num * dim_max, 0);

    fill(cu_vout2, cu_vout2 + v_num * dim_max, 0);
    fill(cu_vout2_sparse, cu_vout2_sparse + v_num * dim_max, 0);

    // SPMM_OPT_SORT opt_sort(graph, cu_indptr_new, cu_indices_new, cu_val, cu_vin, cu_vout_new, v_num, e_num, dim_max);
    // SPMM_OPT_SORT2 opt_sort2(graph, cu_indptr_new, cu_indices_new, cu_val, cu_vin, cu_vout_new2, v_num, e_num, dim_max);
    // SPMM_OPT_SORT_INNERLOOP opt_sort_innerloop(graph, cu_indptr_new, cu_indices_new, cu_val, cu_vin, cu_vout_inner_new, v_num, e_num, dim_max);
    // SPMM_OPT opt(graph, cu_indptr, cu_indices, cu_val, cu_vin, cu_vout, v_num, e_num, dim_max);

    SPMM_OPT2 opt2(graph, cu_indptr, cu_indices, cu_val, cu_vin_sparse, cu_vout2, v_num, e_num, dim_max);

    // SPMM_OPT_INNERLOOP opt_innerloop(graph, cu_indptr, cu_indices, cu_val, cu_vin, cu_vout_inner, v_num, e_num, dim_max);
    // SPMM_GNNA gnna(graph, cu_indptr, cu_indices, cu_val, cu_vin, cu_vout_gnna, v_num, e_num, dim_max);
    // SPMM_COL spmm_col(graph, cu_indptr, cu_indices, cu_val, cu_vin, cu_vout_col, v_num, e_num, dim_max);

    SPMM_OPT2_SPARSE opt2_sparse(graph, cu_indptr, cu_indices, cu_val, cu_vin_sparse_data, cu_vout2_sparse, v_num, e_num, dim_max);
    opt2_sparse.vin_sparse_selector = cu_vin_sparse_selector;
    opt2_sparse.dim_sparse = dim_sparse;

    SPMM_OPT2_SPARSE_SHARED opt2_sparse_shared(graph, cu_indptr, cu_indices, cu_val, cu_vin_sparse_data, cu_vout2_sparse_shared, v_num, e_num, dim_max);
    opt2_sparse_shared.vin_sparse_selector = cu_vin_sparse_selector;
    opt2_sparse_shared.dim_sparse = dim_sparse;

#define CHECK
#define TIMING

    for (int dim = dim_min; dim <= dim_max; dim += interval)
    {
        // cout << "dim = " << dim << endl;

#ifdef CHECK
        spmm_cusparse(cu_indptr, cu_indices, cu_val, cu_vin_sparse, cu_vout_ref, v_num, e_num, dim, 0);

        // for(int i = 0; i < 300; i++){
        //     cout<<cu_vout_ref[i]<<" ";
        // }
        // cout<<endl<<endl;

        // spmm_cusparse(cu_indptr_new, cu_indices_new, cu_val, cu_vin, cu_vout_ref_new, v_num, e_num, dim, 0);
        // spmm_cusparse_coo(cu_coo_row, cu_indices, cu_val, cu_vin, cu_vout_ref_coo, v_num, e_num, dim, 0);
        // opt_sort.do_test(false, dim);
        // opt_sort2.do_test(false, dim);
        // opt_sort_innerloop.do_test(false, dim);
        // opt.do_test(false, dim);
        opt2.do_test(false, dim);

        // for(int i = 0; i < 300; i++){
        //     cout<<cu_vout2[i]<<" ";
        // }
        // cout<<endl<<endl;

        opt2_sparse.do_test(false, dim);

        opt2_sparse_shared.do_test(false, dim);

        // for(int i = 0; i < 300; i++){
        //     cout<<cu_vout2_sparse[i]<<" ";
        // }
        // cout<<endl<<endl;

        // opt_innerloop.do_test(false, dim);
        // gnna.do_test(false, dim);
        // spmm_col.do_test(false, dim);

        bool has_err = 0;
        // cout << "checking cusparse_coo" << endl;
        // check_err(cu_vout_ref_coo, cu_vout_ref, v_num * dim, has_err);
        // cout << "checking opt_sort" << endl;
        // check_err(cu_vout_new, cu_vout_ref_new, v_num * dim, has_err);
        // cout << "checking opt_sort2" << endl;
        // check_err(cu_vout_new2, cu_vout_ref_new, v_num * dim, has_err);
        // cout << "checking opt_sort_innerloop" << endl;
        // check_err(cu_vout_inner_new, cu_vout_ref_new, v_num * dim, has_err);
        // cout << "checking opt" << endl;
        // check_err(cu_vout, cu_vout_ref, v_num * dim, has_err);
        cout << "checking opt2" << endl;
        check_err(cu_vout2, cu_vout_ref, v_num * dim, has_err);

        cout << "checking opt2_sparse" << endl;
        check_err(cu_vout2_sparse, cu_vout_ref, v_num * dim, has_err);

        cout << "checking opt2_sparse_shared" << endl;
        check_err(cu_vout2_sparse_shared, cu_vout_ref, v_num * dim, has_err);

        // cout << "checking opt_innerloop" << endl;
        // check_err(cu_vout_inner, cu_vout_ref, v_num * dim, has_err);
        // cout << "checking gnna" << endl;
        // check_err(cu_vout_gnna, cu_vout_ref, v_num * dim, has_err);
        // cout << "checking spmm_col" << endl;
        // check_err(cu_vout_col, cu_vout_ref, v_num * dim, has_err);

#endif

#ifdef TIMING
        string outstr = to_string(current_file_cnt) + "/" + to_string(total_file_cnt) + " " + graph + " " + to_string(dim);

        double t_cusparse = spmm_cusparse(cu_indptr, cu_indices, cu_val, cu_vin, cu_vout_ref, v_num, e_num, dim, 10);
        cout << outstr << " cusparse " << t_cusparse * 1000 << endl;

        // double t_cusparse_coo = spmm_cusparse_coo(cu_coo_row, cu_indices, cu_val, cu_vin, cu_vout_ref_coo, v_num, e_num, dim, 10);
        // cout << outstr << " cusparse_coo " << t_cusparse_coo * 1000000 << endl;

        // double t_gnna = gnna.do_test(true, dim);
        // cout << outstr << " gnna " << t_gnna * 1000000 << endl;

        // double t_spmm_col = spmm_col.do_test(true, dim);
        // cout << outstr << " spmm_col " << t_spmm_col * 1000000 << endl;

        // double t_opt_sort = opt_sort.do_test(true, dim);
        // cout << outstr << " opt_sort " << t_opt_sort * 1000000 << endl;

        // double t_opt_sort2 = opt_sort2.do_test(true, dim);
        // cout << outstr << " opt_sort2 " << t_opt_sort2 * 1000000 << endl;

        // double t_opt_sort_innerloop = opt_sort_innerloop.do_test(true, dim);
        // cout << outstr << " opt_sort_innerloop " << t_opt_sort_innerloop * 1000000 << endl;

        // double t_opt = opt.do_test(true, dim);
        // cout << outstr << " opt " << t_opt * 1000000 << endl;

        double t_opt2 = opt2.do_test(true, dim);
        cout << outstr << " opt2 " << t_opt2 * 1000 << endl;

        double t_opt2_sparse = opt2_sparse.do_test(true, dim);
        cout << outstr << " opt2_sparse " << t_opt2_sparse * 1000 << endl;

        double t_opt2_sparse_shared = opt2_sparse_shared.do_test(true, dim);
        cout << outstr << " opt2_sparse_shared " << t_opt2_sparse_shared * 1000 << endl;

        // double t_opt_innerloop = opt_innerloop.do_test(true, dim);
        // cout << outstr << " opt_innerloop " << t_opt_innerloop * 1000000 << endl;

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
    cudaFree(cu_vout_inner);
    cudaFree(cu_vout_new);
    cudaFree(cu_vout_new2);
    cudaFree(cu_vout_inner_new);
    cudaFree(cu_vout_gnna);
    cudaFree(cu_vout_ref);
    cudaFree(cu_vout_ref_new);
    cudaFree(cu_vout_ref_coo);
    cudaFree(cu_vout_col);

    cudaFree(cu_vin_sparse);
    cudaFree(cu_vin_sparse_data);
    cudaFree(cu_vin_sparse_selector);
    cudaFree(cu_vout2_sparse);
    cudaFree(cu_vout2_sparse_shared);
}

int main(int argc, char *argv[])
{
    if (argc > 2)
    {
        string arg_graph(argv[1]);
        int dim = atoi(argv[2]);
        cout << "dir = " << base_dir << endl;
        test_graph(arg_graph, dim);
    }
    else
    {
        string folder_path = "/home/xix22010/cuda_projects/hpc_data/";
        string extension = ".config";

        total_file_cnt = 0;
        for (const auto &file : filesystem::directory_iterator(folder_path))
        {
            if (file.path().extension() == extension)
            {
                total_file_cnt++;
            }
        }

        current_file_cnt = 0;
        for (const auto &file : filesystem::directory_iterator(folder_path))
        {
            if (file.path().extension() == extension)
            {
                current_file_cnt++;

                string graph = file.path().stem().string();
                // if (!(graph == "wikikg2" || graph == "rabbit_wikikg2"))
                // {
                //     continue;
                // }
                test_graph(graph, 0);
                cudaDeviceSynchronize();
            }
        }
    }

    return 0;
}