#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <math.h>

#include <cuda.h>
#include <helper_cuda.h>
#include <cusparse_v2.h>

#include <nsparse.hpp>
#include <CSR.hpp>
#include <SpGEMM.hpp>
#include <HashSpGEMM_volta.hpp>
#include "Timing.hpp"

typedef int IT;
//#ifdef FLOAT
//typedef float VT;
//#else
//typedef double VT;
//#endif
typedef double VT;


template <bool sort, class idType, class valType>
void SpGEMM_Hash_Detail(CSR<idType, valType>& a, CSR<idType, valType>& b, CSR<idType, valType> &c, Timing& timing)
{
    double t0, t1;
    t0 = t1  = fast_clock_time();

    BIN<idType, BIN_NUM>* bin = new BIN<idType, BIN_NUM>(a.nrow);

    c.nrow = a.nrow;
    c.ncolumn = b.ncolumn;
    c.device_malloc = true;
    cudaMalloc((void **)&(c.d_rpt), sizeof(idType) * (c.nrow + 1));
    timing.setup = fast_clock_time() - t0;

    t0 = fast_clock_time();
    bin->set_max_bin(a.d_rpt, a.d_colids, b.d_rpt, a.nrow, TS_S_P, TS_S_T);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.symbolic_binning = fast_clock_time() - t0;

    t0 = fast_clock_time();
    hash_symbolic(a, b, c, *bin);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.symbolic = fast_clock_time() - t0;
    
    t0 = fast_clock_time();
    thrust::exclusive_scan(thrust::device, bin->d_count, bin->d_count + (a.nrow + 1), c.d_rpt, 0);
    cudaMemcpy(&(c.nnz), c.d_rpt + c.nrow, sizeof(idType), cudaMemcpyDeviceToHost);
    timing.prefix = fast_clock_time() - t0;
    
    t0 = fast_clock_time();
    cudaMalloc((void **)&(c.d_colids), sizeof(idType) * (c.nnz));
    cudaMalloc((void **)&(c.d_values), sizeof(valType) * (c.nnz));
    timing.allocate = fast_clock_time() - t0;

    t0 = fast_clock_time();
    bin->set_min_bin(a.nrow, TS_N_P, TS_N_T);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.numeric_binning = fast_clock_time() - t0;

    t0 = fast_clock_time();
    hash_numeric<idType, valType, sort>(a, b, c, *bin);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.numeric = fast_clock_time() - t0;

    t0 = fast_clock_time();
    delete bin;
    timing.cleanup = fast_clock_time() - t0;
    timing.total = fast_clock_time() - t1;

}


template <class idType, class valType>
void run_spgemm(CSR<idType, valType>& a, CSR<idType, valType>& b, CSR<idType, valType> &c)
{

    /* Memcpy A and B from Host to Device */
    a.memcpyHtD();
    b.memcpyHtD();
  
    /* Count flop of SpGEMM computation */
    long long int flop_count;
    get_spgemm_flop(a, b, flop_count);

    /* Execution of SpGEMM on Device */
    Timing warmup_timing, bench_timing, timing;
    
    SpGEMM_Hash_Detail<true, idType, valType>(a, b, c, warmup_timing);
    c.release_csr();

    for (int i = 0; i < SpGEMM_TRI_NUM; i++) {
        SpGEMM_Hash_Detail<true, idType, valType>(a, b, c, bench_timing);
        if (i < SpGEMM_TRI_NUM - 1) {
            c.release_csr();
        }
        timing += bench_timing;
    }
    timing /= SpGEMM_TRI_NUM;
    timing.print(flop_count);


    c.memcpyDtH();
    c.release_csr();

#ifdef sfDEBUG
    CSR<IT, VT> cusparse_c;
    SpGEMM_cuSPARSE(a, b, cusparse_c);
    cusparse_c.memcpyDtH();
    if (c == cusparse_c) {
        //cout << "HashSpGEMM is correctly executed" << endl;
        cout << "pass" << endl;
    }
    else{
        cout << "fail" << endl;
    }
    cout << "Nnz of A: " << a.nnz << endl; 
    cout << "Number of intermediate products: " << flop_count / 2 << endl; 
    cout << "Nnz of C: " << c.nnz << endl; 
    cusparse_c.release_cpu_csr();
#endif

    a.release_csr();
    b.release_csr();

}

/*Main Function*/
int main(int argc, char *argv[])
{
    CSR<IT, VT> a, b, c;

    std::string mat1, mat2;
    mat1 = "can_24";
    mat2 = "can_24";
    if(argc == 2){
        mat1 = argv[1];
        mat2 = argv[1];
    }
    if(argc >= 3){
        mat1 = argv[1];
        mat2 = argv[2];
    }
    std::string mat1_file;
    if(mat1.find("ER") != std::string::npos){
        mat1_file = "../matrix/ER/" + mat1 +".mtx";
    }
    else if(mat1.find("G500") != std::string::npos){
        mat1_file = "../matrix/G500/" + mat1 +".mtx";
    }
    else{
        mat1_file = "../matrix/suite_sparse/" + mat1 + "/" + mat1 +".mtx";
    }
    std::string mat2_file;
    if(mat2.find("ER") != std::string::npos){
        mat2_file = "../matrix/ER/" + mat2 +".mtx";
    }
    else if(mat2.find("G500") != std::string::npos){
        mat2_file = "../matrix/G500/" + mat2 +".mtx";
    }
    else{
        mat2_file = "../matrix/suite_sparse/" + mat2 + "/" + mat2 +".mtx";
    }

    /* Set CSR reding from MM file or generating random matrix */
    //cout << "Initialize Matrix A" << endl;
    //cout << "Read matrix data from " << argv[1] << endl;
    a.init_data_from_mtx(mat1_file);

    //cout << "Initialize Matrix B" << endl;
    //cout << "Read matrix data from " << argv[1] << endl;
    b.init_data_from_mtx(mat2_file);
  
    /* Execution of SpGEMM on GPU */
    printf("%s ", mat1.c_str());
    run_spgemm(a, b, c);
    
    a.release_cpu_csr();
    b.release_cpu_csr();
    c.release_cpu_csr();
  
    return 0;

}

