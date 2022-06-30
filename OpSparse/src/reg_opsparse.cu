
#include "kernel_wrapper.cuh"
#include <fstream>
#include <cuda_profiler_api.h>
#include <cub/cub.cuh>
#include "cusparse_spgemm.h"
#include "Timings.h"


void opsparse(const CSR& A, const CSR& B, CSR& C, Meta& meta, Timings& timing){
    
    double t0, t1;
    t1 = t0 = fast_clock_time();
    C.M = A.M;
    C.N = B.N;
    C.nnz = 0;
    h_setup(A, B, C, meta, timing);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.setup = fast_clock_time() - t0;

    // symbolic binning
    t0 = fast_clock_time();
    h_symbolic_binning(C, meta);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.symbolic_binning = fast_clock_time() - t0;


    // symbolic phase
    t0 = fast_clock_time();
    h_symbolic(A, B, C, meta);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.symbolic = fast_clock_time() - t0;



    // numeric binning, exclusive sum, and allocate C
    meta.memset_all(0);
    mint BS = 1024;
    mint GS = div_up(C.M, BS);
    k_numeric_binning<<<GS, BS, 0 , meta.stream[0]>>>(C.d_rpt, C.M,
        meta.d_bin_size, meta.d_total_nnz, meta.d_max_row_nnz);
    meta.D2H_all(0);
    CHECK_ERROR(cudaStreamSynchronize(meta.stream[0]));
    C.nnz = *meta.total_nnz;

    if(*meta.max_row_nnz <= 16){
        k_binning_small<<<GS, BS>>>(meta.d_bins, C.M);
        CHECK_ERROR(cudaMalloc(&C.d_col, C.nnz * sizeof(mint)));
        meta.bin_size[0] = C.M;
        for(int i = 1; i< NUM_BIN; i++){
            meta.bin_size[i] = 0;
        }
        meta.bin_offset[0] = 0;
        for(int i = 1; i < NUM_BIN; i++){
            meta.bin_offset[i] = C.M;
        }
    }
    else{
        meta.memset_bin_size(0);
        meta.bin_offset[0] = 0;
        for(int i = 0; i < NUM_BIN - 1; i++){
            meta.bin_offset[i+1] = meta.bin_offset[i] + meta.bin_size[i];
        }
        meta.H2D_bin_offset(0);

        k_numeric_binning2<<<GS, BS, 0, meta.stream[0]>>>(C.d_rpt, C.M,
            meta.d_bins, meta.d_bin_size, meta.d_bin_offset);
        CHECK_ERROR(cudaMalloc(&C.d_col, C.nnz * sizeof(mint)));
    }
    CHECK_ERROR(cudaDeviceSynchronize());

    cub::DeviceScan::ExclusiveSum(meta.d_cub_storage, meta.cub_storage_size, C.d_rpt, C.d_rpt, C.M + 1);
    CHECK_ERROR(cudaMalloc(&C.d_val, C.nnz * sizeof(mdouble)));
    CHECK_ERROR(cudaDeviceSynchronize());

    // numeric   
    t0 = fast_clock_time();
    h_numeric_full_occu(A, B, C, meta);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.numeric= fast_clock_time() - t0;

    // cleanup
    t0 = fast_clock_time();
    meta.release();
    timing.cleanup = fast_clock_time() - t0;
    timing.total = fast_clock_time() - t1;
}

int main(int argc, char **argv)
{
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
	
    CSR A, B;
    A.construct(mat1_file);
    if(mat1 == mat2){
        B = A;
    }
    else{
        B.construct(mat2_file);
        if(A.N == B.M){
            // do nothing
        }
        else if(A.N < B.M){
            CSR tmp(B, A.N, B.N, 0, 0);
            B = tmp;
        }
        else{
            CSR tmp(A, A.M, B.M, 0, 0);
            A = tmp;
        }
    }

    A.H2D();
    B.H2D();

    long total_flop = compute_flop(A, B);
    CSR C;
    cudaruntime_warmup();
    Meta meta;
    {
        Timings timing;
        opsparse(A, B, C, meta, timing);
        C.release();
    }
    
    mint iter = 10;
    Timings timing, bench_timing;
    for(mint i = 0; i < iter; i++){
        opsparse(A, B, C, meta, timing);
        bench_timing += timing;
        if(i < iter - 1){
            C.release();
        }
    }
    bench_timing /= iter;

    printf("%s ",mat1.c_str());
    bench_timing.reg_print(total_flop * 2);

    // compare result

    //C.D2H();
    //CSR C_ref;
    //cusparse_spgemm(&A, &B, &C_ref);
    //C_ref.D2H();
    //if(C == C_ref){
    //    printf("pass\n");
    //}
    //else{
    //    printf("error\n");
    //}
    
    A.release();
    B.release();

    C.release();
    return 0;
}


