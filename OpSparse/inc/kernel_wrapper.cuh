#include "cuda_common.h"
#include "CSR.h"
#include "Meta.h"
#include "define.h"
#include <cuda_profiler_api.h>
#include "setup.cuh"
#include "binning.cuh"
#include "symbolic.cuh"
#include "numeric.cuh"
#include "Timings.h"

void cudaruntime_warmup(){
    int *d;
    CHECK_ERROR(cudaMalloc(&d, 4));
    CHECK_ERROR(cudaFree(d));
    CHECK_ERROR(cudaDeviceSynchronize());
}


void h_compute_flop(const CSR& A, const CSR& B, CSR& C, Meta& meta){
    mint BS = 1024;
    mint GS = div_up(A.M, BS);
    k_compute_flop<<<GS, BS>>>(A.d_rpt, A.d_col, B.d_rpt, C.M, C.d_rpt, C.d_rpt + C.M);
}



long compute_flop(mint *row_pointer_A, mint *col_index_A, mint *row_pointer_B, mint M, mint *floprC){
    long total_flop = 0;
#pragma omp parallel
{
    long thread_flop = 0;
#pragma omp for
    for(mint i = 0; i < M; i++){
        long local_sum = 0;
        for(mint j = row_pointer_A[i]; j < row_pointer_A[i+1]; j++){
            local_sum += row_pointer_B[col_index_A[j]+1] - row_pointer_B[col_index_A[j]];
        }
        floprC[i] = local_sum;
        thread_flop += local_sum;
    }
#pragma omp critical
{
    total_flop += thread_flop;
}
}
    return total_flop;
}

long compute_flop(const CSR& A, const CSR& B, mint* row_flop){
    return compute_flop(A.rpt, A.col, B.rpt, A.M, row_flop);
}

long compute_flop(const CSR& A, const CSR& B){
    mint *row_flop = new mint [A.M];
    long flop = compute_flop(A.rpt, A.col, B.rpt, A.M, row_flop);
    delete [] row_flop;
    return flop;
}


// setup
void h_setup(const CSR& A, const CSR& B, CSR& C, Meta& meta, Timings& timing){
    meta.allocate_rpt(C); // allocate C.rpt, other init procedure, default stream
    cudaMemset(C.d_rpt + C.M, 0, sizeof(mint));
    h_compute_flop(A, B, C, meta); // compute flop, stream[0]
    meta.allocate(C); // allocate other memory    
    CHECK_ERROR(cudaMemcpy(meta.max_row_nnz, C.d_rpt + C.M, sizeof(mint), cudaMemcpyDeviceToHost));
}

// symbolic binning
inline void h_symbolic_binning(CSR &C, Meta& meta){
    meta.memset_all(0); // memset d_bin_size
    mint BS = 1024;
    mint GS = div_up(C.M, BS);
    if(*meta.max_row_nnz <= 26){
        k_binning_small<<<GS, BS>>>(meta.d_bins, C.M);
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
        k_symbolic_binning<<<GS, BS, 0, meta.stream[0]>>>(
            C.d_rpt, C.M, meta.d_bin_size);
        meta.D2H_bin_size(0);
        meta.memset_bin_size(0);
        meta.bin_offset[0] = 0;
        for(int i = 0; i < NUM_BIN - 1; i++){
            meta.bin_offset[i+1] = meta.bin_offset[i] + meta.bin_size[i];
        }
        meta.H2D_bin_offset(0);
        k_symbolic_binning2<<<GS, BS, 0, meta.stream[0]>>>(
            C.d_rpt, C.M, 
            meta.d_bins, meta.d_bin_size, meta.d_bin_offset);
    }
}


void h_symbolic(const CSR& A, const CSR& B, CSR& C, Meta& meta){
    //double t0, t1;
    if(meta.bin_size[5]){
        k_symbolic_shared_hash_tb<8192><<<meta.bin_size[5], 1024, 0, meta.stream[5]>>>(
            A.d_rpt, A.d_col, B.d_rpt, B.d_col,
            meta.d_bins + meta.bin_offset[5],
            C.d_rpt);
    }
    mint *d_fail_bins, *d_fail_bin_size;
    mint fail_bin_size = 0;
    if(meta.bin_size[7]){ // shared hash with fail
        //t0 = fast_clock_time();
        if(meta.bin_size[7] + 1 <= meta.cub_storage_size/sizeof(mint)){
            d_fail_bins = meta.d_cub_storage;
            d_fail_bin_size = meta.d_cub_storage + meta.bin_size[7];
        }
        else{ // allocate global memory
            CHECK_ERROR(cudaMalloc(&d_fail_bins, (meta.bin_size[7] + 1) * sizeof(mint)));
            d_fail_bin_size = d_fail_bins + meta.bin_size[7];
        }
        CHECK_ERROR(cudaMemsetAsync(d_fail_bin_size, 0, sizeof(mint), meta.stream[7]));
        CHECK_ERROR(cudaFuncSetAttribute(k_symbolic_max_shared_hash_tb_with_fail, 
            cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
        k_symbolic_max_shared_hash_tb_with_fail
            <<<meta.bin_size[7], 1024, 98304, meta.stream[7]>>>(
            A.d_rpt, A.d_col, B.d_rpt, B.d_col,
            meta.d_bins + meta.bin_offset[7],
            d_fail_bins, d_fail_bin_size,
            C.d_rpt);

    }
    if(meta.bin_size[6]){
        k_symbolic_large_shared_hash_tb<<<meta.bin_size[6], 1024, 0, meta.stream[6]>>>(
            A.d_rpt, A.d_col, B.d_rpt, B.d_col,
            meta.d_bins + meta.bin_offset[6],
            C.d_rpt);
    }
    if(meta.bin_size[0]){
        mint BS = PWARP_ROWS * PWARP;
        mint GS = div_up(meta.bin_size[0], PWARP_ROWS);
        k_symbolic_shared_hash_pwarp<<<GS, BS, 0, meta.stream[0]>>>(
            A.d_rpt, A.d_col, B.d_rpt, B.d_col,
            meta.d_bins + meta.bin_offset[0],
            meta.bin_size[0],
            C.d_rpt);
    }

    if(meta.bin_size[7]){
    CHECK_ERROR(cudaMemcpyAsync(&fail_bin_size, d_fail_bin_size, sizeof(mint), cudaMemcpyDeviceToHost, meta.stream[7]));
    CHECK_ERROR(cudaStreamSynchronize(meta.stream[7]));
        if(fail_bin_size){ // global hash
            //printf("inside h_symbolic fail_bin_size %d\n", fail_bin_size);
            mint max_tsize = *meta.max_row_nnz * SYMBOLIC_SCALE_LARGE;
            meta.global_mem_pool_size = fail_bin_size * max_tsize * sizeof(mint);
            CHECK_ERROR(cudaMalloc(&meta.d_global_mem_pool, meta.global_mem_pool_size));
            meta.global_mem_pool_malloced = true;
            k_symbolic_global_hash_tb<<<fail_bin_size, 1024, 0, meta.stream[7]>>>(
                A.d_rpt, A.d_col, B.d_rpt, B.d_col,
                d_fail_bins,
                C.d_rpt, meta.d_global_mem_pool, max_tsize);
        }
    }
    

    if(meta.bin_size[4]){
        k_symbolic_shared_hash_tb<4096><<<meta.bin_size[4], 512, 0, meta.stream[4]>>>(
            A.d_rpt, A.d_col, B.d_rpt, B.d_col,
            meta.d_bins + meta.bin_offset[4],
            C.d_rpt);
    }


    if(meta.bin_size[3]){
        k_symbolic_shared_hash_tb<2048><<<meta.bin_size[3], 256, 0, meta.stream[3]>>>(
            A.d_rpt, A.d_col, B.d_rpt, B.d_col,
            meta.d_bins + meta.bin_offset[3],
            C.d_rpt);
    }
    if(meta.bin_size[2]){
        k_symbolic_shared_hash_tb<1024><<<meta.bin_size[2], 128, 0, meta.stream[2]>>>(
            A.d_rpt, A.d_col, B.d_rpt, B.d_col,
            meta.d_bins + meta.bin_offset[2],
            C.d_rpt);
    }
    if(meta.bin_size[1]){
        k_symbolic_shared_hash_tb<512><<<meta.bin_size[1], 64, 0, meta.stream[1]>>>(
            A.d_rpt, A.d_col, B.d_rpt, B.d_col,
            meta.d_bins + meta.bin_offset[1],
            C.d_rpt);
    }


    if(meta.bin_size[7] && meta.bin_size[7] + 1 > meta.cub_storage_size/sizeof(mint)){
        CHECK_ERROR(cudaFree(d_fail_bins));
    }
}


inline void h_numeric_binning(CSR& C, Meta& meta){
    meta.memset_all(0);
    mint BS = 1024;
    mint GS = div_up(C.M, BS);
    k_numeric_binning<<<GS, BS, 0 , meta.stream[0]>>>(C.d_rpt, C.M,
        meta.d_bin_size, meta.d_total_nnz, meta.d_max_row_nnz);
    meta.D2H_all(0);
    CHECK_ERROR(cudaStreamSynchronize(meta.stream[0]));
    if(*meta.max_row_nnz <= 16){
        k_binning_small<<<GS, BS>>>(meta.d_bins, C.M);
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
    }
}



inline void h_numeric_full_occu(const CSR& A, const CSR& B, CSR& C, Meta& meta){

    if(meta.bin_size[6]){
        CHECK_ERROR(cudaFuncSetAttribute(k_numeric_max_shared_hash_tb_half_occu, 
            cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
        k_numeric_max_shared_hash_tb_half_occu<<<meta.bin_size[6], 1024, 98304, meta.stream[6]>>>
            (A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val,
            meta.d_bins + meta.bin_offset[6],
            C.d_rpt, C.d_col, C.d_val);
    }


    if(meta.bin_size[7]){ // global bin
        //printf("inside h_numeric_phase max_row_nnz %d\n", *meta.max_row_nnz);
        mint max_tsize = *meta.max_row_nnz * NUMERIC_SCALE_LARGE;
        size_t global_size = meta.bin_size[7] * max_tsize * (sizeof(mint) + sizeof(mdouble));
        if(meta.global_mem_pool_malloced){
            if(global_size <= meta.global_mem_pool_size){
                // do nothing
            }
            else{
                CHECK_ERROR(cudaFree(meta.d_global_mem_pool));
                CHECK_ERROR(cudaMalloc(&meta.d_global_mem_pool, global_size));
            }
        }
        else{
            CHECK_ERROR(cudaMalloc(&meta.d_global_mem_pool, global_size));
            meta.global_mem_pool_size = global_size;
            meta.global_mem_pool_malloced = true;
        }
        k_numeric_global_hash_tb_full_occu<<<meta.bin_size[7], 1024, 0, meta.stream[7]>>>
            (A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val,
            meta.d_bins + meta.bin_offset[7], max_tsize, meta.d_global_mem_pool,
            C.d_rpt, C.d_col, C.d_val);
    }

    if(meta.bin_size[5]){
        k_numeric_shared_hash_tb_full_occu<4096, 1024>
            <<<meta.bin_size[5], 1024, 0, meta.stream[5]>>>
            (A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val,
            meta.d_bins + meta.bin_offset[5],
            C.d_rpt, C.d_col, C.d_val);
    }
    if(meta.bin_size[0]){
        mint BS = NUMERIC_PWARP_ROWS * NUMERIC_PWARP;
        mint GS = div_up(meta.bin_size[0], NUMERIC_PWARP_ROWS);
        k_numeric_shared_hash_pwarp<<<GS, BS, 0, meta.stream[0]>>>(
            A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val,
            meta.d_bins + meta.bin_offset[0], meta.bin_size[0],
            C.d_rpt, C.d_col, C.d_val);
    }

    if(meta.bin_size[4]){
        k_numeric_shared_hash_tb_full_occu<2048, 512>
            <<<meta.bin_size[4], 512, 0, meta.stream[4]>>>
            (A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val,
            meta.d_bins + meta.bin_offset[4],
            C.d_rpt, C.d_col, C.d_val);
    }
    if(meta.bin_size[3]){
        k_numeric_shared_hash_tb_full_occu<1024, 256>
            <<<meta.bin_size[3], 256, 0, meta.stream[3]>>>
            (A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val,
            meta.d_bins + meta.bin_offset[3],
            C.d_rpt, C.d_col, C.d_val);
    }

    if(meta.bin_size[2]){
        k_numeric_shared_hash_tb_full_occu<512, 128>
            <<<meta.bin_size[2], 128, 0, meta.stream[2]>>>
            (A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val,
            meta.d_bins + meta.bin_offset[2],
            C.d_rpt, C.d_col, C.d_val);
    }
    if(meta.bin_size[1]){
        k_numeric_shared_hash_tb_full_occu<256, 64>
            <<<meta.bin_size[1], 64, 0, meta.stream[1]>>>
            (A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val,
            meta.d_bins + meta.bin_offset[1],
            C.d_rpt, C.d_col, C.d_val);
    }

    if(meta.global_mem_pool_malloced){
        CHECK_ERROR(cudaFree(meta.d_global_mem_pool));
    }
}


