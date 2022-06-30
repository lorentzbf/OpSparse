
#include "Meta.h"
#include "CSR.h"
#include <cub/cub.cuh>

Meta::Meta(CSR &C){
    allocate_rpt(C);
}

void Meta::allocate_rpt(CSR &C){
    CHECK_ERROR(cudaMalloc(&C.d_rpt, (C.M + 1)*sizeof(mint)));
}

void Meta::allocate(CSR& C){
    M = C.M;
    N = C.N;
    stream = new cudaStream_t [NUM_BIN];
    for(int i = 0; i < NUM_BIN; i++){
        CHECK_ERROR(cudaStreamCreate(stream + i));
    }
        
    cub::DeviceScan::ExclusiveSum(nullptr, cub_storage_size, C.d_rpt, C.d_rpt, M + 1); // calculate tmp_storage_size in bytes

    mint d_combined_size = M  + 2 * NUM_BIN + 2 + cub_storage_size/(sizeof(mint));
    CHECK_ERROR(cudaMalloc(&d_combined_mem, d_combined_size * sizeof(mint)));
    mint combined_size = 2 * NUM_BIN + 2;
    combined_mem = (mint *)malloc(combined_size * sizeof(mint));
    assert(combined_mem != nullptr);

    d_bins = (mint *)d_combined_mem; // size M
    d_bin_size = (mint *)d_combined_mem + M; // size NUM_BIN
    d_max_row_nnz = d_bin_size + NUM_BIN; // size 1
    d_total_nnz = d_bin_size + NUM_BIN + 1; // size 1
    d_bin_offset = d_total_nnz + 1; // size NUM_BIN
    d_cub_storage = d_bin_offset + 1;

    bin_size = (mint*) combined_mem; // size NUM_BIN
    max_row_nnz = bin_size + NUM_BIN; // size 1
    total_nnz = bin_size + NUM_BIN + 1; // size 1
    bin_offset = bin_size + NUM_BIN + 2; // size NUM_BIN
    
    d_global_mem_pool = nullptr;
    global_mem_pool_size = 0;
    global_mem_pool_malloced = false;
}

void Meta::release(){
    cudaFree(d_combined_mem);
    d_combined_mem = nullptr;
    if(stream != nullptr){
        for(int i = 0; i < NUM_BIN; i++){
            cudaStreamDestroy(stream[i]);
        }
        delete [] stream;
        stream = nullptr;
    }
    delete [] combined_mem;
    combined_mem = nullptr;
}

Meta::~Meta(){
    release();
}


void Meta::memset_all(mint stream_idx = 1){
    CHECK_ERROR(cudaMemsetAsync(d_bin_size, 0, (NUM_BIN + 2) * sizeof(mint), stream[stream_idx]));
    //CHECK_ERROR(cudaMemset(d_bin_size, 0, (NUM_BIN + 5) * sizeof(mint)));
}
void Meta::memset_bin_size(mint stream_idx = 1){
    CHECK_ERROR(cudaMemsetAsync(d_bin_size, 0, NUM_BIN * sizeof(mint), stream[stream_idx]));
    //CHECK_ERROR(cudaMemset(d_bin_size, 0, (NUM_BIN + 5) * sizeof(mint)));
}

void Meta::D2H_all(mint stream_idx = 0){
    CHECK_ERROR(cudaMemcpyAsync(bin_size, d_bin_size, (NUM_BIN + 2) * sizeof(mint), cudaMemcpyDeviceToHost, stream[stream_idx]));
    //CHECK_ERROR(cudaMemcpy(bin_size, d_bin_size, NUM_BIN * sizeof(mint), cudaMemcpyHostToDevice));
}

void Meta::D2H_bin_size(mint stream_idx = 0){
    CHECK_ERROR(cudaMemcpyAsync(bin_size, d_bin_size, NUM_BIN * sizeof(mint), cudaMemcpyDeviceToHost, stream[stream_idx]));
    //CHECK_ERROR(cudaMemcpy(bin_size, d_bin_size, NUM_BIN * sizeof(mint), cudaMemcpyHostToDevice));
}

void Meta::H2D_bin_offset(mint stream_idx = 0){
    CHECK_ERROR(cudaMemcpyAsync(d_bin_offset, bin_offset, NUM_BIN * sizeof(mint), cudaMemcpyHostToDevice, stream[stream_idx]));
}


