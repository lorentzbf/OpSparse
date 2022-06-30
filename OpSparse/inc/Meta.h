#ifndef __Z_META__
#define __Z_META__

#include "cuda_common.h"
#include "define.h"

class CSR;
class Meta{
    public:
    // first, allocate C.rpt. 
    // d_row_flop, d_estimated_row_nnz, d_row_nnz are all reused with C.rpt

    // combined memory
    mint *d_combined_mem; // second, allocate for all others
    mint *combined_mem; // second, allocate for all others

    // meta data
    mint M; // number of rows
    mint N; // number of cols
    mint *d_bins; // size M
    mint *d_bin_size; // size NUM_BIN
    mint *d_bin_offset; // size NUM_BIN
    mint *d_max_row_nnz; // size 1
    mint *d_total_nnz; // size 1
    mint *d_cub_storage; // size variable
    mint *bin_size; // size NUM_BIN
    mint *bin_offset; // size NUM_BIN
    mint *max_row_nnz; // size 1
    mint *total_nnz; // size 1
    size_t cub_storage_size;
    cudaStream_t *stream;


    // symbolic global and numeric global, is allocated at runtime
    mint* d_global_mem_pool; // size unknown, allocated at runtime
    size_t global_mem_pool_size;
    bool global_mem_pool_malloced;

    // ********************************************************
    // public method
    Meta(){}
    Meta(const Meta&) = delete;
    Meta &operator=(const Meta&) = delete;
    Meta(CSR &C); // init and first malloc
    void allocate_rpt(CSR& C);
    void allocate(CSR &C); // malloc conbined mem and pin the variables
    void release();

    void memset_bin_size(mint stream_idx); // set d_bin_size only to 0
    void memset_all(mint stream_idx); // set d_bin_size and other to 0
    void D2H_bin_size(mint stream_idx);
    void D2H_all(mint stream_idx);
    void H2D_bin_offset(mint stream_idx);
    ~Meta();
};

#endif
