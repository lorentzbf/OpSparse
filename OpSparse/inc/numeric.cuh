#ifndef __Z_NUMERIC_CUH__
#define __Z_NUMERIC_CUH__

#include "cuda_common.h"
#include "define.h"


// full occu
__global__ void __launch_bounds__(NUMERIC_PWARP_BLOCK_SIZE, 2) k_numeric_shared_hash_pwarp(
    const mint * __restrict__ d_arpt, const mint * __restrict__ d_acol, 
    const mdouble * __restrict__ d_aval,
    const mint * __restrict__ d_brpt, const mint * __restrict__ d_bcol, 
    const mdouble * __restrict__ d_bval,
    mint *d_bins, mint bin_size,
    mint *d_crpt, mint *d_ccol, mdouble* d_cval){

    //long long t0 = clock64();

    mint i = threadIdx.x + blockIdx.x * blockDim.x;
    mint tid = threadIdx.x % NUMERIC_PWARP;
    mint rid = i / NUMERIC_PWARP;
    mint block_rid = rid % NUMERIC_PWARP_ROWS;
    __shared__ mint shared_mem[NUMERIC_PWARP_ROWS * NUMERIC_PWARP_TSIZE * (sizeof(mint) + sizeof(mdouble))/sizeof(mint)];
    const mint tsize = NUMERIC_PWARP_TSIZE - 1;
    mint *mono_shared_col = shared_mem;
    mint *mono_shared_offset = shared_mem + NUMERIC_PWARP_ROWS * tsize;
    mdouble *mono_shared_val = (mdouble*)(shared_mem + NUMERIC_PWARP_ROWS * NUMERIC_PWARP_TSIZE);
    mint j, k;
    for(j = threadIdx.x; j < NUMERIC_PWARP_ROWS * tsize; j += blockDim.x){
        mono_shared_col[j] = -1;
        mono_shared_val[j] = 0;
    }
    if(threadIdx.x < NUMERIC_PWARP_ROWS){
        mono_shared_offset[threadIdx.x] = 0;
    }
    if(rid >= bin_size){
        return;
    }
    __syncthreads();

    rid = d_bins[rid];
    mint *shared_col = mono_shared_col + block_rid * tsize;
    //mint *shared_offset = shared_col + tsize;
    mdouble *shared_val = mono_shared_val + block_rid * tsize;
    mint acol, bcol, hash, old;
    mdouble aval, bval;
    for(j = d_arpt[rid] + tid; j < d_arpt[rid + 1]; j += NUMERIC_PWARP){ // pwarp per row, thread per a item, thread per b row
        acol = d_acol[j];
        aval = d_aval[j];
        for(k = d_brpt[acol]; k < d_brpt[acol + 1]; k++){ // thread per b row
            bcol = d_bcol[k];
            bval = d_bval[k];
            hash = (bcol * HASH_SCALE) % tsize;
            while(1){
#ifdef HASH_SINGLE
                old = atomicCAS(shared_col + hash, -1, bcol);
                if(old == -1 || old == bcol){
                    atomicAdd(shared_val + hash, aval * bval);
                    break;
                }
                else{
                    hash = (hash + 1) < tsize ? hash + 1 : 0;
                }
#endif
#ifdef HASH_MULTI
                if (shared_col[hash] == bcol) {
                    atomicAdd(shared_val + hash, aval * bval);
                    break;
                }
                else if (shared_col[hash] == -1) {
                    old = atomicCAS(shared_col + hash, -1, bcol);
                    if (old == -1) {
                        atomicAdd(shared_val + hash, aval * bval);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) < tsize ? hash + 1 : 0;
                }
#endif
            }
        }
    }
    __syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash pwarp hash %lld\n", t0);
    //}
    //__syncthreads();
    //t0 = clock64();

    //__syncthreads();

    mint c_offset = d_crpt[rid];
    mint row_nnz = d_crpt[rid + 1] - d_crpt[rid];
    mint offset;
    bool valid;
    #pragma unroll
    for(j = 0; j < tsize; j += NUMERIC_PWARP){
        offset = j + tid;
        valid = offset < tsize;
        if(valid){
            acol = shared_col[offset];
            aval = shared_val[offset];
            if(acol != -1){
                offset = atomicAdd(mono_shared_offset + block_rid, 1);
            }
        }
        __syncthreads();
        if(valid && acol != -1){
            shared_col[offset] = acol;
            shared_val[offset] = aval;
        }
    }

    __syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash pwarp condense %lld\n", t0);
    //}
    //__syncthreads();
    //t0 = clock64();

    // count sort the result
    for (j = tid; j < row_nnz; j += NUMERIC_PWARP) {
        acol = shared_col[j];
        offset = 0;
        for (k = 0; k < row_nnz; k++) {
            offset += (unsigned int)(shared_col[k] - acol) >> 31;
        }
        d_ccol[c_offset + offset] = shared_col[j];
        d_cval[c_offset + offset] = shared_val[j];
    }
    //__syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash sort %lld\n", t0);
    //}
}


template <int SH_ROW, int BS>
__global__ void __launch_bounds__(1024,2) k_numeric_shared_hash_tb_full_occu(
    const mint * __restrict__ d_arpt, const mint * __restrict__ d_acol, 
    const mdouble * __restrict__ d_aval,
    const mint * __restrict__ d_brpt, const mint * __restrict__ d_bcol, 
    const mdouble * __restrict__ d_bval,
    mint *d_bins,
    mint *d_crpt, mint *d_ccol, mdouble* d_cval){

    //long long t0 = clock64();

    mint tid = threadIdx.x & (WSIZE - 1);
    mint wid = threadIdx.x / WSIZE;
    mint wnum = blockDim.x / WSIZE;
    mint j, k;
    __shared__ mint shared_mem[SH_ROW * (sizeof(mint) + sizeof(mdouble))/sizeof(mint)];
    const mint tsize = SH_ROW - 1;
    mint *shared_col = shared_mem;
    mint *shared_offset = shared_mem + (SH_ROW - 1);
    mdouble* shared_val = (mdouble*)(shared_mem + SH_ROW);

    for(j = threadIdx.x; j < tsize; j += blockDim.x){
        shared_col[j] = -1;
        shared_val[j] = 0;
    }
    if(threadIdx.x == 0){
        shared_offset[0] = 0;
    }
    __syncthreads();

    mint acol, bcol, hash, old;
    mdouble aval, bval;
    mint rid = d_bins[blockIdx.x];
    mint c_offset = d_crpt[rid];
    mint row_nnz = d_crpt[rid + 1] - d_crpt[rid];

    for(j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum){
        acol = d_acol[j];
        aval = d_aval[j];
        for(k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k+= WSIZE){
            bcol = d_bcol[k];
            bval = d_bval[k];
            hash = (bcol * HASH_SCALE) % tsize;
            while(1){
#ifdef HASH_SINGLE
                old = atomicCAS(shared_col + hash, -1, bcol);
                if(old == -1 || old == bcol){
                    atomicAdd(shared_val + hash, aval * bval);
                    break;
                }
                else{
                    hash = (hash + 1) < tsize ? hash + 1 : 0;
                }
#endif
#ifdef HASH_MULTI
                if (shared_col[hash] == bcol) {
                    atomicAdd(shared_val + hash, aval * bval);
                    break;
                }
                else if (shared_col[hash] == -1) {
                    old = atomicCAS(shared_col + hash, -1, bcol);
                    if (old == -1) {
                        atomicAdd(shared_val + hash, aval * bval);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) < tsize ? hash + 1 : 0;
                }
#endif
            }
        }
    }

    __syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash hash %lld\n", t0);
    //}
    //__syncthreads();
    //t0 = clock64();

    // condense shared hash table
    mint offset;
    bool valid;
    #pragma unroll
    for (j = 0; j < SH_ROW; j += BS){
        offset = j + threadIdx.x;
        valid = offset < tsize;
        if(valid){
            acol = shared_col[offset];
            aval = shared_val[offset];
            if(acol != -1){
                offset = atomicAdd(shared_offset, 1);
            }
        }
        __syncthreads();
        if(valid && acol != -1){
            shared_col[offset] = acol;
            shared_val[offset] = aval;
        }
    }
    
    // count sort the result
    __syncthreads();
    mint count, target;
    for (j = threadIdx.x; j < row_nnz; j += blockDim.x) {
        target = shared_col[j];
        count = 0;
        for (k = 0; k < row_nnz; k++) {
            count += (unsigned int)(shared_col[k] - target) >> 31;
        }
        d_ccol[c_offset + count] = shared_col[j];
        d_cval[c_offset + count] = shared_val[j];
    }
    //__syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash sort %lld\n", t0);
    //}

}


// half occu due to max shared memory
__global__ void __launch_bounds__(1024, 1) k_numeric_max_shared_hash_tb_half_occu(
    const mint * __restrict__ d_arpt, const mint * __restrict__ d_acol, 
    const mdouble * __restrict__ d_aval,
    const mint * __restrict__ d_brpt, const mint * __restrict__ d_bcol, 
    const mdouble * __restrict__ d_bval,
    mint *d_bins,
    mint *d_crpt, mint *d_ccol, mdouble* d_cval){

    //long long t0 = clock64();

    mint tid = threadIdx.x & (WSIZE - 1);
    mint wid = threadIdx.x / WSIZE;
    mint wnum = blockDim.x / WSIZE;
    mint j, k;
    extern __shared__ mint shared_mem[];
    const mint tsize = 8191;
    mint *shared_col = shared_mem;
    mint *shared_offset = shared_mem + tsize;
    mdouble* shared_val = (mdouble*)(shared_mem + (tsize + 1));

    for(j = threadIdx.x; j < tsize; j += blockDim.x){
        shared_col[j] = -1;
        shared_val[j] = 0;
    }
    if(threadIdx.x == 0){
        shared_offset[0] = 0;
    }
    __syncthreads();

    mint acol, bcol, hash, old;
    mdouble aval, bval;
    mint rid = d_bins[blockIdx.x];
    mint c_offset = d_crpt[rid];
    mint row_nnz = d_crpt[rid + 1] - d_crpt[rid];

    for(j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum){
        acol = d_acol[j];
        aval = d_aval[j];
        for(k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k+= WSIZE){
            bcol = d_bcol[k];
            bval = d_bval[k];
            hash = (bcol * HASH_SCALE) % tsize;
            while(1){
#ifdef HASH_SINGLE
                old = atomicCAS(shared_col + hash, -1, bcol);
                if(old == -1 || old == bcol){
                    atomicAdd(shared_val + hash, aval * bval);
                    break;
                }
                else{
                    hash = (hash + 1) < tsize ? hash + 1 : 0;
                }
#endif
#ifdef HASH_MULTI
                if (shared_col[hash] == bcol) {
                    atomicAdd(shared_val + hash, aval * bval);
                    break;
                }
                else if (shared_col[hash] == -1) {
                    old = atomicCAS(shared_col + hash, -1, bcol);
                    if (old == -1) {
                        atomicAdd(shared_val + hash, aval * bval);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) < tsize ? hash + 1 : 0;
                }
#endif
            }
        }
    }
    __syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash hash %lld\n", t0);
    //}
    //__syncthreads();
    //t0 = clock64();

    // condense shared hash table
    mint offset;
    bool valid;
    #pragma unroll
    for (j = 0; j < 8192; j += 1024){
        offset = j + threadIdx.x;
        valid = offset < tsize;
        if(valid){
            acol = shared_col[offset];
            aval = shared_val[offset];
            if(acol != -1){
                offset = atomicAdd(shared_offset, 1);
            }
        }
        __syncthreads();
        if(valid && acol != -1){
            shared_col[offset] = acol;
            shared_val[offset] = aval;
        }
    }

    // count sort the result
    __syncthreads();
    mint count, target;
    for (j = threadIdx.x; j < row_nnz; j += blockDim.x) {
        target = shared_col[j];
        count = 0;
        for (k = 0; k < row_nnz; k++) {
            count += (unsigned int)(shared_col[k] - target) >> 31;
        }
        d_ccol[c_offset + count] = shared_col[j];
        d_cval[c_offset + count] = shared_val[j];
    }
    //__syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash sort %lld\n", t0);
    //}
}


__global__ void __launch_bounds__(1024, 2) k_numeric_global_hash_tb_full_occu(
    const mint * __restrict__ d_arpt, const mint * __restrict__ d_acol, 
    const mdouble * __restrict__ d_aval,
    const mint * __restrict__ d_brpt, const mint * __restrict__ d_bcol, 
    const mdouble * __restrict__ d_bval,
    mint *d_bins, mint max_tsize, mint* d_tables,
    mint *d_crpt, mint *d_ccol, mdouble* d_cval){

    //long long t0 = clock64();

    mint tid = threadIdx.x & (WSIZE - 1);
    mint wid = threadIdx.x / WSIZE;
    mint wnum = blockDim.x / WSIZE;
    mint j, k;
    __shared__ mint shared_offset[1];
    
    mint* table_col = d_tables + blockIdx.x * max_tsize * ((sizeof(mint) + sizeof(mdouble))/sizeof(mint));
    mdouble* table_val = (mdouble*)(table_col + max_tsize);
    mint rid = d_bins[blockIdx.x];
    mint c_offset = d_crpt[rid];
    mint row_nnz = d_crpt[rid + 1] - c_offset;
    mint tsize = row_nnz * NUMERIC_SCALE_LARGE;
    for(j = threadIdx.x; j < tsize; j += blockDim.x){
        table_col[j] = -1;
        table_val[j] = 0;
    }
    if(threadIdx.x == 0){
        shared_offset[0] = 0;
    }
    __syncthreads();

    mint acol, bcol, hash, old;
    mdouble aval, bval;
    for(j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum){
        acol = d_acol[j];
        aval = d_aval[j];
        for(k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k+= WSIZE){
            bcol = d_bcol[k];
            bval = d_bval[k];
            hash = (bcol * HASH_SCALE) % tsize;
            while(1){
#ifdef HASH_SINGLE
                old = atomicCAS(table_col + hash, -1, bcol);
                if(old == -1 || old == bcol){
                    atomicAdd(table_val + hash, aval * bval);
                    break;
                }
                else{
                    hash = (hash + 1) < tsize ? hash + 1 : 0;
                }
#endif
#ifdef HASH_MULTI
                if (table_col[hash] == bcol) {
                    atomicAdd(table_val + hash, aval * bval);
                    break;
                }
                else if (table_col[hash] == -1) {
                    old = atomicCAS(table_col + hash, -1, bcol);
                    if (old == -1) {
                        atomicAdd(table_val + hash, aval * bval);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) < tsize ? hash + 1 : 0;
                }

#endif
            }
        }
    }

    //__syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash global hash %lld\n", t0);
    //}
    //__syncthreads();
    //t0 = clock64();

    // condense shared hash table
    __syncthreads();
    mint offset;
    for (j = threadIdx.x; j < tsize; j += blockDim.x){
        acol = table_col[j];
        aval = table_val[j];
        if(acol != -1){
            offset = atomicAdd(shared_offset, 1);
            d_ccol[c_offset + offset] = acol;
            d_cval[c_offset + offset] = aval;
        }
    }
    __syncthreads();
    for(j = threadIdx.x; j < row_nnz; j += blockDim.x){
        table_col[j] = d_ccol[c_offset + j];
        table_val[j] = d_cval[c_offset + j];
    }

    //__syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash global  condense %lld\n", t0);
    //}
    //__syncthreads();
    //t0 = clock64();

    // count sort the result
    __syncthreads();
    mint count, target;
    for (j = threadIdx.x; j < row_nnz; j += blockDim.x) {
        target = table_col[j];
        count = 0;
        for (k = 0; k < row_nnz; k++) {
            count += (unsigned int)(table_col[k] - target) >> 31;
        }
        d_ccol[c_offset + count] = table_col[j];
        d_cval[c_offset + count] = table_val[j];
    }
    //__syncthreads();
    //t0 = clock64() - t0;
    //if(threadIdx.x == 0){
    //    printf("inside k_numeric_hash sort %lld\n", t0);
    //}
}


#endif
