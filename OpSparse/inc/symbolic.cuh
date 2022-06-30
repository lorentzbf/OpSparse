#ifndef __Z_SYMBOLIC_CUH__
#define __Z_SYMBOLIC_CUH__

#include "cuda_common.h"
#include "define.h"

// good pwarp
__global__ void __launch_bounds__(PWARP_BLOCK_SIZE, 2) k_symbolic_shared_hash_pwarp(
    const mint * __restrict__ d_arpt, const mint * __restrict__ d_acol, 
    const mint * __restrict__ d_brpt, const mint * __restrict__ d_bcol,
    mint * __restrict__ d_bins, 
    mint bin_size,
    mint * __restrict__ d_row_nnz){

    mint i = threadIdx.x + blockIdx.x * blockDim.x;
    mint tid = threadIdx.x & (PWARP - 1);
    mint rid = i / PWARP;
    mint block_rid = rid & (PWARP_ROWS - 1);

    __shared__ mint shared_mem[PWARP_ROWS * PWARP_TSIZE + PWARP_ROWS];
    mint *shared_table = shared_mem;
    mint *shared_nnz = shared_mem + PWARP_ROWS * PWARP_TSIZE;
    mint j, k;
    for(j = threadIdx.x; j < PWARP_ROWS * PWARP_TSIZE; j += blockDim.x){
        shared_table[j] = -1;
    }
    if(threadIdx.x < PWARP_ROWS){
        shared_nnz[threadIdx.x] = 0;
    }
    if(rid >= bin_size){
        return;
    }
    __syncthreads();
    mint *table = shared_table + block_rid * PWARP_TSIZE;

    rid = d_bins[rid];
    mint acol, bcol;
    mint hash, old;
    for(j = d_arpt[rid] + tid; j < d_arpt[rid + 1]; j += PWARP){ // pwarp per row, thread per a item, thread per b row
        acol = d_acol[j];
        for(k = d_brpt[acol]; k < d_brpt[acol + 1]; k++){ // thread per b row
            bcol = d_bcol[k];
            hash = (bcol * HASH_SCALE) & (PWARP_TSIZE - 1);
            while(1){
#ifdef HASH_SINGLE
                old = atomicCAS(table + hash, -1, bcol);
                if(old == -1){
                    atomicAdd(shared_nnz + block_rid, 1);
                    break;
                }
                else if(old == bcol){
                    break;
                }
                else{
                    hash = (hash + 1) & (PWARP_TSIZE - 1);
                }
#endif
#ifdef HASH_MULTI
                if(table[hash] == bcol){
                    break;
                }
                else if (table[hash] == -1){
                    old = atomicCAS(table + hash, -1, bcol);
                    if(old == -1){
                        atomicAdd(shared_nnz + block_rid, 1);
                        break;
                    }
                }
                else{
                    hash = (hash + 1) &(PWARP_TSIZE - 1);
                }
#endif
            }
        }
    }
    __syncthreads();
    if(tid == 0){
        d_row_nnz[rid] = shared_nnz[block_rid];
    }
}

template <int SH_ROW>
__global__ void __launch_bounds__(1024, 2) k_symbolic_shared_hash_tb(
    const mint * __restrict__ d_arpt, const mint * __restrict__ d_acol, 
    const mint * __restrict__ d_brpt, const mint * __restrict__ d_bcol,
    mint * __restrict__ d_bins,
    mint * __restrict__ d_row_nnz){

    //long long t0 = clock64();

    mint tid = threadIdx.x & (WSIZE - 1);
    mint wid = threadIdx.x / WSIZE;
    mint wnum = blockDim.x / WSIZE;
    mint j, k;
    __shared__ mint shared_table[SH_ROW];
    __shared__ mint shared_nnz[1];

    for(j = threadIdx.x; j < SH_ROW; j += blockDim.x){
        shared_table[j] = -1;
    }
    if(threadIdx.x == 0){
        shared_nnz[0] = 0;
        //shared_nnz[0] = threadIdx.x  + *d_fail_bin_size;
    }
    __syncthreads();
    mint acol, bcol, hash, old;
    mint rid = d_bins[blockIdx.x];
    for(j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum){
        acol = d_acol[j];
        for(k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k+= WSIZE){
            bcol = d_bcol[k];
            hash = (bcol * HASH_SCALE) & (SH_ROW - 1);
            while(1){
#ifdef HASH_SINGLE
                old = atomicCAS(shared_table + hash, -1, bcol);
                if(old == -1){
                    atomicAdd(shared_nnz, 1);
                    break;
                }
                else if(old == bcol){
                    break;
                }
                else{
                    hash = (hash + 1) & (SH_ROW - 1);
                }
#endif
#ifdef HASH_MULTI
                if(shared_table[hash] == bcol){
                    break;
                }
                else if (shared_table[hash] == -1){
                    old = atomicCAS(shared_table + hash, -1, bcol);
                    if(old == -1){
                        atomicAdd(shared_nnz, 1);
                        break;
                    }
                }
                else{
                    hash = (hash + 1) &(SH_ROW - 1);
                }
#endif
            }
        }
    }
    __syncthreads();

    if(threadIdx.x == 0){
        d_row_nnz[rid] = shared_nnz[0];
    }

}

__global__ void __launch_bounds__(1024,2) k_symbolic_large_shared_hash_tb(
    const mint * __restrict__ d_arpt, const mint * __restrict__ d_acol, 
    const mint * __restrict__ d_brpt, const mint * __restrict__ d_bcol,
    mint * __restrict__ d_bins, 
    mint * __restrict__ d_row_nnz){

    mint tid = threadIdx.x & (WSIZE - 1);
    mint wid = threadIdx.x / WSIZE;
    mint wnum = blockDim.x / WSIZE;
    mint j, k;
    __shared__ mint shared_mem[12288];
    const mint tsize = 12287;
    mint* shared_table = shared_mem;
    mint* shared_nnz = shared_mem + tsize;
    
    for(j = threadIdx.x; j < tsize; j += blockDim.x){
        shared_table[j] = -1;
    }
    if(threadIdx.x == 0){
        shared_nnz[0] = 0;
    }
    __syncthreads();
    
    mint rid = d_bins[blockIdx.x];
    mint acol, bcol, hash, old;
    for(j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum){
        acol = d_acol[j];
        for(k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k+= WSIZE){
            bcol = d_bcol[k];
            hash = (bcol * HASH_SCALE) % tsize;
            while(1){
#ifdef HASH_SINGLE
                old = atomicCAS(shared_table + hash, -1, bcol);
                if(old == bcol){
                    break;
                }
                else if(old == -1){
                    atomicAdd(shared_nnz, 1);
                    break;
                }
                else{
                    hash = hash + 1 < tsize ? hash + 1 : 0;
                }
#endif
#ifdef HASH_MULTI
                if(shared_table[hash] == bcol){
                    break;
                }
                else if (shared_table[hash] == -1){
                    old = atomicCAS(shared_table + hash, -1, bcol);
                    if(old == -1){
                        atomicAdd(shared_nnz, 1);
                        break;
                    }
                }
                else{
                    hash = hash + 1 < tsize ? hash + 1 : 0;
                }

#endif
            }
        }
    }
    __syncthreads();
    
    if(threadIdx.x == 0){
        d_row_nnz[rid] = shared_nnz[0];
    }
}

__global__ void __launch_bounds__(1024,1) k_symbolic_max_shared_hash_tb_with_fail(
    const mint * __restrict__ d_arpt, const mint * __restrict__ d_acol, 
    const mint * __restrict__ d_brpt, const mint * __restrict__ d_bcol,
    mint * __restrict__ d_bins, 
    mint * __restrict__ d_fail_bins,
    mint * __restrict__ d_fail_bin_size,
    mint * __restrict__ d_row_nnz){

    mint tid = threadIdx.x & (WSIZE - 1);
    mint wid = threadIdx.x / WSIZE;
    mint wnum = blockDim.x / WSIZE;
    mint j, k;
    extern __shared__ mint shared_mem[]; // size 24576
    const mint tsize = 24575;
    mint* shared_table = shared_mem;
    mint* shared_nnz = shared_mem + tsize;
    
    mint thresh_nnz = tsize * THRESH_SCALE;
    for(j = threadIdx.x; j < tsize; j += blockDim.x){
        shared_table[j] = -1;
    }
    if(threadIdx.x == 0){
        shared_nnz[0] = 0;
    }
    __syncthreads();
    
    mint rid = d_bins[blockIdx.x];
    mint acol, bcol, hash, old;
    for(j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum){
        acol = d_acol[j];
        for(k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k+= WSIZE){
            bcol = d_bcol[k];
            hash = (bcol * HASH_SCALE) % tsize;
            while(shared_nnz[0] <= thresh_nnz){
#ifdef HASH_SINGLE
                old = atomicCAS(shared_table + hash, -1, bcol);
                if(old == bcol){
                    break;
                }
                else if(old == -1){
                    atomicAdd(shared_nnz, 1);
                    break;
                }
                else{
                    hash = hash + 1 < tsize ? hash + 1 : 0;
                }
#endif
#ifdef HASH_MULTI
                if(shared_table[hash] == bcol){
                    break;
                }
                else if (shared_table[hash] == -1){
                    old = atomicCAS(shared_table + hash, -1, bcol);
                    if(old == -1){
                        atomicAdd(shared_nnz, 1);
                        break;
                    }
                }
                else{
                    hash = hash + 1 < tsize ? hash + 1 : 0;
                }
#endif
            }
        }
    }
    __syncthreads();

    mint row_nnz;
    mint fail_index;
    if(threadIdx.x == 0){
        row_nnz = shared_nnz[0];
        if(row_nnz <= thresh_nnz){ // success
            d_row_nnz[rid] = row_nnz;
        }
        else{ // fail case
            fail_index = atomicAdd(d_fail_bin_size, 1);
            d_fail_bins[fail_index] = rid;
        }
    }
}

__global__ void __launch_bounds__(1024, 2) k_symbolic_global_hash_tb(
    const mint * __restrict__ d_arpt, const mint * __restrict__ d_acol, 
    const mint * __restrict__ d_brpt, const mint * __restrict__ d_bcol,
    mint * __restrict__ d_bins,
    mint * __restrict__ d_row_nnz, 
    mint * __restrict__ d_table,
    mint max_tsize){ 

    mint tid = threadIdx.x & (WSIZE - 1);
    mint wid = threadIdx.x / WSIZE;
    mint wnum = blockDim.x / WSIZE;
    mint j, k;
    __shared__ mint shared_nnz[1];

    mint rid = d_bins[blockIdx.x];
    mint *table = d_table + blockIdx.x * max_tsize;
    mint tsize = d_row_nnz[rid] * SYMBOLIC_SCALE_LARGE;
    mint acol, bcol, hash, old;
    for(j = threadIdx.x; j < tsize; j += blockDim.x){
        table[j] = -1;
    }
    if(threadIdx.x == 0){
        shared_nnz[0] = 0;
    }
    __syncthreads();
    
    mint nnz = 0;
    for(j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum){
        acol = d_acol[j];
        for(k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k+= WSIZE){
            bcol = d_bcol[k];
            hash = (bcol * HASH_SCALE) % tsize;
            while(1){
#ifdef HASH_SINGLE
                old = atomicCAS(table + hash, -1, bcol);
                if(old == -1){
                    nnz++;
                    break;
                }
                else if(old == bcol){
                    break;
                }
                else{
                    hash = hash + 1 < tsize ? hash + 1 : 0;
                }
#endif
#ifdef HASH_MULTI
                if(table[hash] == bcol){
                    break;
                }
                else if (table[hash] == -1){
                    old = atomicCAS(table + hash, -1, bcol);
                    if(old == -1){
                        nnz++;
                        break;
                    }
                }
                else{
                    hash = hash + 1 < tsize ? hash + 1 : 0;
                }
#endif
            }
        }
    }
    __syncthreads();
    atomicAdd(shared_nnz, nnz);

    __syncthreads();
    if(threadIdx.x == 0){
        d_row_nnz[rid] = shared_nnz[0];
    }
        
}


#endif
