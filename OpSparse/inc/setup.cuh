#ifndef __Z_SETUP_CUH__
#define __Z_SETUP_CUH__

#include "cuda_common.h"
#include "define.h"

__global__ void __launch_bounds__(1024, 2) k_compute_flop(
    const mint* __restrict__ d_arpt, 
    const mint* __restrict__ d_acol,
    const mint* __restrict__ d_brpt,
    mint M,
    mint *d_row_flop,
    mint *d_max_row_flop){

    __shared__ mint shared_max_row_flop[1];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) {
        return;
    }
    if(threadIdx.x == 0){
        shared_max_row_flop[0] = 0;
    }
    __syncthreads();
    mint row_flop = 0;
    mint j;
    mint acol;
    mint arow_start, arow_end;
    arow_start = d_arpt[i];
    arow_end = d_arpt[i+1];
    for (j = arow_start; j < arow_end; j++) {
        acol = d_acol[j];
        row_flop += d_brpt[acol + 1] - d_brpt[acol];
    }
    d_row_flop[i] = row_flop;
    atomicMax(shared_max_row_flop, row_flop);
    __syncthreads();
    if(threadIdx.x == 0){
        atomicMax(d_max_row_flop, shared_max_row_flop[0]);
    }
}


#endif
