#include <iostream>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <nsparse.hpp>
#include <nsparse_asm.hpp>
#include <CSR.hpp>
#include "cuda_common.h"

#ifndef SPGEMM_H
#define SPGEMM_H

template <class idType>
__global__ void set_flop_per_row(idType *d_arpt, idType *d_acol, const idType* __restrict__ d_brpt, long long int *d_flop_per_row, idType nrow)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrow) {
        return;
    }
    idType flop_per_row = 0;
    idType j;
    for (j = d_arpt[i]; j < d_arpt[i + 1]; j++) {
        flop_per_row += d_brpt[d_acol[j] + 1] - d_brpt[d_acol[j]];
    }
    d_flop_per_row[i] = flop_per_row;
}

template <class idType, class valType>
void get_spgemm_flop(CSR<idType, valType> a, CSR<idType, valType> b, long long int &flop)
{
    int GS, BS;
    long long int *d_flop_per_row;

    BS = MAX_LOCAL_THREAD_NUM;
    checkCudaErrors(cudaMalloc((void **)&(d_flop_per_row), sizeof(long long int) * (1 + a.nrow)));
  
    GS = div_round_up(a.nrow, BS);
    set_flop_per_row<<<GS, BS>>>(a.d_rpt, a.d_colids, b.d_rpt, d_flop_per_row, a.nrow);
  
    long long int *tmp = (long long int *)malloc(sizeof(long long int) * a.nrow);
    cudaMemcpy(tmp, d_flop_per_row, sizeof(long long int) * a.nrow, cudaMemcpyDeviceToHost);
    flop = thrust::reduce(thrust::device, d_flop_per_row, d_flop_per_row + a.nrow);

    flop *= 2;
    cudaFree(d_flop_per_row);

}

//template <class idType, class valType>
//cusparseStatus_t SpGEMM_cuSPARSE_numeric(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c, cusparseHandle_t cusparseHandle, cusparseOperation_t trans_a, cusparseOperation_t trans_b, cusparseMatDescr_t descr_a, cusparseMatDescr_t descr_b, cusparseMatDescr_t descr_c);
//
//template <>
//cusparseStatus_t SpGEMM_cuSPARSE_numeric<int, float>(CSR<int, float> a, CSR<int, float> b, CSR<int, float> &c, cusparseHandle_t cusparseHandle, cusparseOperation_t trans_a, cusparseOperation_t trans_b, cusparseMatDescr_t descr_a, cusparseMatDescr_t descr_b, cusparseMatDescr_t descr_c)
//{
//    return cusparseScsrgemm(cusparseHandle, trans_a, trans_b, a.nrow, b.ncolumn, a.ncolumn, descr_a, a.nnz, a.d_values, a.d_rpt, a.d_colids, descr_b, b.nnz, b.d_values, b.d_rpt, b.d_colids, descr_c, c.d_values, c.d_rpt, c.d_colids);
//}
//
//template <>
//cusparseStatus_t SpGEMM_cuSPARSE_numeric<int, double>(CSR<int, double> a, CSR<int, double> b, CSR<int, double> &c, cusparseHandle_t cusparseHandle, cusparseOperation_t trans_a, cusparseOperation_t trans_b, cusparseMatDescr_t descr_a, cusparseMatDescr_t descr_b, cusparseMatDescr_t descr_c)
//{
//    return cusparseDcsrgemm(cusparseHandle, trans_a, trans_b, a.nrow, b.ncolumn, a.ncolumn, descr_a, a.nnz, a.d_values, a.d_rpt, a.d_colids, descr_b, b.nnz, b.d_values, b.d_rpt, b.d_colids, descr_c, c.d_values, c.d_rpt, c.d_colids);
//}

//template <class idType, class valType>
//void SpGEMM_cuSPARSE_kernel(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c, cusparseHandle_t cusparseHandle, cusparseOperation_t trans_a, cusparseOperation_t trans_b, cusparseMatDescr_t descr_a, cusparseMatDescr_t descr_b, cusparseMatDescr_t descr_c)
//{
//    cusparseStatus_t status;
//    c.nrow = a.nrow;
//    c.ncolumn = b.ncolumn;
//    c.device_malloc = true;
//    cudaMalloc((void **)&(c.d_rpt), sizeof(idType) * (c.nrow + 1));
//
//    status = cusparseXcsrgemmNnz(cusparseHandle, trans_a, trans_b, a.nrow, b.ncolumn, a.ncolumn, descr_a, a.nnz, a.d_rpt, a.d_colids, descr_b, b.nnz, b.d_rpt, b.d_colids, descr_c, c.d_rpt, &(c.nnz));
//    if (status != CUSPARSE_STATUS_SUCCESS) {
//        cout << "cuSPARSE failed at Symbolic phase" << endl;
//    }
//
//    cudaMalloc((void **)&(c.d_colids), sizeof(idType) * (c.nnz));
//    cudaMalloc((void **)&(c.d_values), sizeof(valType) * (c.nnz));
//        
//    status = SpGEMM_cuSPARSE_numeric(a, b, c, cusparseHandle, trans_a, trans_b, descr_a, descr_b, descr_c);
//    
//    if (status != CUSPARSE_STATUS_SUCCESS) {
//        cout << "cuSPARSE failed at Numeric phase" << endl;
//    }
//}

//template <class idType, class valType>
//void SpGEMM_cuSPARSE(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c)
//{
//    cusparseHandle_t cusparseHandle;
//    cusparseMatDescr_t descr_a, descr_b, descr_c;
//    cusparseOperation_t trans_a, trans_b;
//
//    trans_a = trans_b = CUSPARSE_OPERATION_NON_TRANSPOSE;
//  
//    /* Set up cuSPARSE Library */
//    cusparseCreate(&cusparseHandle);
//    cusparseCreateMatDescr(&descr_a);
//    cusparseCreateMatDescr(&descr_b);
//    cusparseCreateMatDescr(&descr_c);
//    cusparseSetMatType(descr_a, CUSPARSE_MATRIX_TYPE_GENERAL);
//    cusparseSetMatType(descr_b, CUSPARSE_MATRIX_TYPE_GENERAL);
//    cusparseSetMatType(descr_c, CUSPARSE_MATRIX_TYPE_GENERAL);
//    cusparseSetMatIndexBase(descr_a, CUSPARSE_INDEX_BASE_ZERO);
//    cusparseSetMatIndexBase(descr_b, CUSPARSE_INDEX_BASE_ZERO);
//    cusparseSetMatIndexBase(descr_c, CUSPARSE_INDEX_BASE_ZERO);
//  
//    /* Execution of SpMV on Device */
//    SpGEMM_cuSPARSE_kernel(a, b, c,
//                           cusparseHandle,
//                           trans_a, trans_b,
//                           descr_a, descr_b, descr_c);
//    cudaDeviceSynchronize();
//    
//    c.memcpyDtH();
//
//    c.release_csr();
//    cusparseDestroy(cusparseHandle);
//}



void cusparse_spgemm_inner(int *d_row_ptr_A, int *d_col_idx_A, double *d_csr_values_A,
                       int *d_row_ptr_B, int *d_col_idx_B, double *d_csr_values_B,
                       int **d_row_ptr_C, int **d_col_idx_C, double **d_csr_values_C,
                       int M, int K, int N, int nnz_A, int nnz_B, int* nnz_C){
    CHECK_CUDA(cudaMalloc((void**) d_row_ptr_C, (M+1) * sizeof(int)));
    
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle), "create cusparse handle");
    cusparseSpMatDescr_t matA, matB, matC;
    void *dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, M, K, nnz_A,
                                      d_row_ptr_A, d_col_idx_A, d_csr_values_A,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "create matA" );
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, K, N, nnz_B,
                                      d_row_ptr_B, d_col_idx_B, d_csr_values_B,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "create matB" );
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, M, N, 0,
                                      NULL, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "create matC" );
    cusparseSpGEMMDescr_t spgemmDescr;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDescr), "create spgemm descr");
    double               alpha       = 1.0f;
    double               beta        = 0.0f;
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_64F;

    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB,
                                                 &alpha, matA, matB, &beta, matC,
                                                 computeType, CUSPARSE_SPGEMM_DEFAULT,
                                                 spgemmDescr, &bufferSize1, NULL), 
                                                 "first work estimation");
    CHECK_CUDA(cudaMalloc((void**) &dBuffer1, bufferSize1));
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB,
                                                 &alpha, matA, matB, &beta, matC,
                                                 computeType, CUSPARSE_SPGEMM_DEFAULT,
                                                 spgemmDescr, &bufferSize1, dBuffer1), 
                                                 "second work estimation");
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB,
                                          &alpha, matA, matB, &beta, matC,
                                          computeType, CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDescr, &bufferSize2, NULL), 
                                          "first compute");

    CHECK_CUDA(cudaMalloc((void**) &dBuffer2, bufferSize2));
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB,
                                          &alpha, matA, matB, &beta, matC,
                                          computeType, CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDescr, &bufferSize2, dBuffer2), 
                                          "second compute");

    int64_t M_C, N_C, nnz_C_64I;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &M_C, &N_C, &nnz_C_64I) );
    *nnz_C = nnz_C_64I;
    CHECK_CUDA(cudaMalloc((void**)d_col_idx_C, *nnz_C*sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)d_csr_values_C, *nnz_C*sizeof(double)));
    CHECK_CUSPARSE(cusparseCsrSetPointers(matC, *d_row_ptr_C, *d_col_idx_C, *d_csr_values_C));
    
    CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, opA, opB,
                                       &alpha, matA, matB, &beta, matC,
                                       computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr),
                                       "spgemm copy");
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDescr) );
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) );
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) );
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) );
    CHECK_CUSPARSE( cusparseDestroy(handle) );

    CHECK_CUDA(cudaFree(dBuffer1));
    CHECK_CUDA(cudaFree(dBuffer2));

    CHECK_CUDA(cudaDeviceSynchronize());
}

template <class idType, class valType>
void SpGEMM_cuSPARSE(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c){
    int tmp_nnz;
	cusparse_spgemm_inner(a.d_rpt, a.d_colids, a.d_values,
							b.d_rpt, b.d_colids, b.d_values,
							&(c.d_rpt), &(c.d_colids), &(c.d_values),
							a.nrow, a.ncolumn, b.ncolumn, a.nnz, b.nnz, &(tmp_nnz));
	c.nrow = a.nrow;
	c.ncolumn = b.ncolumn;
	c.nnz = tmp_nnz;
}

#endif

