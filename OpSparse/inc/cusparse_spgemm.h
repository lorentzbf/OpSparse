#include <cusparse.h>
#include <cuda_runtime.h>
#include "cuda_common.h"
#include "CSR.h"

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


void cusparse_spgemm(CSR *a, CSR *b, CSR *c){
	int tmp_nnz;
	cusparse_spgemm_inner(a->d_rpt, a->d_col, a->d_val,
							b->d_rpt, b->d_col, b->d_val,
							&(c->d_rpt), &(c->d_col), &(c->d_val),
							a->M, a->N, b->N, a->nnz, b->nnz, &(tmp_nnz));
	c->M = a->M;
	c->N = b->N;
	c->nnz = tmp_nnz;
}


void cusparse_spgemm(const CSR& A, const CSR& B, CSR& C){
	int tmp_nnz;
	cusparse_spgemm_inner(A.d_rpt, A.d_col, A.d_val,
	    B.d_rpt, B.d_col, B.d_val,
		&(C.d_rpt), &(C.d_col), &(C.d_val),
		A.M, A.N, B.N, A.nnz, B.nnz, &(tmp_nnz));
	C.M = A.M;
	C.N = B.N;
	C.nnz = tmp_nnz;
}
