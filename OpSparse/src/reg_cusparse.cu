#include "kernel_wrapper.cuh"
#include <fstream>
#include <cuda_profiler_api.h>
#include <cub/cub.cuh>
#include "Timings.h"
#include "cusparse_spgemm.h"


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
    double total_flop_G = double(total_flop) * 2/1000000000;
 
    CSR C;
    double t0 = fast_clock_time(), t1;
    cusparse_spgemm(&A, &B, &C);
    C.release();

    int iter = 10;
    t1 = 0;
    for(int i = 0; i < iter; i++){
        t0 = fast_clock_time();
        cusparse_spgemm(&A, &B, &C);
        t1 += fast_clock_time() - t0;
        //printf("iter %d %le\n", i, fast_clock_time() - t0);
        if(i < iter - 1){
            C.release();
        }
    }
    t1 /= iter;
    //printf("executione time %le, flops %lf\n\n", t1, total_flop_G / t1);
    printf("%s %lf\n", mat1.c_str(), total_flop_G / t1);

    A.release();
    B.release();
    C.release();
    return 0;
}


