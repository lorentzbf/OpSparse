#ifndef _Z_COMMON_
#define _Z_COMMON_

#include <cuda_runtime.h>
#include <stdio.h>
#include <exception>
#include <cusparse.h>
#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <algorithm>

#define likely(x) __builtin_expect(x,1)
#define unlikely(x) __builtin_expect(x,0)

//typedef unsigned int mint;
typedef int mint;
typedef double mdouble;

inline static void checkCUDA(cudaError_t err,
							   const char *file,
							   int line)
{
	if (unlikely(err != cudaSuccess))
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			   file, line);
		throw std::exception();
	}
}
// #ifdef _DEBUG || NDEBUG || DEBUG
#define CHECK_CUDA(err) (checkCUDA(err, __FILE__, __LINE__))
#define CHECK_ERROR(err) (checkCUDA(err, __FILE__, __LINE__))

inline void CHECK_CUSPARSE(cusparseStatus_t status, std::string errorMsg="")
{
	if (status != CUSPARSE_STATUS_SUCCESS) {
		std::cout << "CuSparse error: " << errorMsg << std::endl;
		throw std::exception();
	}
}

#define HP_TIMING_NOW(Var) \
  ({ unsigned int _hi, _lo; \
     asm volatile ("lfence\n\trdtsc" : "=a" (_lo), "=d" (_hi)); \
     (Var) = ((unsigned long long int) _hi << 32) | _lo; })

/* precision is 1 clock cycle.
 * execute time is roughly 50 or 140 cycles depends on cpu family */
inline void cpuid(int *info, int eax, int ecx = 0){
    int ax, bx, cx, dx;
    __asm__ __volatile__ ("cpuid": "=a" (ax), "=b" (bx), "=c" (cx), "=d" (dx) : "a" (eax));

    info[0] = ax;
    info[1] = bx;
    info[2] = cx;
    info[3] = dx;
}

inline long get_tsc_freq(){
    static long freq = 0;
    if(unlikely((freq == 0))){
        int raw[4];
        cpuid(raw, 0x16); // get cpu freq
        freq = long(raw[0]) * 1000000;
        //printf("static first call %f\n", freq);
    }
    return freq;
}

inline double fast_clock_time(){
    long counter;
    HP_TIMING_NOW(counter);
    return double(counter)/get_tsc_freq();
}

template <typename T>
inline void D2H(T *dst, T* src, size_t size){
    CHECK_ERROR(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

template <typename T>
inline void H2D(T *dst, T* src, size_t size){
    CHECK_ERROR(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

template <typename T>
inline void D2D(T *dst, T* src, size_t size){
    CHECK_ERROR(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}


#endif
