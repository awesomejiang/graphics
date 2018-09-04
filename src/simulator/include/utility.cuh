#ifndef UTILITY_H
#define UTILITY_H

#include <cuda.h>
#include <cstdio>

#define CUDA_SAFE_CALL(err) __cudaSafeCall(err,__FILE__, __LINE__)
void __cudaSafeCall(cudaError error, const char *file, const int line);

#define CUDA_ERROR_CHECKER __cudaErrorChecker(__FILE__, __LINE__)
void __cudaErrorChecker(const char *file, const int line);

#endif