#ifndef MACROS_H
#define MACROS_H

#if __CUDACC__
	#define __DEVICE__ __device__
	#define __HOST__ __host__
	#define __GLOBAL__ __global__
#else
	#define __DEVICE__ 
	#define __HOST__ 
	#define __GLOBAL__
#endif

#define CUDA_SAFE_CALL(err) __cudaSafeCall(err,__FILE__, __LINE__)

#define CUDA_ERROR_CHECKER __cudaErrorChecker(__FILE__, __LINE__)


#endif