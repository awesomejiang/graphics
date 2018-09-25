#ifndef INDEXING_H
#define INDEXING_H

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


#include "utility.h"

//indexing helper class
class Indexing{
public:
	__DEVICE__ Indexing(int const &w, int const &h);
	__DEVICE__ int getIdx();
	__DEVICE__ int getLeft(int const &idx);
	__DEVICE__ int getRight(int const &idx);
	__DEVICE__ int getTop(int const &idx);
	__DEVICE__ int getBottom(int const &idx);
	__DEVICE__ int getTrace(vec2 const &v);

	__DEVICE__ bool isLeftBoundary(int const &idx);
	__DEVICE__ bool isRightBoundary(int const &idx);
	__DEVICE__ bool isTopBoundary(int const &idx);
	__DEVICE__ bool isBottomBoundary(int const &idx);

	int w, h;
};


#endif
