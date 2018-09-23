#ifndef FLUIDCOMMON_H
#define FLUIDCOMMON_H

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


#include "utility.h"

//indexing helper class
class Indexing{
public:
	__DEVICE__ Indexing(int const &w, int const &h);
	__DEVICE__ int getIdx();
	__DEVICE__ int getLeft();
	__DEVICE__ int getRight();
	__DEVICE__ int getTop();
	__DEVICE__ int getBottom();
	__DEVICE__ int getBackTrace(vec2 const &v, float const &dt);

	__DEVICE__ bool isLeftBoundary();
	__DEVICE__ bool isRightBoundary();
	__DEVICE__ bool isTopBoundary();
	__DEVICE__ bool isBottomBoundary();

	int w, h;
};


#endif
