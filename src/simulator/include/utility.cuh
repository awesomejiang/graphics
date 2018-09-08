#ifndef UTILITY_CUH
#define UTILITY_CUH

#include <cuda.h>
#include <cstdio>

#include "macros.cuh"
#include "scene.h"
#include "vec.cuh"

// cuda related helper functions/macros
#if __CUDACC__

void __cudaSafeCall(cudaError error, const char *file, const int line);

void __cudaErrorChecker(const char *file, const int line);

__DEVICE__ int getIdx();

#endif

//global helper functions
struct Mouse{
	vec2 pos;
	bool pressed;
};

Mouse getMouse(Scene const &scene);

#endif