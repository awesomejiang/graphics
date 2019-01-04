#ifndef UTILITY_H
#define UTILITY_H

#include <cuda.h>
#include <cstdio>

#include "macros.h"
#include "window.h"
#include "vec_float.h"

// cuda related helper functions/macros
#if __CUDACC__

void __cudaSafeCall(cudaError error, const char *file, const int line);

void __cudaErrorChecker(const char *file, const int line);

#define MAX_THREAD 256
#define MAX_BLOCK_X 65535ll
#define MAX_BLOCK_Y 65535ll
#define MAX_BLOCK_Z 65535ll

__DEVICE__ inline int getIdx(){
	int grid = gridDim.x*gridDim.y*blockIdx.z + gridDim.x*blockIdx.y + blockIdx.x;
	return blockDim.x*grid + threadIdx.x;
}

#endif

//global helpers

#define HALFWIDTH 0.3575f * 0.6f
#define HALFHEIGHT 0.6583f * 0.6f

//get mouse position on scene
struct Mouse{
	vec2 pos;
	vec2 dir;
	bool pressed = false;
	bool firstClicked = false;
};

void getMouse(Mouse &mouse, Window const &window);


//ugly hack for:
//"non-pod" offsetof
#define OFFSETOF(t, e) (void*)(&static_cast<t*>(0)->e)

#endif