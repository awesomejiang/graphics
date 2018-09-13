#ifndef UTILITY_H
#define UTILITY_H

#include <cuda.h>
#include <cstdio>

#include "macros.h"
#include "scene.h"
#include "vec_float.h"

// cuda related helper functions/macros
#if __CUDACC__

void __cudaSafeCall(cudaError error, const char *file, const int line);

void __cudaErrorChecker(const char *file, const int line);

__DEVICE__ int getIdx();

#endif

//global helpers

#define HALFWIDTH 0.3575f * 0.6f
#define HALFHEIGHT 0.6583f * 0.6f

struct Mouse{
	vec2 pos;
	bool pressed = false;
	bool firstClicked = false;
};

void getMouse(Mouse &mouse, Scene const &scene);

#endif