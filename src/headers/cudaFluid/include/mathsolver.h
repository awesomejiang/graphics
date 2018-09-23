#ifndef MATHSOLVER_H
#define MATHSOLVER_H

#include "fluidcommon.h"

__GLOBAL__ void diffusionStep(Indexing *indexing, vec2 *currV, vec2 *tempV, vec2 *oldV, float dt, float niu);
__GLOBAL__ void colorDiffusionStep(Indexing *indexing, vec3 *currC, vec3 *tempC, vec3 *oldC, float dt, float niu);
__GLOBAL__ void advect(Indexing *indexing, vec2 *currV, vec2 *oldV, float dt);
__GLOBAL__ void colorAdvect(Indexing *indexing, vec3 *currC, vec3 *oldC, vec2 *oldV, float dt);
__GLOBAL__ void divergence(Indexing *indexing, float *div, vec2 *currV);
__GLOBAL__ void pressureStep(Indexing *indexing, float *currP, float *oldP, float *div);
__GLOBAL__ void correction(Indexing *indexing, vec2 *currV, float *currP);


#endif
