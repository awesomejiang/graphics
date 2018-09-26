#ifndef MATHSOLVER_H
#define MATHSOLVER_H

#include "indexing.h"

//(1-coef*laplace)q = rightHand
//T should be vec2* or vec3*. coef denotes dt*niu(in fluid simulation case)
template <typename T>
__GLOBAL__ void diffusionStep(Indexing *indexing, T *out, T *temp, T *rightHand, float coef);

//dq/dt = -(vel*div)q
template <typename T>
__GLOBAL__ void advect(Indexing *indexing, T *out, T *in, vec2 *vel, float dt);

//out = div(in)
__GLOBAL__ void divergence(Indexing *indexing, float *out, vec2 *in);

//laplace(q) = rightHand
__GLOBAL__ void poissonStep(Indexing *indexing, float *out, float *in, float *rightHand);

//res -= grad(source)
__GLOBAL__ void subGrad(Indexing *indexing, vec2 *res, float *source);

//dirichlet boundary condition
template <typename T>
__GLOBAL__ void dirichletBC(Indexing *indexing, T *field);

//neumann boundary condition
template <typename T>
__GLOBAL__ void neumannBC(Indexing *indexing, T *field);

//helper function
template <typename T>
__DEVICE__ T interpolate(Indexing *indexing, T *field, vec2 const &displace);


#endif
