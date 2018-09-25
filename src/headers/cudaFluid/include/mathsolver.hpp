#include "mathsolver.h"

template <typename T>
__GLOBAL__ void diffusionStep(Indexing *indexing, T* out, T* temp, T* rightHand, float coef){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;
	T const &left = temp[indexing->getLeft(idx)];
	T const &right = temp[indexing->getRight(idx)];
	T const &top = temp[indexing->getTop(idx)];
	T const &bottom = temp[indexing->getBottom(idx)];

	out[idx] = (rightHand[idx] + coef*(left+right+top+bottom))/(1+4*coef);
}

template <typename T>
__GLOBAL__ void advect(Indexing *indexing, T* out, T* in, vec2 *vel, float dt){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;
	//backtrace method
	out[idx] = interpolate(indexing, in, -dt*vel[idx]);
}

template <typename T>
__DEVICE__ T interpolate(Indexing *indexing, T *field, vec2 const &displace){
	//interpolate with 4 neighbors
	//lower left
	auto idx00 = indexing->getTrace(displace);
	auto f00 = field[idx00];
	//lower right
	auto idx10 = indexing->getRight(idx00);
	auto f10 = field[idx10];
	//upper left
	auto idx01 = indexing->getTop(idx00);
	auto f01 = field[idx01];
	//upper right
	auto idx11 = indexing->getRight(idx01);
	auto f11 = field[idx11];

	float x = displace[0] - static_cast<int>(displace[0]);
	x = x<0.0f? x+1.0f: x;
	float y = displace[1] - static_cast<int>(displace[1]);
	y = y<0.0f? y+1.0f: y;

	return (1.0f-x)*(1.0f-y)*f00 + x*(1.0f-y)*f10 + (1.0f-x)*y*f01 + x*y*f11;
}