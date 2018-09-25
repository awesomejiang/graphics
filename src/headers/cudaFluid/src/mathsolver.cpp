#include "mathsolver.h"


__GLOBAL__ void divergence(Indexing *indexing, float *out, vec2 *in){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;
	vec2 const &left = in[indexing->getLeft(idx)];
	vec2 const &right = in[indexing->getRight(idx)];
	vec2 const &top = in[indexing->getTop(idx)];
	vec2 const &bottom = in[indexing->getBottom(idx)];

	out[idx] = (right[0] - left[0] + top[1] - bottom[1])/2.0f;
}


__GLOBAL__ void poissonStep(Indexing *indexing, float *out, float *in, float *rightHand){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;
	float const &left = in[indexing->getLeft(idx)];
	float const &right = in[indexing->getRight(idx)];
	float const &top = in[indexing->getTop(idx)];
	float const &bottom = in[indexing->getBottom(idx)];

	out[idx] = (left + right + top + bottom - rightHand[idx])/4.0f;
}

__GLOBAL__ void subGrad(Indexing *indexing, vec2 *res, float *source){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;
	float const &left = source[indexing->getLeft(idx)];
	float const &right = source[indexing->getRight(idx)];
	float const &top = source[indexing->getTop(idx)];
	float const &bottom = source[indexing->getBottom(idx)];

	res[idx] -= vec2{right-left, top-bottom}/2.0f;
}