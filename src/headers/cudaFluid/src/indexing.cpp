#include "indexing.h"

__DEVICE__ Indexing::Indexing(int const &w, int const &h): w{w}, h{h} {}

__DEVICE__ int Indexing::getIdx(){
	auto x = blockDim.x*blockIdx.x + threadIdx.x;
	auto y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x < w && y < h)
		return y*w + x;
	else
		return -1;
}

__DEVICE__ int Indexing::getLeft(int const &idx){
	auto x = idx%w;
	auto y = idx/w;
	if(x < w && y < h){
		x = (x==0)? x: x-1;
		return y*w + x;
	}
	else
		return -1;
}

__DEVICE__ int Indexing::getRight(int const &idx){
	auto x = idx%w;
	auto y = idx/w;
	if(x < w && y < h){
		x = (x==w-1)? x: x+1;
		return y*w + x;
	}
	else
		return -1;
}

__DEVICE__ int Indexing::getTop(int const &idx){
	auto x = idx%w;
	auto y = idx/w;
	if(x < w && y < h){
		y = (y==h-1)? y: y+1;
		return y*w + x;
	}
	else
		return -1;
}

__DEVICE__ int Indexing::getBottom(int const &idx){
	auto x = idx%w;
	auto y = idx/w;
	if(x < w && y < h){
		y = (y==0)? y: y-1;
		return y*w + x;
	}
	else
		return -1;
}

__DEVICE__ int Indexing::getTrace(vec2 const &v){
	float fX = static_cast<float>(blockDim.x*blockIdx.x + threadIdx.x) + v[0];
	float fY = static_cast<float>(blockDim.y*blockIdx.y + threadIdx.y) + v[1];

	//clamp to edges
	int x = fX<0.0f? 0: (fX>w-1? w-1: fX);
	int y = fY<0.0f? 0: (fY>h-1? h-1: fY);

	return y*w + x;
}

__DEVICE__ bool Indexing::isLeftBoundary(int const &idx){
	return idx%w == 0;
}

__DEVICE__ bool Indexing::isRightBoundary(int const &idx){
	return idx%w == w-1;
}
__DEVICE__ bool Indexing::isTopBoundary(int const &idx){
	return idx/w == h-1;
}

__DEVICE__ bool Indexing::isBottomBoundary(int const &idx){
	return idx/w == 0;
}