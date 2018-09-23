#include "fluidcommon.h"

__DEVICE__ Indexing::Indexing(int const &w, int const &h): w{w}, h{h} {}

__DEVICE__ int Indexing::getIdx(){
	auto x = blockDim.x*blockIdx.x + threadIdx.x;
	auto y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x < w && y < h)
		return y*w + x;
	else
		return -1;
}

__DEVICE__ int Indexing::getLeft(){
	auto x = blockDim.x*blockIdx.x + threadIdx.x;
	auto y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x < w && y < h){
		x = (x==0)? x: x-1;
		return y*w + x;
	}
	else
		return -1;
}

__DEVICE__ int Indexing::getRight(){
	auto x = blockDim.x*blockIdx.x + threadIdx.x;
	auto y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x < w && y < h){
		x = (x==w-1)? x: x+1;
		return y*w + x;
	}
	else
		return -1;
}

__DEVICE__ int Indexing::getTop(){
	auto x = blockDim.x*blockIdx.x + threadIdx.x;
	auto y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x < w && y < h){
		y = (y==0)? y: y-1;
		return y*w + x;
	}
	else
		return -1;
}

__DEVICE__ int Indexing::getBottom(){
	auto x = blockDim.x*blockIdx.x + threadIdx.x;
	auto y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x < w && y < h){
		y = (y==h-1)? y: y+1;
		return y*w + x;
	}
	else
		return -1;
}

__DEVICE__ int Indexing::getBackTrace(vec2 const &v, float const &dt){
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	x -= static_cast<int>(v[0]*dt);
	y -= static_cast<int>(v[1]*dt);

	//clamp to edges
	x = x<0? 0: (x>w-1? w-1: x);
	y = y<0? 0: (y>h-1? h-1: y);

	return y*w + x;
}

__DEVICE__ bool Indexing::isLeftBoundary(){
	auto x = blockDim.x*blockIdx.x + threadIdx.x;
	return x==0;
}

__DEVICE__ bool Indexing::isRightBoundary(){
	auto x = blockDim.x*blockIdx.x + threadIdx.x;
	return x==w-1;
}
__DEVICE__ bool Indexing::isTopBoundary(){
	auto y = blockDim.y*blockIdx.y + threadIdx.y;
	return y==0;
}

__DEVICE__ bool Indexing::isBottomBoundary(){
	auto y = blockDim.y*blockIdx.y + threadIdx.y;
	return y==h-1;
}