#include "mathsolver.h"

__GLOBAL__ void colorDiffusionStep(Indexing *indexing, vec3 *currC, vec3 *tempC, vec3 *oldC, float dt, float niu){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;
	vec3 const &leftC = tempC[indexing->getLeft()];
	vec3 const &rightC = tempC[indexing->getRight()];
	vec3 const &topC = tempC[indexing->getTop()];
	vec3 const &bottomC = tempC[indexing->getBottom()];

	currC[idx] = (oldC[idx] + niu*dt*(leftC+rightC+topC+bottomC))/(1+4*niu*dt);
}


__GLOBAL__ void diffusionStep(Indexing *indexing, vec2 *currV, vec2 *tempV, vec2 *oldV, float dt, float niu){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;
	vec2 const &leftV = tempV[indexing->getLeft()];
	vec2 const &rightV = tempV[indexing->getRight()];
	vec2 const &topV = tempV[indexing->getTop()];
	vec2 const &bottomV = tempV[indexing->getBottom()];

	currV[idx] = (oldV[idx] + niu*dt*(leftV+rightV+topV+bottomV))/(1+4*niu*dt);
}

__GLOBAL__ void colorAdvect(Indexing *indexing, vec3 *currC, vec3 *oldC, vec2 *oldV, float dt){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;
	//get backtrace
	auto backtrace = indexing->getBackTrace(oldV[idx], dt);
	currC[idx] = oldC[backtrace];
	if(idx == 160000-1200)
		printf("color: %f, %d\n", currC[idx][0], idx-backtrace);
}

__GLOBAL__ void advect(Indexing *indexing, vec2 *currV, vec2 *oldV, float dt){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;
	//get backtrace
	auto backtrace = indexing->getBackTrace(oldV[idx], dt);
	currV[idx] = oldV[backtrace];
	if(idx == 160000-1200)
		printf("vel: %d\n", idx-backtrace);
}


//jacobi iterative solver: div(v*)
__GLOBAL__ void divergence(Indexing *indexing, float *div, vec2 *currV){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;
	vec2 const &leftV = currV[indexing->getLeft()];
	vec2 const &rightV = currV[indexing->getRight()];
	vec2 const &topV = currV[indexing->getTop()];
	vec2 const &bottomV = currV[indexing->getBottom()];

	div[idx] = (rightV[0] - leftV[0] + topV[1] - bottomV[1])/2.0f;
}

__GLOBAL__ void pressureStep(Indexing *indexing, float *currP, float *oldP, float *div){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;
	float const &leftP = oldP[indexing->getLeft()];
	float const &rightP = oldP[indexing->getRight()];
	float const &topP = oldP[indexing->getTop()];
	float const &bottomP = oldP[indexing->getBottom()];

	currP[idx] = (leftP + rightP + topP + bottomP - div[idx])/4.0f;
}

__GLOBAL__ void correction(Indexing *indexing, vec2 *currV, float *currP){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;
	float const &leftP = currP[indexing->getLeft()];
	float const &rightP = currP[indexing->getRight()];
	float const &topP = currP[indexing->getTop()];
	float const &bottomP = currP[indexing->getBottom()];

	currV[idx] -= vec2{rightP-leftP, topP-bottomP}/2.0f;
//	if(idx == 160000-1200)
//		printf("vel: %f, %f\n", currV[idx][0], currV[idx][1]);
}