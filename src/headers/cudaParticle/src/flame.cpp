#include "flame.h"

__GLOBAL__ void updateKernel(FlameParticle* dp, int n, Mouse const &mouse){
    int index = getIdx();
    if(index > n)
    	return ;

    dp[index] = FlameParticle();
}

template class ParticleSystem<FlameParticle>;