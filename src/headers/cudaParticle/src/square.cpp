#include "square.h"

//VAO
void Square::setVAO() const{
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, sizeof(vec2)/sizeof(GL_FLOAT), GL_FLOAT, GL_FALSE,
	 sizeof(SquareParticle), OFFSETOF(SquareParticle, pos));

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, sizeof(vec2)/sizeof(GL_FLOAT), GL_FLOAT, GL_FALSE,
	 sizeof(SquareParticle), OFFSETOF(SquareParticle, vel));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, sizeof(vec4)/sizeof(GL_FLOAT), GL_FLOAT, GL_FALSE,
	 sizeof(SquareParticle), OFFSETOF(SquareParticle, color));
}


//kernels
__GLOBAL__ void initKernel(SquareParticle* p, int n, Mouse const &mouse){
	auto idx = getIdx();
	if(idx > n)
		return ;
	
	SquareParticle &pr = p[idx];

	curand_init(clock64(), idx, 0, &pr.rand);
	//position
	float randX = (curand_uniform(&pr.rand)*2.0f - 1.0f) * HALFWIDTH;
	float randY = (curand_uniform(&pr.rand)*2.0f - 1.0f) * HALFHEIGHT;
	switch(idx%4){
		case 0: pr.pos = vec2(-HALFWIDTH, randY); break;
		case 1: pr.pos = vec2(HALFWIDTH, randY); break;
		case 2: pr.pos = vec2(randX, -HALFHEIGHT); break;
		case 3: pr.pos = vec2(randX, HALFHEIGHT); break;
	}

	float r = 0.1f*curand_normal(&pr.rand), theta = curand_uniform(&pr.rand)*2*M_PI;
	pr.pos += vec2(r*cos(theta), r*sin(theta));
	
	pr.vel = {0.0f, 0.0f};
	pr.color = vec4(0.1f, 0.2f, 0.3f, 1.0f);
}


__GLOBAL__ void updateKernel(SquareParticle* p, int n, Mouse const &mouse){
	auto idx = getIdx();
	if(idx > n)
		return ;

	SquareParticle &pr = p[idx];

	float r = 0.001f*curand_normal(&pr.rand), theta = curand_uniform(&pr.rand)*2*M_PI;
	pr.pos += vec2(r*cos(theta), r*sin(theta));
		
	if(mouse.firstClicked)
		pr.color -= vec4(0.0001f, 0.0002f, 0.0003f, 0.0f);
}