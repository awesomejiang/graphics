#include "gravity.h"

//VAO
void Gravity::setVAO() const{
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, sizeof(vec2)/sizeof(GL_FLOAT), GL_FLOAT, GL_FALSE,
	 sizeof(GravityParticle), OFFSETOF(GravityParticle, pos));

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, sizeof(vec2)/sizeof(GL_FLOAT), GL_FLOAT, GL_FALSE,
	 sizeof(GravityParticle), OFFSETOF(GravityParticle, vel));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, sizeof(vec4)/sizeof(GL_FLOAT), GL_FLOAT, GL_FALSE,
	 sizeof(GravityParticle), OFFSETOF(GravityParticle, color));
}


//kernels
__GLOBAL__ void initKernel(GravityParticle* p, int n, Mouse const &mouse){
	auto idx = getIdx();
	if(idx > n)
		return ;

	GravityParticle &pr = p[idx];

	pr.live = true;
	curand_init(clock64(), idx, 0, &pr.rand);
	pr.pos = {curand_uniform(&pr.rand) * 2.0f - 1.0f, -1.0f};
	pr.vel = {0.0f, 0.0f};
	pr.color = {1.0f, 1.0f, 1.0f, 1.0f};
}


__GLOBAL__ void updateKernel(GravityParticle* p, Mouse const &mouse){
	auto idx = getIdx();

	GravityParticle &pr = p[idx];

	float SOFTEN = 0.00000001f;
	float THRESH = 0.01f;

	if(mouse.pressed){
		// gravity
		float G = 0.000005f;
		auto dist = length(mouse.pos-pr.pos) + SOFTEN;
		pr.vel += G/(dist*dist*dist) * (mouse.pos-pr.pos);

		auto spd = length(pr.vel);
		if(spd > THRESH)
			pr.vel /= (spd/THRESH);
	}
	//printf("%f, %f\n", pos[0], pos[1]);

	//pos move
	pr.pos += pr.vel;
	//add a downside force field
	float g = 0.000005f;
	pr.vel += vec2(0.0f, -g);

	//when meet boundary
	//bump back and get a vel decay
	if(pr.pos[0] < -1.0f){
		pr.pos[0] = -1.0f;
		pr.vel[0] = abs(pr.vel[0]) * 0.5;
	}
	else if(pr.pos[0] > 1.0f){
		pr.pos[0] = 1.0f;
		pr.vel[0] = -abs(pr.vel[0]) * 0.5;
	}
	if(pr.pos[1] > 1.0f){
		pr.pos[1] = 1.0f;
		pr.vel[1] = -abs(pr.vel[1]) * 0.5;
	}
	else if(pr.pos[1] < -1.0f){
		pr.pos[1] = -1.0f;
		pr.vel[1] = abs(pr.vel[1]) * 0.5;
	}
}