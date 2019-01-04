#include "fluidSystem.h"

void FluidSystem::addParticle(ParticleParams const &params){
	particles.emplace_back(params);
}


void FluidSystem::render(Camera const &camera){
	//
	//update cell
	gc.clear();
	for(auto &p: particles){
		gc.insertParticle(p.getParticle(), p.getGrid(), p.getBlock(), p.getNum());
	}
	auto cells = gc.getCells(); 

	//update particles and draw
	shader.use();
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	shader.setUniform("projection", camera.getProjection());
	shader.setUniform("view", camera.getView());

	for(auto &p: particles){
		p.render(cells);
		auto color = p.getColor();
		shader.setUniform("color", color[0], color[1], color[2], color[3]);
	}

	glDisable(GL_BLEND);
}

/*
//VAO
void Fluid::setVAO() const{
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, sizeof(vec3)/sizeof(GL_FLOAT), GL_FLOAT, GL_FALSE,
	 sizeof(FluidParticle), OFFSETOF(FluidParticle, pos));

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, sizeof(vec4)/sizeof(GL_FLOAT), GL_FLOAT, GL_FALSE,
	 sizeof(FluidParticle), OFFSETOF(FluidParticle, color));
}


//kernels
__GLOBAL__ void initKernel(FluidParticle* p, int n, Mouse const &mouse){
	auto idx = getIdx();
	if(idx > n)
		return ;

	FluidParticle &pr = p[idx];

	curand_init(clock64(), idx, 0, &pr.rand);
	pr.pos = {curand_uniform(&pr.rand) * 2.0f - 1.0f, 0.0f, 0.0f};
	pr.vel = {0.0f, 0.0f, 0.0f};
	pr.color = {1.0f, 1.0f, 1.0f, 1.0f};
}


__GLOBAL__ void updateKernel(FluidParticle* p, int n, Mouse const &mouse){
	auto idx = getIdx();
	if(idx > n)
		return ;

	FluidParticle &pr = p[idx];

	//params
	float dt = 0.001f;
	float h = 0.001f;
	float mass = 0.0001f;
	float k = 0.0001f;
	float gamma = 1.1f;
	float density = 1.0f;//can be computed by above params

	//loop all neighbors to compute density
	pr.density = 0;
	for(int i=0; i<n; ++i){
		pr.density += weight(pr.pos, p[i].pos, h);
	}
	pr.density *= mass;

	//compute pressure based on density
	pr.pressure = k*pr.density/gamma*(pow(pr.density/density, gamma) - 1);

	//compute external force
	vec3 f;
	for(int i=0; i<n; ++i){
		f += (pr.pressure/(pr.density*pr.density))*divWeight(pr.pos, p[i].pos, h);
	}
	f *= -2*mass*mass;
	f += vec3{0.0f, 0.0f, -1.0f};

	//update pos and vel
	pr.vel += f*dt/mass;
	pr.pos += dt*pr.vel;

	//when meet boundary
	//bump back and get a vel decay
	for(int i=0; i<3; ++i){
		if(pr.pos[i] < -1.0f){
			pr.pos[i] = -1.0f;
			pr.vel[i] = abs(pr.vel[i]) * 0.5;
		}
		else if(pr.pos[i] > 1.0f){
			pr.pos[i] = 1.0f;
			pr.vel[i] = -abs(pr.vel[i]) * 0.5;
		}
	}
}

//w = 0.25*(2-q)^3 - (1-q)^3 q=0~1
//w = 0.25*(2-q)^3
//w = 0
__DEVICE__ float weight(vec3 const &src, vec3 const &dst, float const &h){
	auto q = length(src-dst)/h;
	return 0.25 * (q<2? pow(2-q, 3): 0.0f) - (q<1? pow(1-q, 3): 0.0);
}

__DEVICE__ vec3 divWeight(vec3 const &src, vec3 const &dst, float const &h){
	auto diff = (dst-src)/h;
	auto q = length(diff);
	
	vec3 ret;
	if(q < 2){
		ret += 0.25f * (-3.0f*powf(q-2, 2)/q) * diff;
	}
	if(q < 1){
		ret -= (-3.0f*powf(q-1, 2)/q) * diff;
	}

	return 1/h * ret;
}
*/