#ifndef STRUCTS_H
#define STRUCTS_H

#define MAX_PARTICLE 3

struct DeviceParticleArray{
	vec3 *pos = nullptr;
	vec3 *vel = nullptr;
	vec3 *force = nullptr;
	float *density = nullptr;
	float *pressure = nullptr;
};

struct DeviceGridCell{
	int num = 0;
	int index[MAX_PARTICLE];
};


struct Cube{
	float lowerX, upperX, lowerY, upperY, lowerZ, upperZ;
};

struct ParticleParams{
	int num;
	float k, gamma, miu, h, dt;
	vec4 color;
};


#endif