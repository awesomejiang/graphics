#ifndef DEVICESTRUCTS_H
#define DEVICESTRUCTS_H

#define MAX_PARTICLE 10

struct DeviceParticle{
	vec3 pos;
	vec3 vel;
	float density;
	float pressure;
};

struct DeviceGridCell{
	int num = 0;
	vec3 pos[MAX_PARTICLE];
	int padding;
};


#endif