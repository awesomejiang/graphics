#ifndef GRIDCELLS_H
#define GRIDCELLS_H

#include <cuda.h>

#include "utility.h"
#include "vec_float.h"
#include "structs.h"

class GridCells{
public:
	GridCells(float h);
	~GridCells();

	void clear();
	void insertParticle(DeviceParticleArray const &dpa, ParticleParams params, dim3 g, dim3 b);

	DeviceGridCell *getCells() const;

private:
	dim3 block;
	dim3 grid;
	unsigned int cellDim, cellNum;
	DeviceGridCell *cells = nullptr;
	void deployGrid();
};

//cuda kernels
//__GLOBAL__ void initGridCells(DeviceGridCell *cells, int num);
__GLOBAL__ void clearGridCells(DeviceGridCell *cells, int num);
__GLOBAL__ void updateGridCells(DeviceGridCell *cells, vec3 *positions, int pNum, float h);
//__GLOBAL__ void destroyGridCells(DeviceGridCell *cells, int cellNum);

#endif