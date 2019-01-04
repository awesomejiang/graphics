#ifndef GRIDCELLS_H
#define GRIDCELLS_H

#include <cuda.h>

#include "utility.h"
#include "vec_float.h"
#include "deviceStructs.h"

class GridCells{
public:
	GridCells(float h);
	~GridCells();

	void clear();
	void insertParticle(DeviceParticle const *p, dim3 g, dim3 b, int num);

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
__GLOBAL__ void updateGridCells(DeviceGridCell *cells, DeviceParticle const *p, int pNum, int cellDim);
//__GLOBAL__ void destroyGridCells(DeviceGridCell *cells, int cellNum);

#endif