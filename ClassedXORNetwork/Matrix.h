#pragma once
#include "Header.h"

class Matrix
{
public:
	float* matrix;
	uint32_t rows;
	uint32_t columns;
	uint32_t totalSize;

	Matrix(uint32_t rows = 1, uint32_t columns = 1)
	{
		this->rows = rows;
		this->columns = columns;
		totalSize = rows * columns;
		matrix = new float[totalSize];
	}

	~Matrix()
	{
		delete[] matrix;
	}

	void Randomize()
	{
		cpuGenerateUniform(matrix, totalSize, -1.0f, 1.0f);
	}
};