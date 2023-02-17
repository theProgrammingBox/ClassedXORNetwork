#pragma once
#include "Layer.h"

class LinearLayer : public Layer
{
public:
	LinearLayer(uint32_t outputSize)
	{
		outputMatrix = new Matrix(1, outputSize);
	}

	~LinearLayer()
	{
		delete[] weights;
		delete[] output;
	}

	void Forward() override
	{
		cpuSgemmStridedBatched(
			false, false,
			outputSize, 1, inputSize,
			&GLOBAL::ONEF,
			weights, inputSize, 0,
			input, inputSize, 0,
	}
	
	Matrix* weightMatrix;
};