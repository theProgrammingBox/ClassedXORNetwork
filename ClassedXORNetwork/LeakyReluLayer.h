#pragma once
#include "Layer.h"

class LeakyReluLayer : public Layer
{
public:
	LeakyReluLayer(uint32_t outputSize)
	{

	}
	~LeakyReluLayer() {}
	
	void Forward() override
	{
		cpuSgemmStridedBatched(
			false, false,
			outputMatrix->columns, outputMatrix->rows, inputMatrix->columns,
			&GLOBAL::ONEF,
			weightMatrix->matrix, weightMatrix->columns, 0,
			inputMatrix->matrix, inputMatrix->columns, 0,
			&GLOBAL::ZEROF,
			outputMatrix->matrix, outputMatrix->columns, 0,
			1);
		cpuLeakyRelu(outputMatrix->matrix, activationMatrix->matrix, outputMatrix->totalSize);
	}

	void Backward() override
	{
		cpuLeakyReluDerivative(inputDerivativeMatrix->matrix, activationDerivativeMatrix->matrix, outputDerivativeMatrix->matrix, outputMatrix->totalSize);
		cpuSgemmStridedBatched(
			true, false,
			inputDerivativeMatrix->columns, inputDerivativeMatrix->rows, outputDerivativeMatrix->columns,
			&GLOBAL::ONEF,
			weightMatrix->matrix, weightMatrix->columns, 0,
			outputDerivativeMatrix->matrix, outputDerivativeMatrix->columns, 0,
			&GLOBAL::ZEROF,
			inputDerivativeMatrix->matrix, inputDerivativeMatrix->columns, 0,
			1);
	}

private:
	void cpuLeakyRelu(float* input, float* output, uint32_t size)
	{
		for (size_t counter = size; counter--;)
			output[counter] = (((*(int32_t*)(input + counter) & 0x80000000) >> 31) * 0.9f + 0.1f) * input[counter];
	}

	void cpuLeakyReluDerivative(float* input, float* gradient, float* output, uint32_t size)
	{
		for (size_t counter = size; counter--;)
			output[counter] = (((*(int32_t*)(input + counter) & 0x80000000) >> 31) * 0.9f + 0.1f) * gradient[counter];
	}

	Matrix inputMatrix;
	Matrix weightMatrix;
	Matrix biasMatrix;
	Matrix productMatrix;
	Matrix activationMatrix;
};