#pragma once
#include "Layer.h"

class LeakyReluLayer : public Layer
{
public:
	LeakyReluLayer(uint32_t outputSize)
	{
		productMatrix = Matrix(1, outputSize);
		activationMatrix = Matrix(1, outputSize);

		productDerivativeMatrix = Matrix(1, outputSize);
	}
	
	~LeakyReluLayer() override
	{
	}
	
	void Forward() override
	{
		cpuSgemmStridedBatched(
			false, false,
			productMatrix.columns, productMatrix.rows, inputMatrix->columns,
			&GLOBAL::ONEF,
			weightMatrix.matrix, weightMatrix.columns, 0,
			inputMatrix->matrix, inputMatrix->columns, 0,
			&GLOBAL::ZEROF,
			productMatrix.matrix, productMatrix.columns, 0,
			1);
		cpuSaxpy(productMatrix.totalSize, &GLOBAL::ONEF, biasMatrix.matrix, 1, productMatrix.matrix, 1);
		cpuLeakyRelu(productMatrix.matrix, activationMatrix.matrix, productMatrix.totalSize);
	}

	void Backward() override
	{
		cpuLeakyReluDerivative(productMatrix.matrix, activationDerivativeMatrix->matrix, productDerivativeMatrix.matrix, activationDerivativeMatrix->totalSize);
		cpuSgemmStridedBatched(
			true, false,
			inputDerivativeMatrix.columns, inputDerivativeMatrix.rows, productDerivativeMatrix.columns,
			&GLOBAL::ONEF,
			weightMatrix.matrix, weightMatrix.columns, 0,
			productDerivativeMatrix.matrix, productDerivativeMatrix.columns, 0,
			&GLOBAL::ZEROF,
			inputDerivativeMatrix.matrix, inputDerivativeMatrix.columns, 0,
			1);
		cpuSgemmStridedBatched(
			false, true,
			weightDerivativeMatrix.columns, weightDerivativeMatrix.rows, inputMatrix->rows,
			&GLOBAL::ONEF,
			productDerivativeMatrix.matrix, productDerivativeMatrix.columns, 0,
			inputMatrix->matrix, inputMatrix->columns, 0,
			&GLOBAL::ZEROF,
			weightDerivativeMatrix.matrix, weightDerivativeMatrix.columns, 0,
			1);
	}

	void Update(float scalar) override
	{
		cpuSaxpy(weightDerivativeMatrix.totalSize, &scalar, weightDerivativeMatrix.matrix, 1, weightMatrix.matrix, 1);
		cpuSaxpy(biasDerivativeMatrix.totalSize, &scalar, biasDerivativeMatrix.matrix, 1, biasMatrix.matrix, 1);
	}

	void AssignInputMatrix(Matrix* inputMatrix) override
	{
		this->inputMatrix = inputMatrix;
	}
	
	Matrix* GetOutputMatrix() override
	{
		return &activationMatrix;
	}

	void AssignOutputDerivativeMatrix(Matrix* outputDerivativeMatrix) override
	{
		this->activationDerivativeMatrix = outputDerivativeMatrix;
	}

	Matrix* GetInputDerivativeMatrix() override
	{
		return &inputDerivativeMatrix;
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

	Matrix* inputMatrix;
	Matrix weightMatrix;
	Matrix biasMatrix;
	Matrix productMatrix;
	Matrix activationMatrix;

	Matrix inputDerivativeMatrix;
	Matrix weightDerivativeMatrix;
	Matrix biasDerivativeMatrix;
	Matrix productDerivativeMatrix;
	Matrix* activationDerivativeMatrix;
};