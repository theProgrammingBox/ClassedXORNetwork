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

		biasMatrix = Matrix(1, outputSize);
		biasDerivativeMatrix = Matrix(1, outputSize);

		biasMatrix.Randomize();
	}
	
	~LeakyReluLayer() override
	{
		delete[] weightMatrix.matrix;
		delete[] biasMatrix.matrix;
		delete[] productMatrix.matrix;
		delete[] activationMatrix.matrix;
		
		delete[] inputDerivativeMatrix.matrix;
		delete[] weightDerivativeMatrix.matrix;
		delete[] biasDerivativeMatrix.matrix;
		delete[] productDerivativeMatrix.matrix;
	}

	void AssignInputMatrix(Matrix* inputMatrix) override
	{
		this->inputMatrix = inputMatrix;
		inputDerivativeMatrix = Matrix(1, inputMatrix->columns);

		weightMatrix = Matrix(inputMatrix->columns, productMatrix.columns);
		weightDerivativeMatrix = Matrix(inputMatrix->columns, productMatrix.columns);

		weightMatrix.Randomize();
	}

	Matrix* GetOutputMatrix() override
	{
		return &activationMatrix;
	}

	void AssignOutputDerivativeMatrix(Matrix* outputDerivativeMatrix) override
	{
		activationDerivativeMatrix = outputDerivativeMatrix;
	}

	Matrix* GetInputDerivativeMatrix() override
	{
		return &inputDerivativeMatrix;
	}

	void Print() override
	{
		PrintMatrix(inputMatrix->matrix, inputMatrix->rows, inputMatrix->columns, "inputMatrix");
		PrintMatrix(weightMatrix.matrix, weightMatrix.rows, weightMatrix.columns, "weightMatrix");
		PrintMatrix(biasMatrix.matrix, biasMatrix.rows, biasMatrix.columns, "biasMatrix");
		PrintMatrix(productMatrix.matrix, productMatrix.rows, productMatrix.columns, "productMatrix");
		PrintMatrix(activationMatrix.matrix, activationMatrix.rows, activationMatrix.columns, "activationMatrix");

		PrintMatrix(inputDerivativeMatrix.matrix, inputDerivativeMatrix.rows, inputDerivativeMatrix.columns, "inputDerivativeMatrix");
		PrintMatrix(weightDerivativeMatrix.matrix, weightDerivativeMatrix.rows, weightDerivativeMatrix.columns, "weightDerivativeMatrix");
		PrintMatrix(biasDerivativeMatrix.matrix, biasDerivativeMatrix.rows, biasDerivativeMatrix.columns, "biasDerivativeMatrix");
		PrintMatrix(productDerivativeMatrix.matrix, productDerivativeMatrix.rows, productDerivativeMatrix.columns, "productDerivativeMatrix");
		PrintMatrix(activationDerivativeMatrix->matrix, activationDerivativeMatrix->rows, activationDerivativeMatrix->columns, "activationDerivativeMatrix");
	}

	void Reset() override
	{
		inputDerivativeMatrix.Zero();
		weightDerivativeMatrix.Zero();
		biasDerivativeMatrix.Zero();
		productDerivativeMatrix.Zero();
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
		cpuLeakyRelu(productMatrix.matrix, activationMatrix.matrix, activationMatrix.totalSize);
	}

	void Backward() override
	{
		cpuLeakyReluDerivative(productMatrix.matrix, activationDerivativeMatrix->matrix, productDerivativeMatrix.matrix, productDerivativeMatrix.totalSize);
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
			&GLOBAL::ONEF,
			weightDerivativeMatrix.matrix, weightDerivativeMatrix.columns, 0,
			1);
		cpuSaxpy(biasDerivativeMatrix.totalSize, &GLOBAL::ONEF, productDerivativeMatrix.matrix, 1, biasDerivativeMatrix.matrix, 1);
	}

	void Update(float scalar) override
	{
		cpuSaxpy(weightMatrix.totalSize, &scalar, weightDerivativeMatrix.matrix, 1, weightMatrix.matrix, 1);
		cpuSaxpy(biasMatrix.totalSize, &scalar, biasDerivativeMatrix.matrix, 1, biasMatrix.matrix, 1);
		Reset();
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