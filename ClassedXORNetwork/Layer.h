#pragma once
#include "Matrix.h"

class Layer
{
public:
	Layer() {};
	virtual ~Layer() {};

	virtual void Forward() = 0;
	virtual void Backward() = 0;
	virtual void Update(float scalar) = 0;
	void AssignInputMatrix(Matrix* inputMatrix) { this->inputMatrix = inputMatrix; }
	Matrix* GetOutputMatrix() { return outputMatrix; }

	Matrix* inputMatrix;
	Matrix* 
	Matrix* outputMatrix;
};