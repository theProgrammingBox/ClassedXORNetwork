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
	
	virtual void AssignInputMatrix(Matrix* inputMatrix) = 0;
	virtual Matrix* GetOutputMatrix() = 0;
	
	virtual void AssignOutputDerivativeMatrix(Matrix* outputDerivativeMatrix) = 0;
	virtual Matrix* GetInputDerivativeMatrix() = 0;
};