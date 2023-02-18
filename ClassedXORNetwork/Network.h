#pragma once
#include "Layer.h"

class Network
{
public:
	vector<Layer*> Layers;

	Network() {}
	~Network() {}
	
	void AddLayer(Layer* layer)
	{
		Layers.push_back(layer);
	}
	
	void Initialize(Matrix* inputMatrix, Matrix* outputDerivativeMatrix)
	{
		Layers.back()->AssignOutputDerivativeMatrix(outputDerivativeMatrix);
		for (uint32_t i = Layers.size() - 1; i--;)
		{
			Layers[i]->AssignOutputDerivativeMatrix(Layers[i + 1]->GetInputDerivativeMatrix());
			Layers[i + 1]->AssignInputMatrix(Layers[i]->GetOutputMatrix());
			Layers[i + 1]->Reset();
		}
		Layers[0]->AssignInputMatrix(inputMatrix);
		Layers[0]->Reset();
	}

	void Print()
	{
		for (auto& layer : Layers)
			layer->Print();
	}
	
	Matrix* Forward()
	{
		for (auto& layer : Layers)
			layer->Forward();
		return Layers.back()->GetOutputMatrix();
	}
	
	Matrix* Backward()
	{
		for (auto& layer : Layers)
			layer->Backward();
		return Layers[0]->GetInputDerivativeMatrix();
	}
	
	void Update(float scalar)
	{
		for (auto& layer : Layers)
			layer->Update(scalar);
	}
};