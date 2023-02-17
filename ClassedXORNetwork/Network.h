#pragma once
#include "Layer.h"

class Network
{
public:
	vector<Layer*> Layers;

	Network() {}
	~Network() { for (auto& layer : Layers) delete layer; }
	
	void AddLayer(Layer* layer) { Layers.push_back(layer); }
	void Initialize(Matrix* inputMatrix, Matrix* outputDerivativeMatrix)
	{
		this->inputMatrix = inputMatrix;
		this->outputDerivativeMatrix = outputDerivativeMatrix;

		Layers[0]->AssignInputMatrix(inputMatrix);
		for (int i = 1; i < Layers.size(); i++)
			Layers[i]->AssignInputMatrix(Layers[i - 1]->GetOutputMatrix());

		outputMatrix = Layers[Layers.size() - 1]->GetOutputMatrix();

		Layers[Layers.size() - 1]->AssignOutputDerivativeMatrix(outputDerivativeMatrix);
		for (int i = Layers.size() - 2; i >= 0; i--)
			Layers[i]->AssignOutputDerivativeMatrix(Layers[i + 1]->GetInputDerivativeMatrix());

		inputDerivativeMatrix = Layers[0]->GetInputDerivativeMatrix();
	}
	
	Matrix* Forward() { for (auto& layer : Layers) layer->Forward(); }
	Matrix* Backward() { for (auto& layer : Layers) layer->Backward(); }
	void Update(float scalar) { for (auto& layer : Layers) layer->Update(scalar); }

	Matrix* inputMatrix;
	Matrix* outputMatrix;

	Matrix* inputDerivativeMatrix;
	Matrix* outputDerivativeMatrix;
};