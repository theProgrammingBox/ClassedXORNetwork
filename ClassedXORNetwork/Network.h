#pragma once
#include "Layer.h"

class Network
{
public:
	vector<Layer*> Layers;

	Network() {}
	~Network() { for (auto& layer : Layers) delete layer; }
	
	void AddLayer(Layer* layer) { Layers.push_back(layer); }
	void Forward() { for (auto& layer : Layers) layer->Forward(); }
	void Backward() { for (auto& layer : Layers) layer->Backward(); }
};