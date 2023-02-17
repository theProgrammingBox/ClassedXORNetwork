#include "Network.h"

int main()
{
	Network network;
	network.AddLayer(new LinearLayer(2, 2));
	network.AddLayer(new SigmoidLayer(2));
	network.AddLayer(new LinearLayer(2, 1));
	network.AddLayer(new SigmoidLayer(1));

	vector<float> inputs = { 0, 0 };
	vector<float> targets = { 0 };
	for (uint32_t i = 0; i < 100000; i++) {
		network.Layers[0]->Outputs = inputs;
		network.Forward();
		network.Layers[3]->Errors[0] = targets[0] - network.Layers[3]->Outputs[0];
		network.Backward();
		network.Update();
	}

	network.Layers[0]->Outputs = { 0, 0 };
	network.Forward();
	cout << "0 XOR 0 = " << network.Layers[3]->Outputs[0] << endl;

	network.Layers[0]->Outputs = { 0, 1 };
	network.Forward();
	cout << "0 XOR 1 = " << network.Layers[3]->Outputs[0] << endl;

	network.Layers[0]->Outputs = { 1, 0 };
	network.Forward();
	cout << "1 XOR 0 = " << network.Layers[3]->Outputs[0] << endl;

	network.Layers[0]->Outputs = { 1, 1 };
	network.Forward();
	cout << "1 XOR 1 = " << network.Layers[3]->Outputs[0] << endl;

	return 0;
}