#include "Network.h"
#include "LeakyReluLayer.h"

int main()
{
	Network network;
	
	Matrix inputMatrix(1, 2);
	Matrix* outputMatrix;
	Matrix outputDerivativeMatrix(1, 1);
	Matrix* inputDerivativeMatrix;

	LeakyReluLayer leakyReluLayer(1);

	network.AddLayer(&leakyReluLayer);
	network.Initialize(&inputMatrix, &outputDerivativeMatrix);

	for (uint32_t i = 100; i--;)
	{
		inputMatrix.matrix[0] = 0.0f;
		inputMatrix.matrix[1] = 1.0f;

		outputMatrix = network.Forward();

		outputDerivativeMatrix.matrix[0] = 0.5f - outputMatrix->matrix[0];

		inputDerivativeMatrix = network.Backward();

		//network.Print();
		
		PrintMatrix(outputMatrix->matrix, outputMatrix->rows, outputMatrix->columns, "Output Matrix");

		network.Update(3.0f);
	}
	
	delete[] inputMatrix.matrix;
	delete[] outputDerivativeMatrix.matrix;

	return 0;
}