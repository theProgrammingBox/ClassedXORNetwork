#include "Network.h"
#include "LeakyReluLayer.h"

int main()
{
	Network network;
	
	Matrix inputMatrix(1, 2);
	Matrix* outputMatrix;
	Matrix outputDerivativeMatrix(1, 1);
	Matrix* inputDerivativeMatrix;

	LeakyReluLayer leakyReluLayer(2);

	network.AddLayer(&leakyReluLayer);
	network.Initialize(&inputMatrix, &outputDerivativeMatrix);

	for (uint32_t i = inputMatrix.totalSize; i--;)
		inputMatrix.matrix[i] = i;

	outputMatrix = network.Forward();
	
	PrintMatrix(inputMatrix.matrix, inputMatrix.rows, inputMatrix.columns, "Input Matrix");
	PrintMatrix(outputMatrix->matrix, outputMatrix->rows, outputMatrix->columns, "Output Matrix");/**/

	//out
	
	delete[] inputMatrix.matrix;
	delete[] outputDerivativeMatrix.matrix;

	return 0;
}