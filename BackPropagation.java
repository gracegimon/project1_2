import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
/*
	Backpropagation for a neural network
	of two layers
*/
class BackPropagation{

	static double[][] trainData;
	static double[][] weightsInput; // Wjk, row: Neuron, col: Xk
	static double[][] weightsOutput; // Wji, row: Hidden, col: Output
	static double[] deltaW; // Delta diference , Dj, Di 
	static double[] derivativeValue;
	static double[] activationValue; //Value from the activation function on each neuron, Aj, Ai
	static double[][] errors; // Errors for each weight 
	/*
		Example:  neuron[j] in the layer[i]
	*/
	int numberOfNeurons = 2; // Number of Neurons in the hidden layer
	int numberOfInputs = 2;
	int numberOfOutputs = 2;
	double error = 0;
	double output = 0;
	int maxIterations = 0;

	public Backpropagation(int numberOfWeights, int numberOfNeurons, int numberOfOutputs, int numberOfExamples){
		trainData = new double[numberOfExamples][numberOfWeights];
		numberOfNeurons = numberOfNeurons;
		numberOfOutputs = numberOfOutputs;
		numberOfInputs = numberOfWeights;
		weightsInput = new double[numberOfNeurons][numberOfWeights];
		weightsOutput = new double[numberOfNeurons][numberOfOutputs];
		deltaW = new double[numberOfOutputs+numberOfNeurons];
		derivativeValue = new double[numberOfOutputs+numberOfNeurons];
		activationValue = new double[numberOfOutputs+numberOfNeurons+numberOfInputs];

	}

/**
 *  @param  weights : Weights from a neuron 
*/
private double activationFunction(double[] weights, int ini){
	double activation = 0;
	for (int i = 0; i < weights.length; i++)// Sumi Wij * activationValue
	{ 
		activation += weights[i] * activationValue[ini];
		ini++;
	}
	
	return activation; 
}

private double derivativeActivationFunction(double value ){
	return (1.0 - (Math.pow(value),2));
}

private double calculateErrorHidden(int j)
{
	double error = 0.0;
	for (int k = numberOfNeurons; k < deltaW.length; k++)
	{
		for (int i = 0 ; i<weightsOutput.length; i++)
		{
			error += weightsOutput[j][i] * deltaW[k]
		}
	}
	return error;
}


/** 
*	@param weights : Input weights
*/
private void backPropagation(double[] weights, double learningRate){

		// Random initialization of weights
		for (int i = 0; i < numberOfNeurons; i++)
		{
			for (int j = 0; j< numberOfInputs; j++)
			{
				weightsInput[i][j] = Math.random()/10; // Check
			}		
		}
		// In this case, numberOfInputs = numberOfOutputs
		for (int i = 0; i < numberOfNeurons; i++)
		{
			for (int j = 0; j< numberOfOutputs; j++)
			{
				weightsOutput[i][j] = Math.random()/10; // Check
			}		
		}

		int currentIteration = 0;
		
		do
		{
			for (int i = 0; i< trainData.length; i++)
			{

				//Calculate activation function for input
				for (int j = 0; j<trainData[i].length -1 ; j++)
				{
					activationValue[j] = trainData[i][j]; //Input layer (Xk)
				}

				//Calculate activation function for each hidden layer
				// Supposing just 1 hidden layer
				int iniHidden = numberOfInputs;
				for (int k = 0; k< weightsInput.length; k++)
				{
					for (int j = iniHidden; j< numberOfInputs + numberOfNeurons; j++)
					{
						double activation = Math.tanh.value(activationFunction(weightsInput[k],0));
						activationValue[j] = activation; // Sum * Ak
						derivativeValue[j-numberOfInputs] = derivativeActivationFunction(activation);

					}

				}

				int iniOutput = numberOfInputs+ numberOfNeurons;

				//Calculate activation function for output layer
				for (int k = 0; k< weightsOutput.length; k++)
				{
					for (int j = iniOutput; j< iniOutput + numberOfOutputs ; j++)
					{
						double activation = Math.tanh.value(activationFunction(weightsOutput[k],iniOutput));
						activationValue[j] = activation;
						derivativeValue[j-numberOfNeurons] = derivativeActivationFunction(activation);

					}

				}
				double target = trainData[i][trainData[i].length-1];
				// Calculate Delta i
				for (int j = numberOfNeurons; j < deltaW.length; j++)
				{ //Last positions of the array
					
					deltaW[j] = (target - activationValue[j+numberOfInputs]) * derivativeValue[j];
				}

				//Calculate Delta j
				for (int j = 0; j< numberOfNeurons; j++)
				{
					deltaW[j] = derivativeValue[j] * calculateErrorHidden(j);
				}


				// Update weights
				for (int k = 0; k< numberOfNeurons; k++)
				{
					for (int j = 0; j< numberOfOutputs; j++)
					{
						weightsOutput[k][j] += learningRate;
						weightsOutput[k][j] *= activationValue[k+numberOfInputs] * deltaW[j+numberOfNeurons];
					}
				}

			}


			currentIteration++;
		}while(currentIteration != maxIterations)

}


public static void main(String[] args) {
	Backpropagation bp = new Backpropagation();

	System.out.println("Weights");
}


}