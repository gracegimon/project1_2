import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/*
	Backpropagation for a neural network
	of two layers
*/
class BackPropagationL{

	static double[][] trainData;
	static double[][] weightsInput; // Wjk, row: Neuron, col: Xk
	static double[][] weightsOutput; // Wji, row: Hidden, col: Output
	// static double[] deltaWJ; // Delta diference , Dj, Di
	// static double[] deltaWJI; // Delta diference , Dj, Di 
	// static double[] derivativeValueJ;
	// static double[] derivativeValueI; 
	// static double[] activationValueJ; //Value from the activation function on each hidden neuron
	// static double[] activationValueI; //Value from the activation function on each output neuron
	// static double[]	activationValueK;
	Vector<Neuron> inputNeuron;
	Vector<Neuron> hiddenNeuron;
	Vector<Neuron> outputNeuron;

	static double[][] errors; // Errors for each weight 
	/*
		Example:  neuron[j] in the layer[i]
	*/
	int numberOfNeurons = 2; // Number of Neurons in the hidden layer
	int numberOfInputs = 2; // Number of neurons in the input layer
	int numberOfOutputs = 2; // Number of neurons in the output layer
	double error = 0;
	double output = 0;
	int maxIterations = 0;

	public BackPropagationL(int numberOfWeights, int numberOfNeurons, int numberOfOutputs, int numberOfExamples){
		trainData = new double[numberOfExamples][numberOfWeights];
		numberOfNeurons = numberOfNeurons;
		numberOfOutputs = numberOfOutputs;
		numberOfInputs = numberOfWeights;
		inputNeurons = new Vector<Neuron>();
		hiddenNeurons = new Vector<Neuron>();
		outputNeurons = new Vector<Neuron>();
		weightsInput = new double[numberOfNeurons][numberOfWeights];
		weightsOutput = new double[numberOfNeurons][numberOfOutputs];

	}

/**
 *  @param  weights : Weights from a neuron 
*/
private double activationFunction(Neuron neuron){
	double activation = 0;
	double[] weights = neuron.weights;
	if (neuron.type == 1){ 
		double activationValues = new double[numberOfInputs];
		//Then the ActivationValues come from input layer
		for (int l =0; l< inputNeurons.length; l++)
		{
			activationValues[l] = inputNeurons.get(l).activationValue;
		}

		for (int i = 0; i < weights.length; i++)// Sumi Wij * activationValue
		{ 
			activation += weights[i] * activationValue[i];
		}		
	}
	else {
		//ActivationValues come from hidden layer
		double activationValues = new double[numberOfNeurons];
		for (int l =0; l< hiddenNeurons.length; l++)
		{
			activationValues[l] = hiddenNeurons.get(l).activationValue;
		}

		for (int i = 0; i < weights.length; i++)// Sumi Wij * activationValue
		{ 
			activation += weights[i] * activationValue[i];
		}	

	}
	return Math.tanh.value(activation); 
}

private double derivativeActivationFunction(double value ){
	return (1.0 - (Math.pow(value),2));
}

private double calculateErrorHidden(Neuron n) 
{ //MODIFY!!! 
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
			for (int i = 0; i< trainData.length; i++) //For each example
			{
				Neuron n;

				if (i == 0) //no neuron created
				{
					//Calculate activation function for input
					for (int j = 0; j < numberOfInputs; j++)
					{
						n = new Neuron(numberOfInputs);
						//Add weights
						n.activationValue = trainData[i][j];
						inputNeurons.add(n);
						 //Input layer (Xk)
					}

					//Calculate activation function for hidden
					for (int j = 0; j < numberOfNeurons; j++)
					{
						n = new Neuron(numberOfInputs);
						n.activationValue = activationFunction(n);
						n.derivativeValue = derivativeActivationFunction(n.activationValue);
						hiddenNeurons.add(n);
					}

					for (int j = 0; j < numberOfOutputs; j++)
					{
						n = new Neuron(numberOfNeurons);
						n.activationValue = activationFunction(n);
						n.derivativeValue = derivativeActivationFunction(n.activationValue);
						hiddenNeurons.add(n);
					}

				}
						
				else
				{
					for (int l = 0; l< inputNeurons.length; l++)
					{
						n = inputNeurons.get(l);
						n.activationValue = trainData[i][j];
					}

					for (int l = 0; l< hiddenNeurons.length; l++)
					{
						n = hiddenNeurons.get(l);
						n.activationValue = activationFunction(n);
						n.derivativeValue = derivativeActivationFunction(n.activationValue);
					}

					for (int l = 0; l< outputNeurons.length; l++)
					{
						n = outputNeurons.get(l);
						n.activationValue = activationFunction(n);
						n.derivativeValue = derivativeActivationFunction(n.activationValue);
					}							

				}
				
				double target = trainData[i][trainData[i].length-1];
				// Calculate Delta i
				for (int j = 0; j < outputNeurons.length; j++)
				{ 
					n = outputNeurons.get(j);
					n.deltaW = (target - n.activationValue) * n.derivativeValue;
				}

				//Calculate Delta j
				for (int j = 0; j< hiddenNeurons.length; j++)
				{
					n = hiddenNeurons.get(j);
					n.deltaW = n.derivativeValue * calculateErrorHidden(n);
				}


				// Update weights for Wij
				for (int j = 0; j< outputNeurons.length; j++)
				{
					for (int k = 0; j< hiddenNeurons.length; j++)
					{
						// MODIFY!!
						weightsOutput[k][j] += learningRate;
						weightsOutput[k][j] *= activationValue[k+numberOfInputs] * deltaW[j+numberOfNeurons];
					}
				}

			}
			currentIteration++;
		}while(currentIteration != maxIterations)

}


public static void main(String[] args) {
	BackPropagationL bp = new BackPropagationL();

	System.out.println("Weights");
}


}