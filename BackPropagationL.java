import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Vector;
import java.util.HashSet;

/*
	Backpropagation for a neural network
	of two layers
*/
class BackPropagationL{

	static double[][] trainData;
	// static double[] deltaWJ; // Delta diference , Dj, Di
	// static double[] deltaWJI; // Delta diference , Dj, Di 
	// static double[] derivativeValueJ;
	// static double[] derivativeValueI; 
	// static double[] activationValueJ; //Value from the activation function on each hidden neuron
	// static double[] activationValueI; //Value from the activation function on each output neuron
	// static double[]	activationValueK;
	Vector<Neuron> inputNeurons;
	Vector<Neuron> hiddenNeurons;
	Vector<Neuron> outputNeurons;

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

	}

/**
 *  @param  weights : Weights from a neuron 
*/
private double activationFunction(Neuron neuron)
{
	double activation = 0;
	Vector<Synapsis> weights = neuron.weights;
	
		double[] activationValues = new double[weights.size()];
		double[] weightsValues = new double[weights.size()];
		
		for (int l =0; l< weights.size(); l++)
		{
			activationValues[l] = weights.get(l).parent.activationValue;
			weightsValues[l] = weights.get(l).value;

		}

		for (int i = 0; i < activationValues.length; i++)// Sumi Wij * activationValue
		{ 
			activation += weightsValues[i] * activationValues[i];
			
		}		
	
	return Math.tanh(activation); 
}

private double derivativeActivationFunction(double value )
{
	return (1.0 - (Math.pow(value,2)));
}

private double calculateErrorHidden() 
{ 
	double error = 0.0;
	for (int k = 0; k < outputNeurons.size(); k++)
	{
		Neuron n = outputNeurons.get(k);
		for (int j = 0; j < n.weights.size(); j++)
		{
			error += n.weights.get(j).value * n.deltaW;
		}	
	}
	return error;
}


/** 
*   @param learningRate : learning rate
*/
private void backPropagation(double learningRate)
{

		int currentIteration = 0;
		
		do
		{
			for (int i = 0; i< trainData.length; i++) //For each example
			{
				Neuron n;

				if (currentIteration == 0) //no neuron created
				{
					//Calculate activation function for input
					for (int j = 0; j < numberOfInputs; j++)
					{
						n = new Neuron();
						n.activationValue = trainData[i][j];
						inputNeurons.add(n);
						 //Input layer (Xk)
					}

					//Calculate activation function for hidden
					for (int j = 0; j < numberOfNeurons; j++)
					{
						n = new Neuron();
						for (int k = 0; k < inputNeurons.size(); k++){
							n.AddSynapsis(new Synapsis(Math.random()/10, inputNeurons.get(k)));
						}

						n.activationValue = activationFunction(n);
						n.derivativeValue = derivativeActivationFunction(n.activationValue); 
						hiddenNeurons.add(n);

					}

					for (int j = 0; j < numberOfOutputs; j++)
					{
						n = new Neuron();
						for (int k = 0; k < hiddenNeurons.size(); k++){
							n.AddSynapsis(new Synapsis(Math.random()/10, hiddenNeurons.get(k)));
						}

						n.activationValue = activationFunction(n);
						n.derivativeValue = derivativeActivationFunction(n.activationValue);
						hiddenNeurons.add(n);
					}

				}
						
				else
				{
					for (int l = 0; l< inputNeurons.size(); l++)
					{
						n = inputNeurons.get(l);
						n.activationValue = trainData[i][l];
					}

					for (int l = 0; l< hiddenNeurons.size(); l++)
					{
						n = hiddenNeurons.get(l);
						n.activationValue = activationFunction(n);
						n.derivativeValue = derivativeActivationFunction(n.activationValue);
					}

					for (int l = 0; l< outputNeurons.size(); l++)
					{
						n = outputNeurons.get(l);
						n.activationValue = activationFunction(n);
						n.derivativeValue = derivativeActivationFunction(n.activationValue);
					}							

				}
				
				double target = trainData[i][trainData[i].length-1];
				// Calculate Delta i
				for (int j = 0; j < outputNeurons.size(); j++)
				{ 
					n = outputNeurons.get(j);
					n.deltaW = (target - n.activationValue) * n.derivativeValue;
				}

				//Calculate Delta j
				for (int j = 0; j< hiddenNeurons.size(); j++)
				{
					n = hiddenNeurons.get(j);
					n.deltaW = n.derivativeValue * calculateErrorHidden();
				}


				// Update weights for Wij
				for (int j = 0; j< outputNeurons.size(); j++)
				{
					n = outputNeurons.get(j);
					for (int k=0; k<n.weights.size(); k++){
						Synapsis s = n.weights.get(k);
						s.value += learningRate * s.parent.activationValue * n.deltaW;
					}
				}

			}
			currentIteration++;
		}while(currentIteration != maxIterations);

}

public HashSet<Sample> setExamples(int numberOfExamples){
	// We are making values inside the circle area (target == -1)
	HashSet<Sample> samples = new HashSet<Sample>();
	while (samples.size() < numberOfExamples/2)
	{
		double radioX = Math.random()*6;
		double radioY = Math.random()*6;
		double alfaX = Math.toRadians(Math.random()*360);
		double alfaY = Math.toRadians(Math.random()*360);
		Sample sample = new Sample(radioX +Math.cos(alfaX) + 10, radioY + Math.cos(alfaY) + 10, -1);
		samples.add(sample);
		System.out.println(" Sample "+ sample.x + "   "+ sample.y+ "   " +sample.target);
		
	}

	int dummy = 0;
	while (samples.size() < numberOfExamples )
	{
		//Outside the circle area
		double x = Math.random()*5;
		double y = Math.random()*5;
		double x1 = 16.0 + Math.random()*5;
		double y1 = 16.0 + Math.random()*5;

		switch (dummy)
		{
			case 0:
				Sample sample = new Sample(x, y, 1);
				samples.add(sample);
				System.out.println(" Sample "+ sample.x + "   "+ sample.y+ "   " +sample.target);
			break;
			case 1:
				sample = new Sample(x,y1,1);
				samples.add(sample);
				System.out.println(" Sample "+ sample.x + "   "+ sample.y+ "   " +sample.target);
			break;
			case 2:
				sample = new Sample(x1,y,1);
				samples.add(sample);
				System.out.println(" Sample "+ sample.x + "   "+ sample.y+ "   " +sample.target);
			break;
			case 3:
				sample = new Sample(x1,y1,1);
				samples.add(sample);
				System.out.println(" Sample "+ sample.x + "   "+ sample.y+ "   " +sample.target);
			break;
		}
		dummy = ( dummy == 3 ? 0 : dummy + 1);
	}	
	return samples;
}


public static void main(String[] args) 
{
    String filename = "";

    if (args.length < 1)
    {
        System.out.println("\nFaltan argumentos.\n");
    }

    filename = args[0];
    
    System.out.println("Filename: " + filename);
    
    readData(filename);

	//BackPropagationL bp = new BackPropagationL(2, 3, 2, 4);

	//System.out.println("Weights");
}


}