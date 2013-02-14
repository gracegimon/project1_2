import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Vector;

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


	// Saves training data in a global array
	private static void readData(String filename) 
	{
        int numberOfData = 0;
        if (filename.equals("datos_r6_n500.txt"))
            numberOfData = 500;
        else if (filename.equals("datos_r6_n1000.txt"))
            numberOfData = 1000;
        else if (filename.equals("datos_r6_n2000.txt"))
            numberOfData = 2000;
            
    
		try 
        {
			BufferedReader br_train = new BufferedReader(new FileReader(filename));
			String str;
			//int numberOfExamples, numberOfWeights, i;
            int i;
            String[] strArr;
			// Reads first line with the info
			/*str = br_train.readLine();
			String[] strArr = str.split(" ");
			numberOfExamples = Integer.parseInt(strArr[0]);
			numberOfWeights = Integer.parseInt(strArr[1]);*/

			// Initializes the Examples and Weights arrays
			//trainData = new double[numberOfExamples][3];      

            double[][] data = new double[numberOfData][3];
            
			i = 0;

			// Reads examples
			while ( (str = br_train.readLine()) != null )
			{
				strArr = str.split(" ");
				double[] lineData = new double[3];
                for (int j = 0; j < strArr.length; j++)
				{
					data[i][j] = Double.parseDouble(strArr[j]);                      
				}
				i++;
			}

            System.out.println("\nData:\n");
            for (int ii = 0; ii < data.length; ii++)
            {
                System.out.println("X: " + data[ii][0] + " | Y: " + data[ii][1] + " | Target: " + data[ii][2]);
            }
            
			br_train.close();
		} catch(IOException e) {
			e.printStackTrace();
		}
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