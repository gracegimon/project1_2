import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Vector;
import java.util.HashSet;

/*
	Backpropagation for a neural network
	of two layers
*/
class BackPropagationL{

	static double[][] trainData;
	Vector<Neuron> inputNeurons;
	Vector<Neuron> hiddenNeurons;
	Vector<Neuron> outputNeurons;
	static double [] outputs;

	static double[][] errors; // Errors for each weight 
	/*
		Example:  neuron[j] in the layer[i]
	*/
	int numberOfNeurons = 2; // Number of Neurons in the hidden layer
	int numberOfInputs = 2; // Number of neurons in the input layer
	int numberOfOutputs = 1; // Number of neurons in the output layer
	double error = 0;
	double output = 0;
	int maxIterations = 100;

	public BackPropagationL(int numberOfWeights, int numberOfNeurons, int numberOfOutputs, int numberOfExamples){
		//trainData = new double[numberOfExamples][numberOfWeights];
		numberOfNeurons = numberOfNeurons;
		numberOfOutputs = numberOfOutputs;
		numberOfInputs = numberOfWeights;
		inputNeurons = new Vector<Neuron>();
		hiddenNeurons = new Vector<Neuron>();
		outputNeurons = new Vector<Neuron>();

	}

// Sigmoid Function
private double sigmoidFunc(double value)
{
	return (1 / (1 + Math.exp(-1 * value)));
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
	
	return sigmoidFunc(activation); 
}

private double derivativeActivationFunction(double value )
{
	//return (1.0 - (Math.pow(value,2)));
	return (value * (1 - value));
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

				if (i == 0 && currentIteration == 0) //no neuron created
				{
					//System.out.println("Input Neurons size " +inputNeurons.size() );
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
						n.error += Math.pow((trainData[i].length-1) - n.activationValue, 2);
						n.derivativeValue = derivativeActivationFunction(n.activationValue);
						outputNeurons.add(n);
					}

				}
						
				else
				{

					if (currentIteration > 0) 
					{
						for (int l = 0; l< outputNeurons.size(); l++)
                   		{
                           		 n = outputNeurons.get(l);
                    	//      System.out.println("Activation F outputNeurons " + n.activationValue);
            	                n.error = 0;
                        }
		
					}

					//System.out.println("Input Neurons size " +inputNeurons.size() );
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
					//	System.out.println("Activation F outputNeurons " + n.activationValue);
						n.error += Math.pow((trainData[i].length-1) - n.activationValue, 2);
						n.derivativeValue = derivativeActivationFunction(n.activationValue);
						outputs[i] = n.activationValue;
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
			Neuron n;
			for (int l = 0; l< outputNeurons.size(); l++)
            {
                n = outputNeurons.get(l);
                System.out.println("Error de neurona "+ l + " : "+  n.error);
			}

			currentIteration++;
		}while(currentIteration != maxIterations);

		/* Write results into a file */
		writeData();

}

public HashSet<Sample> setExamples(int numberOfExamples, int gridSize){

    double prop = gridSize / 5;
	double radius = (gridSize - (prop*2)) / 2; 

	// We are making values inside the circle area (target == -1)
	HashSet<Sample> samples = new HashSet<Sample>();
	while (samples.size() < numberOfExamples/2)
	{
		double radioX = Math.random()*radius;
		double radioY = Math.random()*radius;
		double alfaX = Math.toRadians(Math.random()*360);
		double alfaY = Math.toRadians(Math.random()*360);
		Sample sample = new Sample(radioX +Math.cos(alfaX) + gridSize/2, radioY + Math.cos(alfaY) + gridSize/2, -1);
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

			//double[][] data
			trainData = new double[numberOfData][3];
			outputs = new double[numberOfData];
			            
			i = 0;

			// Reads examples
			while ( (str = br_train.readLine()) != null )
			{
			strArr = str.split(" ");
			double[] lineData = new double[3];
			                for (int j = 0; j < strArr.length; j++)
			{
			trainData[i][j] = Double.parseDouble(strArr[j]);
			}
			i++;
			}

			            
		     /*       for (int ii = 0; ii < trainData.length; ii++)
		            {
		                System.out.println("X: " + trainData[ii][0] + " | Y: " + trainData[ii][1] + " | Target: " + trainData[ii][2]);
		            }
		       */    
			       System.out.println("\nData:\n OK"); 
			br_train.close();
		} catch(IOException e) {
			e.printStackTrace();
		}
	}


public void writeData()
{
	FileWriter fichero = null;
    PrintWriter pw = null;
    try
    {
        fichero = new FileWriter("resultados.txt");
        pw = new PrintWriter(fichero);
        pw.println("x   y   output");
        for (int i = 0; i < trainData.length; i++)
        {
				pw.println(trainData[i][0]+"\t"+trainData[i][1]+"\t"+outputs[i]);
          
        }

    } catch (Exception e) {
        e.printStackTrace();
    } finally {
       try {
       // Nuevamente aprovechamos el finally para 
       // asegurarnos que se cierra el fichero.
       if (null != fichero)
          fichero.close();
       } catch (Exception e2) {
          e2.printStackTrace();
       }
    }

}


public static void main(String[] args) 
{
    String filename = "";

    if (args.length < 1)
    {
        System.out.println("\nFaltan argumentos.\n");
        System.exit(-1);
    }

    filename = args[0];
    
    System.out.println("Filename: " + filename);
    
    readData(filename);

	BackPropagationL bp = new BackPropagationL(2, 3, 1,trainData.length);
	bp.backPropagation(0.05);

	//System.out.println("Weights");
}


}
