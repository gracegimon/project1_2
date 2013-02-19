import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Vector;
import java.util.Iterator;
import java.util.HashSet;

//Probar mas de dos capas

/*
	Backpropagation for a neural network
	of two layers
*/
class BackPropagationL{

	static double[][] trainData;
	Vector<Neuron> inputNeurons;
	Vector<Neuron> hiddenNeurons;
	Vector<Neuron> hiddenNeurons2;
	Vector<Neuron> outputNeurons;
	static double [] outputs;

	static double[][] errors; // Errors for each weight 
	/*
		Example:  neuron[j] in the layer[i]
	*/
	int numberOfNeurons; // Number of Neurons in the hidden layer
	int numberOfInputs; // Number of neurons in the input layer
	int numberOfOutputs; // Number of neurons in the output layer
	int numberOfLayers;
	double error = 0;
	double output = 0;
	int maxIterations = 100;

	public BackPropagationL(int numberOfWeights, int numberOfNeurons, int numberOfOutputs, int numberOfExamples, int numberOfLayers){
		//trainData = new double[numberOfExamples][numberOfWeights];
		this.numberOfNeurons = numberOfNeurons;
		this.numberOfOutputs = numberOfOutputs;
		this.numberOfInputs = numberOfWeights;
		this.numberOfLayers = numberOfLayers;
		this.inputNeurons = new Vector<Neuron>();
		this.hiddenNeurons = new Vector<Neuron>();
		this.outputNeurons = new Vector<Neuron>();
		this.hiddenNeurons2 = new Vector<Neuron>();

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
		activation += neuron.weight0;

	
	return sigmoidFunc(activation); 
}

private double derivativeActivationFunction(double value )
{
	//return (1.0 - (Math.pow(value,2)));
	return (value * (1 - value));
}

private double calculateErrorHidden(int layer) 
{ 
	double error = 0.0;
	if (layer == 1) //First hidden layer
		for (int k = 0; k <outputNeurons .size(); k++)
		{
			Neuron n = outputNeurons.get(k);
			for (int j = 0; j < n.weights.size(); j++)
			{
				error += n.weights.get(j).value * n.deltaW;
			}	
		}
	else{ //Second hidden layer
		for (int k = 0; k < hiddenNeurons2.size(); k++)
		{
			Neuron n = hiddenNeurons2.get(k);
			for (int j = 0; j < n.weights.size(); j++)
			{
				error += n.weights.get(j).value * n.deltaW;
			}	
		}
	}
	return error;
}

private double Random(){
	double rand = Math.random() * 2 -1;
	System.out.println(" Peso, random: "+rand);
	return rand;
}

/** 
*   @param learningRate : learning rate
*/
private void backPropagation(double learningRate)
{

		int currentIteration = 0;
		
		do
		{
			System.out.println("******************************");
			System.out.println("ITERACION  " + currentIteration);
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


					// if number of layers greater

					if (numberOfLayers == 1){

						//Calculate activation function for hidden
						for (int j = 0; j < numberOfNeurons; j++)
						{
							n = new Neuron();
							for (int k = 0; k < inputNeurons.size(); k++){ // Weights -1, 1
								n.AddSynapsis(new Synapsis(Random(), inputNeurons.get(k)));
							//	System.out.println("Pesos neurona "+ k +" : " + n.weights.get(k).value);
							}

							n.activationValue = activationFunction(n);
						//	System.out.println("Activation Value "+ j+ "  : "+n.activationValue);
							n.derivativeValue = derivativeActivationFunction(n.activationValue); 
						//	System.out.println("Derivative Value "+ j+ "  : "+n.derivativeValue);
							hiddenNeurons.add(n);

						}

						for (int j = 0; j < numberOfOutputs; j++)
						{
							n = new Neuron();
							for (int k = 0; k < hiddenNeurons.size(); k++){
								n.AddSynapsis(new Synapsis(Random(), hiddenNeurons.get(k)));
							}

							n.activationValue = activationFunction(n);
							System.out.println("Target - output "+trainData[i][trainData[i].length-1]+ "  "+n.activationValue);
							n.error += Math.pow((trainData[i][trainData[i].length-1]) - n.activationValue, 2);
							n.derivativeValue = derivativeActivationFunction(n.activationValue);
							outputNeurons.add(n);
						}

					}
					else{ // MORE LAYERS

						//Calculate activation function for hidden
						//First layer
						int neuronsFirstLayer = (int) Math.floor(numberOfNeurons/2);
						System.out.println("Number of first layer " + neuronsFirstLayer);

						for (int j = 0; j < neuronsFirstLayer; j++)
						{
							n = new Neuron();
							for (int k = 0; k < inputNeurons.size(); k++){ // Weights -1, 1
								n.AddSynapsis(new Synapsis(Random(), inputNeurons.get(k)));
								System.out.println("Pesos neurona "+ k +" : " + n.weights.get(k).value);
							}

							n.activationValue = activationFunction(n);
							System.out.println("Activation Value "+ j+ "  : "+n.activationValue);
							n.derivativeValue = derivativeActivationFunction(n.activationValue); 
							System.out.println("Derivative Value "+ j+ "  : "+n.derivativeValue);
							hiddenNeurons.add(n);

						}

						//Second Layer
						System.out.println("Number of second layer" + (numberOfNeurons - neuronsFirstLayer));
						for (int j = 0; j < numberOfNeurons - neuronsFirstLayer; j++)
						{
							n = new Neuron();
							for (int k = 0; k < hiddenNeurons.size(); k++){ // Weights -1, 1
								n.AddSynapsis(new Synapsis(Random(), hiddenNeurons.get(k)));
							//	System.out.println("Pesos neurona "+ k +" : " + n.weights.get(k).value);
							}

							n.activationValue = activationFunction(n);
							System.out.println("Activation Value "+ j+ "  : "+n.activationValue);
							n.derivativeValue = derivativeActivationFunction(n.activationValue); 
							System.out.println("Derivative Value "+ j+ "  : "+n.derivativeValue);
							hiddenNeurons2.add(n);

						}


						for (int j = 0; j < numberOfOutputs; j++)
						{
							n = new Neuron();
							for (int k = 0; k < hiddenNeurons2.size(); k++){
								n.AddSynapsis(new Synapsis(Random(), hiddenNeurons2.get(k)));
							}

							n.activationValue = activationFunction(n);
							n.error += Math.pow((trainData[i][trainData[i].length-1]) - n.activationValue, 2);
							n.derivativeValue = derivativeActivationFunction(n.activationValue);
							outputNeurons.add(n);
						}

					}

				}
						
				else // Next iterations
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

					//Only for more layers
				
					for (int l = 0; l< hiddenNeurons2.size(); l++)
					{
						n = hiddenNeurons2.get(l);
						n.activationValue = activationFunction(n);
						n.derivativeValue = derivativeActivationFunction(n.activationValue);
					}


					for (int l = 0; l< outputNeurons.size(); l++)
					{
						n = outputNeurons.get(l);
						n.activationValue = activationFunction(n);
					//	System.out.println("Activation F outputNeurons " + n.activationValue);
						n.error += Math.pow((trainData[i][trainData[i].length-1]) - n.activationValue, 2);
					//	System.out.println("Target - output "+trainData[i][trainData[i].length-1]+ "  "+n.activationValue);

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
				if (numberOfLayers == 1){
					for (int j = 0; j< hiddenNeurons.size(); j++)
					{
						n = hiddenNeurons.get(j);
						n.deltaW = n.derivativeValue * calculateErrorHidden(1);
					}				
				}
				else{
					// FOR MORE LAYERS ONLY
					for (int j = 0; j< hiddenNeurons2.size(); j++)
					{
						n = hiddenNeurons2.get(j);
						n.deltaW = n.derivativeValue * calculateErrorHidden(1);
					}

					for (int j = 0; j< hiddenNeurons.size(); j++)
					{
						n = hiddenNeurons.get(j);
						n.deltaW = n.derivativeValue * calculateErrorHidden(2);
					}		

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
				if (numberOfLayers > 1){
				
					for (int j = 0; j< hiddenNeurons2.size(); j++)
					{
						n = hiddenNeurons2.get(j);
						for (int k=0; k<n.weights.size(); k++){
							Synapsis s = n.weights.get(k);
							s.value += learningRate * s.parent.activationValue * n.deltaW;
						}
					}

				}
				for (int j = 0; j< hiddenNeurons.size(); j++)
				{
					n = hiddenNeurons.get(j);
					for (int k=0; k<n.weights.size(); k++){
						Synapsis s = n.weights.get(k);
						s.value += learningRate * s.parent.activationValue * n.deltaW;
					}
				}

			}
			Neuron n1;
			for (int l = 0; l< outputNeurons.size(); l++)
            {
                n1 = outputNeurons.get(l);
                System.out.println("Error de neurona "+ l + " : "+  n1.error);
			}

			currentIteration++;
		}while(currentIteration != maxIterations);

		/* Write results into a file */
		writeData();

}

public void generateExamples(int numberOfExamples){

	HashSet<Sample> insideCircle = new HashSet<Sample>();
	HashSet<Sample> outsideCircle = new HashSet<Sample>();
	while (insideCircle.size() < numberOfExamples/2 || outsideCircle.size() < numberOfExamples/2 ){
		double x = Math.random()*20;
		double y = Math.random()*20;
		Sample s;
		
		if ( Math.pow((x - 10),2) + Math.pow((y-10),2) < 36){
			if(insideCircle.size() < numberOfExamples/2){
			s = new Sample(x,y,0);
			insideCircle.add(s);
			}
		}
		else{
			if ( outsideCircle.size() < numberOfExamples/2){
							s = new Sample(x,y,1);
							outsideCircle.add(s);
			}
			
		}
	}

Iterator<Sample> it = insideCircle.iterator();
	while( it.hasNext()){
		System.out.println("INSIDE");
			Sample s = it.next();
			System.out.print(s.x+" "+s.y+"  "+s.target);
			System.out.println("  Distancia "+Math.sqrt(Math.pow(s.x-10,2)+ Math.pow(s.y-10,2) ) + "  ");
		}
		
		it = outsideCircle.iterator();
		
		while( it.hasNext()){
			System.out.println("OUTSIDE");
			Sample s = it.next();
			System.out.print(s.x+" "+s.y+"  "+s.target);
			System.out.println("  Distancia "+Math.sqrt(Math.pow(s.x-10,2)+ Math.pow(s.y-10,2) ) + "  ");
		}
	writeExamples(insideCircle,outsideCircle);
	

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
	        int numberOfTestData = 0;
	        if (filename.equals("datos_r6_n500.txt"))
	            numberOfData = 500;
	        else if (filename.equals("datos_r6_n1000.txt"))
	            numberOfData = 1000;
	        else if (filename.equals("datos_r6_n2000.txt"))
	            numberOfData = 2000;
	        else if(filename.equals("training3_50.txt"))
	        	numberOfData = 170;
	        else if(filename.equals("test3_50.txt"))
	        	numberOfTestData = 171;
	        else if(filename.equals("training3_60.txt"))
	        	numberOfData = 204;
	        else if(filename.equals("test3_60.txt"))
	        	numberOfTestData = 137;
	        else if(filename.equals("training3_70.txt"))
	        	numberOfData = 238;
	        else if(filename.equals("test3_70.txt"))
	        	numberOfTestData = 103;
	        else if(filename.equals("training3_80.txt"))
	        	numberOfData = 272;
	        else if(filename.equals("test3_80.txt"))
	        	numberOfTestData = 69;
	        else if(filename.equals("training3_90.txt"))
	        	numberOfData = 306;
	        else if(filename.equals("test3_90.txt"))
	        	numberOfTestData = 35;

		try
		{
			BufferedReader br_train = new BufferedReader(new FileReader(filename));
			String str;
			//int numberOfExamples, numberOfWeights, i;
		    int i, numberOfAttributes = 0;
			// Reads first line to know how many attributes are
			str = br_train.readLine();
			String[] strArr = str.split(" ");
			numberOfAttributes = strArr.length;

			// Initializes the Examples and Weights arrays
			//trainData = new double[numberOfExamples][3];

			//double[][] data
			trainData = new double[numberOfData][numberOfAttributes];
			outputs = new double[numberOfData];
			            
			i = 0;

			// Reads examples
			while ( (str = br_train.readLine()) != null )
			{
			strArr = str.split(" ");
			double[] lineData = new double[numberOfAttributes];
			                for (int j = 0; j < strArr.length; j++)
			{
				trainData[i][j] = Double.parseDouble(strArr[j]);
			}
			i++;
			}
  
            System.out.println("\nData:\n OK"); 
			br_train.close();
		} catch(IOException e) {
			e.printStackTrace();
		}
	}

public void writeExamples(HashSet<Sample> insideCircle, HashSet<Sample> outsideCircle){
	FileWriter file = null;
    PrintWriter pw = null;
    try
    {
        file = new FileWriter("GeneratedExamples.csv");
        pw = new PrintWriter(file);


		Iterator<Sample> it = insideCircle.iterator();
		
		while( it.hasNext()){
			Sample s = it.next();
			pw.println(s.x+","+s.y+","+s.target);
		}
		
		it = outsideCircle.iterator();
		
		while( it.hasNext()){
			Sample s = it.next();
			pw.println(s.x+","+s.y+","+s.target);
		}

	}catch (Exception e) {
        e.printStackTrace();
    } finally {
       try {

       if (null != file)
          file.close();
       } catch (Exception e2) {
          e2.printStackTrace();
       }
    }
}

public void writeData()
{
	FileWriter file = null;
    PrintWriter pw = null;
    try
    {
        file = new FileWriter("resultados.txt");
        pw = new PrintWriter(file);
        pw.println("x   y   output");
        int good = 0;
        for (int i = 0; i < trainData.length; i++)
        {
        		if (outputs[i] >= 0.5)
					outputs[i] = 1;
				else
					outputs[i] = 0;

				pw.println(trainData[i][0]+"\t"+trainData[i][1]+"\t"+outputs[i]);
				if (outputs[i] ==trainData[i][trainData[i].length-1] )
					good++;

          
        }
        pw.println(good);

    } catch (Exception e) {
        e.printStackTrace();
    } finally {
       try {
       if (null != file)
          file.close();
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

	BackPropagationL bp = new BackPropagationL(2, 4, 1,trainData.length,1);
	//bp.backPropagation(0.05);
	bp.generateExamples(20);

	//System.out.println("Weights");
}


}
