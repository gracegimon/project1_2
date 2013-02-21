/*
 * Proyecto 1 Parte II
 * Redes Neuronales
 *
 * Autores:
 *    - Grace Gimon
 *    - Oriana Baldizan
 *    - Christian Chomiak
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Vector;
import java.util.Iterator;
import java.util.HashSet;


/*
	Backpropagation for a neural network
	of two layers
*/
class BackPropagationL{

	static double[][] trainData;
	static double[][] testData;
	static double[] normMatrix;
	Vector<Neuron> inputNeurons;
	Vector<Neuron> hiddenNeurons;
	Vector<Neuron> hiddenNeurons2;
	Vector<Neuron> outputNeurons;
	static double [] outputs;

	static double[] errors; // Errors

	/*
		Example:  neuron[j] in the layer[i]
	*/
	int numberOfNeurons; // Number of Neurons in the hidden layer
	int numberOfInputs; // Number of neurons in the input layer
	int numberOfOutputs; // Number of neurons in the output layer
	int numberOfLayers;
	double error = 0;
	double output = 0;
	int maxIterations;

	public BackPropagationL(int numberOfWeights, int numberOfNeurons, int numberOfOutputs, int numberOfExamples, int numberOfLayers,int maxIterations){
		//trainData = new double[numberOfExamples][numberOfWeights];
		this.numberOfNeurons = numberOfNeurons;
		this.numberOfOutputs = numberOfOutputs;
		this.numberOfInputs = numberOfWeights;
		this.numberOfLayers = numberOfLayers;
		this.inputNeurons = new Vector<Neuron>();
		this.hiddenNeurons = new Vector<Neuron>();
		this.outputNeurons = new Vector<Neuron>();
		this.hiddenNeurons2 = new Vector<Neuron>();
		this.maxIterations = maxIterations;
		this.errors = new double[maxIterations];

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
	double error2 = 0.0;
	if (layer == 1) //First hidden layer
		for (int k = 0; k <outputNeurons .size(); k++)
		{
			Neuron n = outputNeurons.get(k);
			for (int j = 0; j < n.weights.size(); j++)
			{
				error2 += n.weights.get(j).value * n.deltaW;
			}	
		}
	else{ //Second hidden layer
		for (int k = 0; k < hiddenNeurons2.size(); k++)
		{
			Neuron n = hiddenNeurons2.get(k);
			for (int j = 0; j < n.weights.size(); j++)
			{
				error2 += n.weights.get(j).value * n.deltaW;
			}	
		}
	}
	return error2;
}

private double Random(){
	double rand = (Math.random()-0.5)/10;
	System.out.println(" Peso, random: "+rand);
	return rand;
}  

/** 
*   @param learningRate : learning rate
*   @param type : 0 - Training. 1 - Testing
*/
private void backPropagation(double learningRate, int type)
{

		int currentIteration = 0;

		for (int i = 0; i<errors.length; i++){
			errors[i] = 0;
		}
		
		do
		{
		//	System.out.println("******************************");
		//	System.out.println("ITERACION  " + currentIteration);
			for (int i = 0; i< trainData.length; i++) //For each example
			{
				Neuron n;

				if (i == 0 && currentIteration == 0 && type ==0) //no neuron created
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
					else
					{ // MORE LAYERS

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

					if (i == 0)
						for (int l = 0; l< outputNeurons.size(); l++)
	     				{
							n = outputNeurons.get(l);
							n.error = 0;
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
				
					for (int l = 0; l < hiddenNeurons2.size(); l++)
					{
						n = hiddenNeurons2.get(l);
						n.activationValue = activationFunction(n);
						n.derivativeValue = derivativeActivationFunction(n.activationValue);
					}


					for (int l = 0; l < outputNeurons.size(); l++)
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
				if (type == 0)
				{
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
					else
					{
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
			}
			Neuron n1;
			for (int l = 0; l < outputNeurons.size(); l++)
      		{
                n1 = outputNeurons.get(l);
                errors[currentIteration] = n1.error / trainData.length;
                System.out.println("error "+ n1.error  / trainData.length);
			}

			currentIteration++;
			//if (type == 1)
			//	System.out.println("cIt: " + currentIteration + " :: Max: "+ maxIterations);
			
		} while(currentIteration != maxIterations);

		/* Write results into a file */
		writeData(trainData[0].length - 1);

}

public static void generateExamples(int numberOfExamples){

	HashSet<Sample> insideCircle = new HashSet<Sample>();
	HashSet<Sample> outsideCircle = new HashSet<Sample>();
	while (insideCircle.size() < numberOfExamples/2 || outsideCircle.size() < numberOfExamples/2 ){
		float x = (float)Math.random()*20;
		float y = (float)Math.random()*20;
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

/*
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
*/
	writeExamples(insideCircle,outsideCircle);


}

// READ and WRITE FILES functions

	// Normalize data
	private static void normalizeData()
	{
		double aux;
		int numberOfAttributes = trainData[0].length;
		//normMatrix = new double[numberOfAttributes];

		// Calculates the average for each attribute
		for (int i=0; i < normMatrix.length; i++)
		{
				normMatrix[i] = normMatrix[i] / trainData.length;
		}

		// Normalize the data 
			for (int i=0; i < trainData.length; i++) 
			{
				for (int j=0; j < numberOfAttributes; j++)
				{
					//System.out.println("normMatrix: " + normMatrix[j]);
					aux = trainData[i][j] / normMatrix[j]; 
					if (j < numberOfAttributes-1)
						trainData[i][j] = aux;
				}
			}
	}


	// Saves training data in a global array
	private static void readData2(String filename) 
	{
			int numberOfData = 0;

			// Experiment 2 files
      if(filename.equals("training3_50.txt"))
          numberOfData = 170;
      else if(filename.equals("training3_60.txt"))
	        numberOfData = 204;
      else if(filename.equals("training3_70.txt"))
	       	numberOfData = 238;
      else if(filename.equals("training3_80.txt"))
	       	numberOfData = 272;
      else if(filename.equals("training3_90.txt"))
	       	numberOfData = 306;

      else if(filename.equals("test3_50.txt"))
	       	numberOfData = 171;
	    else if(filename.equals("test3_60.txt"))
	       	numberOfData = 137;
	    else if(filename.equals("test3_70.txt"))
	       	numberOfData = 103;
	    else if(filename.equals("test3_80.txt"))
	       	numberOfData = 69;
	    else if(filename.equals("test3_90.txt"))
	       	numberOfData = 35;

			try // Reads and saves content
		{
			BufferedReader br = new BufferedReader(new FileReader(filename));
			String str;
		  int i = 0, numberOfAttributes = 0;

			// Reads first line to know how many attributes are
			str = br.readLine();
			String[] strArr = str.split(",");
			numberOfAttributes = strArr.length;

			trainData = new double[numberOfData][numberOfAttributes];
			normMatrix = new double[numberOfAttributes];
			
	

			outputs = new double[numberOfData];
			for (int j = 0 ; j < strArr.length; j++)
			{
						trainData[i][j] = Double.parseDouble(strArr[j]);
						if (j < strArr.length-1) // Last value is normalized
						{
							normMatrix[j] += Double.parseDouble(strArr[j]);
						}
			}

			// Read examples
			while ( (str = br.readLine()) != null )
			{
			    i++;
			    strArr = str.split(",");
			    for (int j = 0; j < strArr.length; j++)
			    {
				   			trainData[i][j] = Double.parseDouble(strArr[j]);

								if (j < strArr.length-1) // Last value is normalized
									normMatrix[j] += Double.parseDouble(strArr[j]);
			    }

			}

  
      		System.out.println("\nData:\n OK"); 
			br.close();

			// Normalize data
			normalizeData();

			for(i=0; i < numberOfData; i++)
			{
				for (int j=0; j < numberOfAttributes; j++)
				{
				//	System.out.println("Elem " + i + " "+ j+ " "+ trainData[i][j]);
				}
			}

			

		} catch(IOException e) {
			e.printStackTrace();
		}
	}


  // Saves training and testing data in global arrays
	private static void readData1(String filename)
	{
	    int numberOfData = 0;

      // Experiment 1 files
      	if (filename.equals("datos_r6_n500.txt"))
	        numberOfData = 500;
      	else if (filename.equals("datos_r6_n1000.txt"))
	        numberOfData = 1000;
	    else if (filename.equals("datos_r6_n2000.txt"))
	        numberOfData = 2000;
		else if (filename.equals("GeneratedExamples500.txt"))
				numberOfData = 500;
		else if (filename.equals("GeneratedExamples1000.txt"))
				numberOfData = 1000;
		else if (filename.equals("GeneratedExamples2000.txt"))
				numberOfData = 2000;
		else if (filename.equals("GeneratedExamples10000.txt"))
				numberOfData = 10000;
			else {
				System.out.println("FILE NOT FOUND!!!!");
			}

		try // Reads and saves content
		{
			BufferedReader br = new BufferedReader(new FileReader(filename));
			String str;
		    int i = 0, numberOfAttributes = 0;

			// Reads first line to know how many attributes are
			str = br.readLine();
			String[] strArr = str.split(" ");
			numberOfAttributes = strArr.length;

			trainData = new double[numberOfData][numberOfAttributes];
			
			outputs = new double[numberOfData];

			for (int j = 0 ; j < strArr.length; j++)
			{
				trainData[i][j] = Double.parseDouble(strArr[j]);
			}

			// Read examples
			while ( (str = br.readLine()) != null )
			{
			    i++;
			    strArr = str.split(" ");
			    for (int j = 0; j < strArr.length; j++)
			    {
		   			trainData[i][j] = Double.parseDouble(strArr[j]);
			    }

			}
  
      		System.out.println("\nData:\n OK"); 
			br.close();
		} catch(IOException e) {
			e.printStackTrace();
		}
	}


  // Write the generated examples in a file
  public static void writeExamples(HashSet<Sample> insideCircle, HashSet<Sample> outsideCircle)
{
	FileWriter file = null;
    PrintWriter pw = null;
    try
    {
		int numberOfExamples = insideCircle.size() + outsideCircle.size();
        file = new FileWriter("GeneratedExamples"+numberOfExamples +".txt");
        pw = new PrintWriter(file);

        // Write first line
        //pw.println("X Y CLASS");

        // Write points inside the circle
        Iterator<Sample> it = insideCircle.iterator();
		    while( it.hasNext()){
			      Sample s = it.next();
			      pw.println(s.x+" "+s.y+" "+s.target);
		    }
		
        // Write points outside the circle
		    it = outsideCircle.iterator();
    		while( it.hasNext()){
		        Sample s = it.next();
			      pw.println(s.x+" "+s.y+" "+s.target);
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


// Write the obtained results in a file
public void writeData(int numberOfAttributes)
{
	  FileWriter file = null;
    PrintWriter pw = null;
    try
    {
        file = new FileWriter("resultados"+(error++ )+".txt");
        pw = new PrintWriter(file);
        pw.println("X Y CLASS");
        int good = 0;
        for (int i = 0; i < trainData.length; i++)
        {
        		if (outputs[i] >= 0.5)
					outputs[i] = 1;
				else
					outputs[i] = 0;
				for (int j = 0; j<numberOfAttributes; j++){
					pw.print(trainData[i][j]+" ");

				}
				pw.println(outputs[i]);
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


// MAIN FUNCTION

/* Number of Neurons hidden layer
* Number of Max Iterations 
* Type of experiment ( 1: Examples given through console, 2 : Generated Examples, 3: Liver Disorder)
* Params of the type of experiment
* 1: Training file
* 2: Number of Examples
* 3: Training file, Testing file
*
*/

public static void main(String[] args) 
{
    String trainFile = "", testFile = "";
		BackPropagationL bp;

    if (args.length < 3)
    {
        System.out.println("\nPor favor indique el num de neuronas hidden,num de iteraciones, num de experimento y los archivos correspondientes.\n");
        System.exit(-1);
    }

    else if (args[3] == "2" && args.length < 3) 
	{
		System.out.println("\nPor favor indique el archivo de prueba.\n");
      	System.exit(-1);
    }
    System.out.println("Argumentos ");
    System.out.println(args[0]+" "+args[1]+" "+args[2]+" "+args[3]);
    int neurons = Integer.parseInt(args[0]);
    int iterations = Integer.parseInt(args[1]);
		if (args[2].equals("1")) 
		{
			trainFile = args[3];
    		readData1(trainFile);	

			// Training
			bp = new BackPropagationL(2, neurons, 1,trainData.length,1,iterations);
			bp.backPropagation(0.05,0);

			// Testing
			generateExamples(10000);  // Funciona pero no para 10000
			bp.maxIterations = 1;
			readData1("GeneratedExamples10000.txt");
			bp.backPropagation(0.05,1);
		}
		else if (args[2].equals("2"))
		{
			System.out.println("Argumentos : ");
			System.out.println(" "+args[3]);
				// Training
				generateExamples(Integer.parseInt(args[3]));
				readData1("GeneratedExamples"+ Integer.parseInt(args[3]) +".txt");
				bp = new BackPropagationL(2, neurons, 1,trainData.length,1,iterations);
				bp.backPropagation(0.05,0);

				// Testing
				generateExamples(1000);
				readData1("GeneratedExamples1000.txt");	
				bp.maxIterations = 1;	
				bp.backPropagation(0.05,1);
		}
    else if (args[2].equals("3"))
    {
			// Training
			trainFile = args[3];
			readData2(trainFile);
			bp = new BackPropagationL(2, neurons, 1,trainData.length,1,iterations);
			bp.backPropagation(0.05,0);

			XYChart chart = new XYChart();
			chart.getChart(errors,neurons+"_"+iterations+"_"+trainData.length);

				// Testing
       		testFile = args[4];
        	readData2(testFile);
        	bp.maxIterations = 1;
			bp.backPropagation(0.05,1);

			
    }

}


}

