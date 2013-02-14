
public class Neuron{
	public double activationValue;
	public double deltaW;
	public double derivativeValue;
	public double[][] weights; //Weights that are received
	public int type; // 0: input, 1:hidden, 2: output

	public Neuron(){

	}

	public Neuron(int numberOfInputs)
	{
		weights = new double[numberOfInputs];
	}


}