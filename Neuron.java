import java.util.Vector;

public class Neuron{
	public double activationValue;
	public double deltaW;
	public double derivativeValue;
	public Vector<Synapsis> weights; //Weights that are received. row: each neuron, col: each weight
	public int type; // 0: input, 1:hidden, 2: output
	public double error; // NUll for hidden and input


	public Neuron()
	{

		weights = new Vector<Synapsis>();
		
	}

	public void AddSynapsis(Synapsis s)
	{
		weights.add(s);
	}


}
