import java.util.Vector;

public class Neuron{
	public double activationValue;
	public double deltaW;
	public double derivativeValue;
	public Vector<Synapsis> weights; //Weights that are received. row: each neuron, col: each weight
	public double weight0;
	public int type; // 0: input, 1:hidden, 2: output
	public double error; // NUll for hidden and input


	public Neuron(boolean ultima)
	{

		weights = new Vector<Synapsis>();
		error = 0;
		//weight0 = (Math.random()-0.5)/10;
		if (!ultima)
			weight0 = ((Math.random()*2) - 1)/4;
		else
			weight0 = ((Math.random()*4) - 2)/4;
		System.out.println(" Weight0   " +weight0);
		
	}

	public void AddSynapsis(Synapsis s)
	{
		weights.add(s);
	}


}
