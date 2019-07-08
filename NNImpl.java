/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */

import java.util.*;


public class NNImpl{
	public ArrayList<Node> inputNodes = null; // list of the input layer nodes.
	public ArrayList<Node> hiddenNodes = null; // list of the hidden layer nodes
	public ArrayList<Node> outputNodes = null; // list of the output layer nodes

	public ArrayList<Instance> trainingSet = null; // the training set

	Double learningRate = 1.0; // variable to store the learning rate
	int maxEpoch = 1; // variable to store the maximum number of epochs

	/**
	 * This constructor creates the nodes necessary for the neural network
	 * Also connects the nodes of different layers
	 * After calling the constructor the last node of both inputNodes and  
	 * hiddenNodes will be bias nodes. 
	 */

	public NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[][] outputWeights)
	{
		this.trainingSet = trainingSet;
		this.learningRate = learningRate;
		this.maxEpoch = maxEpoch;

		// input layer nodes
		inputNodes = new ArrayList<Node>();
		int inputNodeCount = trainingSet.get(0).attributes.size();
		int outputNodeCount = trainingSet.get(0).classValues.size();
		for(int i = 0; i < inputNodeCount; i++)
		{
			Node node = new Node(0);
			inputNodes.add(node);
		}

		// bias node from input layer to hidden
		Node biasToHidden = new Node(1);
		inputNodes.add(biasToHidden);

		// hidden layer nodes
		hiddenNodes = new ArrayList<Node> ();
		for(int i = 0; i < hiddenNodeCount; i++)
		{
			Node node = new Node(2);
			// Connecting hidden layer nodes with input layer nodes
			for(int j = 0; j < inputNodes.size(); j++)
			{
				NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j),hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}

		// bias node from hidden layer to output
		Node biasToOutput = new Node(3);
		hiddenNodes.add(biasToOutput);

		// Output node layer
		outputNodes = new ArrayList<Node>();
		for (int i = 0; i < outputNodeCount; i++)
		{
			Node node = new Node(4);
			// Connecting output layer nodes with hidden layer nodes
			for(int j = 0; j < hiddenNodes.size(); j++)
			{
				NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}	
			outputNodes.add(node);
		}	
	}

	/**
	 * Get the output from the neural network for a single instance
	 * Return the index with highest output values. For example if the outputs
	 * of the outputNodes are [0.1, 0.5, 0.2, 0.1, 0.1], it should return 1. If outputs
	 * of the outputNodes are [0.1, 0.5, 0.1, 0.5, 0.2], it should return 3. 
	 * The parameter is a single instance. 
	 */

	public int calculateOutputForInstance(Instance instance)
	{
		// TODO: add code here
		// Update the input values of each hidden node based on the instance's attributes
		// Iterate through each hidden node minus the bias node
		for (int i = 0; i < hiddenNodes.size() - 1; ++i)
		{
			Node node = hiddenNodes.get(i);
			// Update each parent's input value for each hidden node
			for (int j = 0; j < node.parents.size() - 1; ++j)
			{
				NodeWeightPair pair = node.parents.get(j);
				pair.node.setInput(instance.attributes.get(j));
			}
		}
		// Propagate the values forward
		propagateForward();
		// Determine the best classification for the instance
		int index = 0;
		double bestClassification = -1;
		double output = 0.0;
		for (int i = 0; i < outputNodes.size(); ++i)
		{
			output = outputNodes.get(i).getOutput();
			if (output >= bestClassification)
			{
				bestClassification = output;
				index = i;
			}
		}
		return index;
	}


	/**
	 * Train the neural networks with the given parameters
	 * 
	 * The parameters are stored as attributes of this class
	 */

	public void train()
	{
		// TODO: add code here
		// Repeat based on maxEpoch value
		for (int epoch = 0; epoch < maxEpoch; ++epoch)
		{
			// Iterate through the instances
			for (int i = 0; i < trainingSet.size(); ++i)
			{
				Instance instance = trainingSet.get(i);
				// Propagate the inputs forward to compute the outputs
				calculateOutputForInstance(instance);
				// Propagate the deltas backward from output layer to input layer
				// Output layer to hidden layer deltas
				ArrayList<Double> outLayerDeltas = new ArrayList<Double>();
				for (int j = 0; j < outputNodes.size(); ++j)
				{
					Node node = outputNodes.get(j);
					outLayerDeltas.add(calculateSigmoidDeriv(node) * ((double) instance.classValues.get(j) - node.getOutput())); // Fix?
				}
				// Hidden layer to input layer deltas
				ArrayList<Double> hidLayerDeltas = new ArrayList<Double>();
				for (int j = 0; j < hiddenNodes.size() - 1; ++j)
				{
					Node currNode = hiddenNodes.get(j);
					double sigmoidDeriv = calculateSigmoidDeriv(currNode);
					double sum = 0.0;
					for (int k = 0; k < outputNodes.size(); ++k)
					{
						for (NodeWeightPair pair : outputNodes.get(k).parents)
						{
							if (pair.node == currNode)
							{
								sum += pair.weight * outLayerDeltas.get(k);
								break;
							}
						}
					}
					hidLayerDeltas.add(sigmoidDeriv * sum); // Fix?
				}
				// Update weights of input to hidden layer using deltas
				for (int j = 0; j < hiddenNodes.size() - 1; ++j)
				{
					for (NodeWeightPair pair : hiddenNodes.get(j).parents)
					{
						pair.weight = pair.weight + (learningRate * pair.node.getOutput() * hidLayerDeltas.get(j));
					}
				}
				// Update weights of input to output layer using deltas
				for (int j = 0; j < outputNodes.size(); ++j)
				{
					for (NodeWeightPair pair : outputNodes.get(j).parents)
					{
						pair.weight = pair.weight + (learningRate * pair.node.getOutput() * outLayerDeltas.get(j));
					}
				}
			}
		}
	}

	private void propagateForward()
	{
		for (int i = 0; i < 2; ++i)
		{
			// Hidden Layer 
			if (0 == i)
			{
				for (Node node : hiddenNodes)
				{
					node.calculateOutput();
				}
			}
			// Output Layer
			else
			{
				for (Node node : outputNodes)
				{
					node.calculateOutput();
				}
			}
		}
	}
	private double calculateSigmoidDeriv(Node node)
	{
		// g'(x) = g(x)(1 - g(x))
		double sigmoidCalc = 0.0;
		double sum = 0.0;
		for (int i = 0; i < node.parents.size(); ++i)
		{
			sum += node.parents.get(i).node.getOutput() * node.parents.get(i).weight;
		}
		sigmoidCalc = 1 / (1 + Math.pow(Math.E, (-1) * sum));
		return sigmoidCalc * (1.0 - sigmoidCalc);
	}
}
