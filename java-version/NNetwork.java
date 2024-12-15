import java.io.*;
import java.util.*;

// Represents a neural network

public class NNetwork {

    public int numLayers;
    public List<Layer> layers;

    // Creates a new neural network with the desired number and size
    // of each layer. Note that the first parameter is the size of the
    // input, and not an actual layer.

    public NNetwork(int... sizes) {
        this.numLayers = sizes.length - 1;
        this.layers = new ArrayList<>();

        for(int i = 1; i < sizes.length; i++) {
            if(i == numLayers) {
                layers.add(new Layer(sizes[i], sizes[i-1], new Softmax(), true));
            } else {
                layers.add(new Layer(sizes[i], sizes[i-1], new ReLU(), true));
            }
            
        }
    }

    // Creates a new neural network from a text file containing
    // network dimensions, weights & biases

    public NNetwork(File f) throws FileNotFoundException {
        Scanner sc = new Scanner(f);
        Scanner dimensions = new Scanner(sc.nextLine());

        List<Integer> sizes = new ArrayList<>();
  
        while(dimensions.hasNext()) {
            sizes.add(dimensions.nextInt());
        }
        dimensions.close();

        this.numLayers = sizes.size() - 1;
        this.layers = new ArrayList<>();
        
        // create blank network
        for(int i = 1; i < sizes.size(); i++) {
            if(i == numLayers) {
                layers.add(new Layer(sizes.get(i), sizes.get(i-1), new Softmax(), false));
            } else {
                layers.add(new Layer(sizes.get(i), sizes.get(i-1), new ReLU(), false));
            }
            // fill in weights
            for(int j = 0; j < layers.get(i-1).neurons; j++) {
                for(int k = 0; k < layers.get(i-1).prevLayerNeurons; k++) {
                    layers.get(i-1).weights[j][k] = sc.nextDouble();
                }
            }
            //fill in biases
            for(int j = 0; j < layers.get(i-1).neurons; j++) {
                layers.get(i-1).biases[j] = sc.nextDouble();;
            }
        }
        sc.close();
    }

    // Writes the dimensions, weights, and biases to a given file in
    // the format
    //
    // Dimensions
    //
    // Layer 0 Weights
    // Layer 0 Biases
    //
    // ...
    //
    // Layer n Weights
    // Layer n Biases

    public void writeParamsToFile(FileWriter w) throws IOException {
        w.write(layers.get(0).prevLayerNeurons + " ");
        for(int i = 0; i < numLayers; i++) {
            w.write(layers.get(i).neurons + " ");
        }
        w.write("\n\n");
        for(int i = 0; i < numLayers; i++) {
            for(int j = 0; j < layers.get(i).neurons; j++) {
                for(int k = 0; k < layers.get(i).prevLayerNeurons; k++) {
                    w.write(layers.get(i).weights[j][k] + " ");
                }
            }
            w.write("\n\n");
            for(int j = 0; j < layers.get(i).neurons; j++) {
                w.write(layers.get(i).biases[j] + " ");
            }
            w.write("\n\n");
        }
    }

    // Calculates the cost of the network's performance for a given
    // training example using cross entropy loss
    // Parameters: the index of the neuron that should be the most
    // brightly lit

    public double getCost(double correct) {
        double total = 0;
        for(int i = 0; i < layers.get(numLayers - 1).neurons; i++) {
            if(correct == i) {
                total -= Math.log(layers.get(numLayers - 1).activations[i]);
            }
        }
        return total;
    }

    // Returns the index of the neuron in the last layer that has the
    // strongest activation

    public int getHighestActivation() {
        double highest = 0;
        int index = 0;
        for(int i = 0; i < layers.get(numLayers - 1).neurons; i++) {
            if(layers.get(numLayers - 1).activations[i] > highest) {
                highest = layers.get(numLayers - 1).activations[i];
                index = i;
            }
        }
        return index;
    }

    // Returns the activations of the last layer of the network
    // Mostly for testing purposes

    public double[] getLastLayer() {
        return layers.get(numLayers - 1).activations;
    }

    // Passes the input through the network, layer by layer
    // Parameters: a double[] of the image input

    public void forwardPropNetwork(double[] input) {
        layers.get(0).forwardPropLayer(input);
        for(int i = 1; i < numLayers; i++) {
            layers.get(i).forwardPropLayer(layers.get(i-1).activations);
        }
    }

    // Performs the backpropagation that determines the gradient of the cost function

    public void backPropNetwork(double correct, double[] input) {
        for(int i = numLayers - 1; i >= 0; i--) {
            // last layer uses a different backprop function due to softmax
            if(i == numLayers - 1) {
                layers.get(i-1).layerGradient = layers.get(i).backPropLayer(layers.get(i-1).activations, correct);
            } 
            if(i != 0 && i != numLayers - 1) {
                layers.get(i-1).layerGradient = layers.get(i).backPropLayer(layers.get(i-1).activations);
            }
            if(i == 0) {
                layers.get(i).backPropLayer(input);
            }
        }
    }

    // Updates the weights and biases of a layer, then zeroes
    // all the gradients

    public void updateParams(double learningRate, double batchSize) {
        for(int i = 0; i < numLayers; i++) {
            for(int j = 0; j < layers.get(i).neurons; j++) {
                layers.get(i).biases[j] -= learningRate * layers.get(i).biasGradient[j] / batchSize;
                for(int k = 0; k < layers.get(i).prevLayerNeurons; k++) {
                    layers.get(i).weights[j][k] -= learningRate * layers.get(i).weightGradient[j][k] / batchSize;
                }
            }
        }
        zeroGradients();
    }

    // Helper method for updateParams, zeroes all the gradients after
    // weights and biases are updated

    private void zeroGradients() {
        for(int i = 0; i < numLayers; i++) {
            layers.get(i).zeroGradients();
        }
    }

    // Each network is made up of layers

    public class Layer {

        public int neurons;
        public int prevLayerNeurons;
        public double[] activations;
        public double[] z;
        public double[][] weights;
        public double[] biases;
        public ActivationFunction f;

        public double[] layerGradient;
        public double[][] weightGradient;
        public double[] biasGradient;

        // Creates a new layer consisting of neurons and their associated
        // weights and biases
        
        private Layer(int neurons, int prevLayerNeurons, ActivationFunction f, boolean randomize) {
            this.neurons = neurons;
            this.prevLayerNeurons = prevLayerNeurons;
    
            this.activations = new double[neurons];
            this.z = new double[neurons];
            this.weights = new double[neurons][prevLayerNeurons];
            this.biases = new double[neurons];
            this.f = f;

            this.layerGradient = new double[neurons];
            this.weightGradient = new double[neurons][prevLayerNeurons];
            this.biasGradient = new double[neurons];

            if(randomize) {
                randomizeParams();
            }
        }

        // Randomizes the network weights & biases

        private void randomizeParams() {
            Random r = new Random();

            for(int i = 0; i < neurons; i++) {
                double stDev = Math.sqrt(2.0 / (prevLayerNeurons + neurons));
                biases[i] = 0.005 * r.nextGaussian();
                for(int j = 0; j < prevLayerNeurons; j++) {
                    weights[i][j] = stDev * r.nextGaussian();
                }
            }
        }

        // Propagates forward based on the input from the previous layer
        // Parameters: a double[] of the previous layer's activations

        private void forwardPropLayer(double[] previousActivations) {
            if(prevLayerNeurons != previousActivations.length) {
                throw new IllegalArgumentException();
            }
            for(int i = 0; i < neurons; i++) {
                double weightedSum = 0;
                weightedSum += biases[i];
                for(int j = 0; j < prevLayerNeurons; j++) {
                    weightedSum += previousActivations[j] * weights[i][j];
                }
                z[i] = weightedSum;
            }
            for(int i = 0; i < neurons; i++) {
                activations[i] = f.activation(z, z[i]);
            }
        }

        // Performs backpropagation for a single layer, updating the
        // weight and bias gradients
        // Paramters: the previous layer's activations
        // Returns: The gradient of the cost function with respect to
        // previous layer (or, the sensitivity of the cost function to 
        // the previous layer's activations)

        // *I think* we want to have each layer return the gradient
        // for the activation layer of the previous neurons, while updating
        // the gradient for its own weights & biases

        private double[] backPropLayer(double[] previousActivations) {
            double[] prevLayerGradient = new double[prevLayerNeurons];

            for(int i = 0; i < neurons; i++) {
                biasGradient[i] += layerGradient[i] * f.derivative(z[i]);
                for(int j = 0; j < prevLayerNeurons; j++) {
                    weightGradient[i][j] += layerGradient[i] * f.derivative(z[i]) * previousActivations[j];
                }
            }
            for(int j = 0; j < prevLayerNeurons; j++) {
                for(int i = 0; i < neurons; i++) {
                    prevLayerGradient[j] += weights[i][j] * f.derivative(z[i]) * layerGradient[i];
                }
            }
            layerGradient = new double[neurons];

            return prevLayerGradient;
        }

        // Only used for the last layer when softmax allows the calculation of
        // weight & bias gradients without the activation function derivative

        private double[] backPropLayer(double[] previousActivations, double correct) {
            double[] prevLayerGradient = new double[prevLayerNeurons];

            for(int i = 0; i < neurons; i++) {
                double y = 0;
                if(correct == i) {
                    y = 1;
                }
                biasGradient[i] += activations[i] - y;
                for(int j = 0; j < prevLayerNeurons; j++) {
                    weightGradient[i][j] += (activations[i] - y) * previousActivations[j];
                }
            }
            for(int j = 0; j < prevLayerNeurons; j++) {
                for(int i = 0; i < neurons; i++) {
                    double y = 0;
                    if(correct == i) {
                        y = 1;
                    }
                    prevLayerGradient[j] += weights[i][j] * (activations[i] - y);
                }
            }
            layerGradient = new double[neurons];

            return prevLayerGradient;
        }

        // Resets the gradients for the layer

        private void zeroGradients() {
            layerGradient = new double[neurons];
            biasGradient = new double[neurons];
            weightGradient = new double[neurons][prevLayerNeurons];
        }
    }

}