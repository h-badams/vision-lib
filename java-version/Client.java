import java.io.*;
import java.util.*;

// Validating the NNetwork code on the MNIST data set

public class Client {

    // Hyperparameters
    private static final int EPOCHS = 6;
    private static final double BATCH_SIZE = 128.0;
    private static final double LR_DECAY = 0.95;
    private static double LEARNING_RATE = 0.06;

    public static void main(String[] args) throws IOException {

        NNetwork testNet = new NNetwork(784, 32, 16, 10);

        for(int epoch = 0; epoch < EPOCHS; epoch++) {
            double epochCost = 0;
            double correctPredictions = 0;
            double totalExamples = 0;

            DataReader d = new DataReader(
                "java-version/mnist-data/train-images.idx3-ubyte",
                 "java-version/mnist-data/train-labels.idx1-ubyte");

            while(true) {
                double[][] data = d.readImage();

                if(data == null) {
                    break;
                }

                double[] input = data[0];
                double label = data[1][0];

                testNet.forwardPropNetwork(input);

                epochCost += testNet.getCost(label);

                if(testNet.getHighestActivation() == label) {
                    correctPredictions++;
                }
                totalExamples++;

                testNet.backPropNetwork(label, input);                

                if(totalExamples % BATCH_SIZE == 0 && totalExamples != 0) {
                    testNet.updateParams(LEARNING_RATE, BATCH_SIZE);
                }
            }
            LEARNING_RATE *= LR_DECAY;

            // Log epoch results
            double accuracy = (double) 100 * correctPredictions / totalExamples;
            System.out.printf("Epoch %d: Cost = %.4f, Training Accuracy = %.2f%%\n", epoch + 1, epochCost, accuracy);
            //System.out.printf("%d\t%.4f\n", epoch, accuracy);
        }

        // check testing data set 
        logTestAccurracy(testNet);

        saveNetworkConfiguration(testNet);
    }

    public static void saveNetworkConfiguration(NNetwork network) throws IOException {
        Scanner sc = new Scanner(System.in);
        System.out.print("Would you like to save the current configuration? (y/n) ");
        String response = sc.nextLine();
        if(response.equals("y")) {
            System.out.print("Name the configuration file: ");
            String filename = sc.nextLine();
            String path = "java-version/saved-network-weights/" + filename;

            File config = new File(path);
            config.createNewFile();

            FileWriter w = new FileWriter(config);
            network.writeParamsToFile(w);
            w.close();
        }
        sc.close();
    }

    public static void logTestAccurracy(NNetwork network) throws IOException {
        DataReader d = new DataReader(
            "java-version/mnist-data/t10k-images.idx3-ubyte",
            "java-version/mnist-data/t10k-labels.idx1-ubyte");
        double testCost = 0;
        double correctPredictions = 0;
        double totalExamples = 0;

        while(true) {
            double[][] data = d.readImage();

            if(data == null) {
                break;
            }
            
            double[] input = data[0]; 
            double label = data[1][0];

            network.forwardPropNetwork(input);

            testCost += network.getCost(label);

            if(network.getHighestActivation() == label) {
                correctPredictions++;
            }
            totalExamples++;
        }

        double accuracy = (double) 100 * correctPredictions / totalExamples;
        System.out.printf("Test Set: Cost = %.4f, Accuracy = %.2f%%\n", testCost, accuracy);
    }
}
