import java.io.*;

public class Testing {

    // TODO method comment
    public static void main(String[] args) throws IOException {

        NNetwork testNetwork = new NNetwork(new File("configurations/toddlerNN2-128-64-32-16-978.txt"));
        DataReader d = new DataReader("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
        double testCost = 0;
        double correctPredictions = 0;
        double totalExamples = 0;

        ImageMaker imageMaker = new ImageMaker();

        while(true) {
            double[][] data = d.readImage();

            if(data == null) {
                break;
            }
            
            double[] input = data[0]; 
            double label = data[1][0];

            testNetwork.forwardPropNetwork(input);

            testCost += testNetwork.getCost(label);

            if(testNetwork.getHighestActivation() == label) {
                correctPredictions++;
            } else {
                imageMaker.saveIncorrectImage(input, (int) totalExamples);
            }
            totalExamples++;
        }

        double accuracy = (double) 100 * correctPredictions / totalExamples;
        System.out.printf("Test Set: Cost = %.4f, Accuracy = %.2f%%\n", testCost, accuracy);

    }
}
