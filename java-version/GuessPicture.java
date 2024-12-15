import java.io.*;
import java.awt.*;

public class GuessPicture {

    public static void main(String[] args) throws FileNotFoundException{
        
        NNetwork network = new NNetwork(new File("configurations/toddlerNN2-128-64-32-16-978.txt"));
        //NNetwork network = new NNetwork(new File("configurations/toddlerNN1-32-32-965.txt"));
        Picture guess = new Picture("pictures/testing-set-picture-9.jpg");
        //Picture guess = new Picture("featurevisuals/5-1000000-steps.jpg");
        Color[][] pixels = guess.getPixels();

        double[] input = new double[784];
        for(int i = 0; i < 28; i++) {
            for(int j = 0; j < 28; j++) {
                input[(28 * i) + j] = pixels[i][j].getRed() / 255.0;
            }
        }

        network.forwardPropNetwork(input);
        printGuess(network);
    }

    public static void printGuess(NNetwork network) {
        int guess = network.getHighestActivation();
        double[] activations = network.getLastLayer();

        System.out.println();
        System.out.println(guess);
        System.out.println();

        for(int i = 0; i < activations.length; i++) {
            System.out.printf("Confidence for %d: %.3f%%\n", i, activations[i] * 100);
        }
    }
}
