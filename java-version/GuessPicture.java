import java.io.*;
import java.awt.*;

// (MNIST specific)
// evaluates a pretrained network on a 28x28 greyscale image

public class GuessPicture {

    public static void main(String[] args) throws FileNotFoundException{
        
        NNetwork network = new NNetwork(
            new File("java-version/saved-network-weights/testNN2-128-64-32-16-978.txt"));

        // Make your own greyscale 28x28 image, put it in the pictures folder
        Picture guess = new Picture("java-version/pictures/testing-set-picture-9.jpg");
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
