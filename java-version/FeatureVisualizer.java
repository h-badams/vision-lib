import java.awt.*;
import java.io.*;
import java.util.Random;

// Starts with an image of random noise, then updates the image
// pixels to maximize the network's confidence in a given digit,
// i.e. the number seven. 

public class FeatureVisualizer {

    public static final int IMG_WIDTH = 28;
    public static final int STEPS = 1000;
    public static final double LEARNING_RATE = 1;
    public static final int PIXEL_VAR = 25;

    // Number we want to visualize
    public static final int LOGIT = 3;

    public static void main(String[] args) throws FileNotFoundException {

        // Load in pretrained network
        NNetwork network = new NNetwork(new File(("java-version/saved-network-weights/testNN1-32-32-965.txt")));
        Random r = new Random();

        // Make a random image as array
        double[] image = new double[IMG_WIDTH * IMG_WIDTH];
        for(int i = 0; i < image.length; i++) {
            image[i] = r.nextInt(128 - PIXEL_VAR, 128 + PIXEL_VAR);
        }
        
        for(int steps = 0; steps < STEPS; steps++) {
            for(int i = 0; i < image.length; i++) {
                network.forwardPropNetwork(image);

                // replace with recursion ?
                double pixelGradient = 0;
                for(int j = 0; j < network.layers.get(0).neurons; j++) {
                    for(int k = 0; k < network.layers.get(1).neurons; k++) {
                        
                        pixelGradient += 
                        network.layers.get(0).weights[j][i] * network.layers.get(0).f.derivative(network.layers.get(0).z[j]) *
                        network.layers.get(1).weights[k][j] * network.layers.get(1).f.derivative(network.layers.get(1).z[k]) * 
                        network.layers.get(2).weights[LOGIT][k];
                        
                    }
                }
                image[i] += pixelGradient * LEARNING_RATE;
            }
        }

        System.out.println(getLogitActivation(network, image));

        // Make color array from num array
        Color[][] pixels = new Color[IMG_WIDTH][IMG_WIDTH];
        for(int i = 0; i < image.length; i++) {
            int intensity = Math.min(Math.max((int) image[i], 0), 255);
            pixels[i / IMG_WIDTH][i % IMG_WIDTH] = new Color(intensity, intensity, intensity);
        }

        // Make an image
        Picture img = new Picture(IMG_WIDTH, IMG_WIDTH);
        img.setPixels(pixels);
        String path = "java-version/pictures/" + LOGIT + "-" + STEPS + "-steps" + ".jpg";
        img.save(path);
    }

    public static double getLogitActivation(NNetwork network, double[] input) {
        double[] normalizedInput = new double[input.length];
        for(int i = 0; i < input.length; i++) {
            normalizedInput[i] = Math.min(Math.max(input[i] / 255.0, 0), 1.0);
        }
        network.forwardPropNetwork(normalizedInput);
        //return network.layers.get(network.numLayers-1).z[LOGIT];
        return network.layers.get(network.numLayers-1).activations[LOGIT];
    }
}
