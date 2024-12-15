import java.awt.*;
import java.io.*;
import java.util.Random;

// TODO class comment

public class FeatureVisualizer {

    public static final int IMG_WIDTH = 28;
    public static final int STEPS = 1500;
    public static final double LEARNING_RATE = 1;
    public static final int PIXEL_VAR = 25;

    // number we want
    public static final int LOGIT = 7;

    public static void main(String[] args) throws FileNotFoundException {
        NNetwork network = new NNetwork(new File(("configurations/toddlerNN1-64-64-972.txt")));
        Random r = new Random();

        // make a random image as array
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

                //System.out.println(pixelGradient);
                image[i] += pixelGradient * LEARNING_RATE;
            }
        }

        System.out.println(getLogitActivation(network, image));

        // make color array from num array
        Color[][] pixels = new Color[IMG_WIDTH][IMG_WIDTH];
        for(int i = 0; i < image.length; i++) {
            int intensity = Math.min(Math.max((int) image[i], 0), 255);
            pixels[i / IMG_WIDTH][i % IMG_WIDTH] = new Color(intensity, intensity, intensity);
        }

        // make an image
        Picture img = new Picture(IMG_WIDTH, IMG_WIDTH);
        img.setPixels(pixels);
        String path = "featurevisuals/" + LOGIT + "-" + STEPS + "-steps" + ".jpg";
        img.save(path);
    }

    public static double getLogitActivation(NNetwork network, double[] input) {
        double[] normalizedInput = new double[input.length];
        for(int i = 0; i < input.length; i++) {
            normalizedInput[i] = Math.min(Math.max(input[i] / 255.0, 0), 1.0);
        }
        network.forwardPropNetwork(normalizedInput);
        return network.layers.get(network.numLayers-1).z[LOGIT];
        //return network.layers.get(network.numLayers-1).activations[LOGIT];
    }
}
