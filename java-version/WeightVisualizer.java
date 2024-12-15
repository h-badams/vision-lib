import java.awt.*;
import java.io.*;

// Creates a visual image of what a single neuron does by
// associating each pixel with the weight connecting a neuron
// in the input layer with that neuron, then coloring according
// to the weight strength

public class WeightVisualizer {

    public static final int IMG_WIDTH = 28;
    public static double minWeight = 0;
    public static double maxWeight = 0;

    public static void main(String[] args) throws FileNotFoundException {
        NNetwork network = new NNetwork(new File(("configurations/toddlerNN1-32-32-965.txt")));

        for(int i = 0; i < 32; i++) {
            for(int j = 0; j < IMG_WIDTH * IMG_WIDTH; j++) {
                double w = network.layers.get(0).weights[i][j];
                if(w > maxWeight) {
                    maxWeight = w;
                } else if(w < minWeight) {
                    minWeight = w;
                }
            }
        }

        for(int i = 0; i < 32; i++) {
            Color[][] pixels = new Color[IMG_WIDTH][IMG_WIDTH];
            for(int j = 0; j < IMG_WIDTH * IMG_WIDTH; j++) {
                pixels[j / IMG_WIDTH][j % IMG_WIDTH] = specialFunction(network.layers.get(0).weights[i][j]);
            }

            Picture img = new Picture(IMG_WIDTH, IMG_WIDTH);
            img.setPixels(pixels);
            String path = "weightvisuals/neuron-" + i + ".jpg";
            img.save(path);
        }
    }

    public static Color specialFunction(double x) {
        if(x > 0) {
            int intensity = (int) (300.0 * x / maxWeight);
            intensity = Math.min(intensity, 255);
            return new Color(intensity, 0, 0);
        } else {
            int intensity = (int) (300.0 * x / minWeight);
            intensity = Math.min(intensity, 255);
            return new Color(0, 0, intensity);
        }
    }

}

