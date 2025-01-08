import java.io.*;
import java.awt.*;

// (MNIST specific) Turns data from ubyte file into a jgp image

public class ImageMaker {

    public static final int IMG_WIDTH = 28;
    public static final int NUM_IMAGES = 10;

    public static void main(String[] args) throws IOException {
        DataReader d = new DataReader(
            "java-version/mnist-data/t10k-images.idx3-ubyte",
            "java-version/mnist-data/t10k-labels.idx1-ubyte");

        for(int i = 0; i < NUM_IMAGES; i++) {
            double[][] input = d.readImage();
            double[] image = input[0];
            Color[][] pixels = new Color[IMG_WIDTH][IMG_WIDTH];

            for(int j = 0; j < image.length; j++) {
                int value = (int) (image[j] * 255.0);
                pixels[j / IMG_WIDTH][j % IMG_WIDTH] = new Color(value, value, value);
            }
    
            Picture img = new Picture(IMG_WIDTH, IMG_WIDTH);
            img.setPixels(pixels);
            String path = "java-version/pictures/testing-set-picture-" + i + ".jpg";
            img.save(path);
        }
    }

    public void saveIncorrectImage(double[] image, int iD) {
        Color[][] pixels = new Color[IMG_WIDTH][IMG_WIDTH];

        for(int j = 0; j < image.length; j++) {
            int value = (int) (image[j] * 255.0);
            pixels[j / IMG_WIDTH][j % IMG_WIDTH] = new Color(value, value, value);
        }

        Picture img = new Picture(IMG_WIDTH, IMG_WIDTH);
            img.setPixels(pixels);
            String path = "java-version/badguesses/incorrect-guess-" + iD + ".jpg";
            img.save(path);

    }
}
