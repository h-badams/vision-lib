import java.io.*;

// Reads in ubyte data

public class DataReader {

    DataInputStream dIn;
    DataInputStream lIn;
    int rows;
    int cols;

    // Creates a new DataReader object that reads in image and label data

    public DataReader(String dFile, String lFile) throws IOException {
        dIn = new DataInputStream(new BufferedInputStream(new FileInputStream(dFile)));
        dIn.readInt();
        dIn.readInt();

        rows = dIn.readInt();
        cols = dIn.readInt();

        lIn = new DataInputStream(new BufferedInputStream(new FileInputStream(lFile)));
        lIn.readInt();
        lIn.readInt();
    }

    // Creates a double array that stores pixel data as a number between
    // zero and one, as well as the image label

    public double[][] readImage() throws IOException {
        try {
            double[][] data = new double[2][];
            data[0] = new double[rows * cols];
            data[1] = new double[1];

            data[1][0] = lIn.readUnsignedByte();

            for(int i = 0; i < rows * cols; i++) {
                data[0][i] = dIn.readUnsignedByte() / 255.0;
            }
            return data;
        } catch(Exception e) {
            return null;
        }
    }

    // Closes label and data files - for reasons I guess

    public void closeFile() throws IOException {
        dIn.close();
        lIn.close();
    }
}