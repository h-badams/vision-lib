public class Softmax implements ActivationFunction {

    public double activation(double[] z, double activation) {
        double sum = 0;
        for(int i = 0; i < z.length; i++) {
            sum += Math.exp(z[i]);
        }
        return Math.exp(activation) / sum;
    }

    public double derivative(double x) {
        throw new IllegalStateException();
    }
}
