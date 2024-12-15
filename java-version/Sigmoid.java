public class Sigmoid implements ActivationFunction {

    public double activation(double[] layer, double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public double derivative(double x) {
        return activation(null, x) * (1 - activation(null, x));
    }
}
