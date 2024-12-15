public class ReLU implements ActivationFunction {
    
    public double activation(double[] layer, double x) {
        return x > 0 ? x : 0;
    }
    public double derivative(double x) {
        return x > 0 ? 1 : 0;
    }
}
