public interface ActivationFunction {

    public abstract double activation(double[] layer, double x);
    public abstract double derivative(double x);

}
