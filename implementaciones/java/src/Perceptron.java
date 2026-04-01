public class Perceptron {
    public enum Activation {
        LINEAL, ESCALON, SIGMOIDAL, RELU, SOFTMAX, TANH
    }

    private final double learningRate;
    private final int epochs;
    private final Activation activation;
    private double[] weights;
    private double bias;

    public Perceptron(double learningRate, int epochs, Activation activation) {
        if (epochs < 10) {
            throw new IllegalArgumentException("epochs must be >= 10");
        }
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.activation = activation;
    }

    public void train(double[][] x, int[] y) {
        int samples = x.length;
        int features = x[0].length;
        this.weights = new double[features];
        this.bias = 0.0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < samples; i++) {
                double linear = dot(x[i], weights) + bias;
                double pred = activate(linear);
                double error = y[i] - pred;
                double adjust = activation == Activation.ESCALON
                    ? learningRate * error
                    : learningRate * error * derivative(linear);

                for (int j = 0; j < features; j++) {
                    weights[j] += adjust * x[i][j];
                }
                bias += adjust;
            }
        }
    }

    public int predict(double[] input) {
        double linear = dot(input, weights) + bias;
        double raw = activate(linear);
        if (activation == Activation.TANH) {
            return raw >= 0.0 ? 1 : 0;
        }
        return raw >= 0.5 ? 1 : 0;
    }

    public double accuracy(double[][] x, int[] y) {
        int ok = 0;
        for (int i = 0; i < x.length; i++) {
            if (predict(x[i]) == y[i]) {
                ok++;
            }
        }
        return (double) ok / x.length;
    }

    private double activate(double x) {
        switch (activation) {
            case LINEAL:
                return x;
            case ESCALON:
                return x >= 0.0 ? 1.0 : 0.0;
            case SIGMOIDAL:
                return 1.0 / (1.0 + Math.exp(-x));
            case RELU:
                return Math.max(0.0, x);
            case SOFTMAX:
                return softmaxBinary(x);
            case TANH:
                return Math.tanh(x);
            default:
                return x;
        }
    }

    private double derivative(double x) {
        switch (activation) {
            case LINEAL:
                return 1.0;
            case ESCALON:
                return 1.0;
            case SIGMOIDAL:
                double s = 1.0 / (1.0 + Math.exp(-x));
                return s * (1.0 - s);
            case RELU:
                return x > 0.0 ? 1.0 : 0.0;
            case SOFTMAX:
                double sm = softmaxBinary(x);
                return sm * (1.0 - sm);
            case TANH:
                double t = Math.tanh(x);
                return 1.0 - t * t;
            default:
                return 1.0;
        }
    }

    private double softmaxBinary(double x) {
        double ex = Math.exp(x);
        return ex / (ex + 1.0);
    }

    private static double dot(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
}
