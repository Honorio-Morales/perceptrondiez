using System;

public enum ActivationType
{
    Lineal,
    Escalon,
    Sigmoidal,
    Relu,
    Softmax,
    Tanh
}

public class Perceptron
{
    private readonly double learningRate;
    private readonly int epochs;
    private readonly ActivationType activation;
    private double[] weights = Array.Empty<double>();
    private double bias;

    public Perceptron(double learningRate = 0.1, int epochs = 50, ActivationType activation = ActivationType.Escalon)
    {
        if (epochs < 10)
        {
            throw new ArgumentException("epochs must be >= 10");
        }

        this.learningRate = learningRate;
        this.epochs = epochs;
        this.activation = activation;
    }

    public void Train(double[][] x, int[] y)
    {
        int samples = x.Length;
        int features = x[0].Length;
        weights = new double[features];
        bias = 0.0;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < samples; i++)
            {
                double linear = Dot(x[i], weights) + bias;
                double prediction = Activate(linear);
                double error = y[i] - prediction;
                double adjust = activation == ActivationType.Escalon
                    ? learningRate * error
                    : learningRate * error * Derivative(linear);

                for (int j = 0; j < features; j++)
                {
                    weights[j] += adjust * x[i][j];
                }

                bias += adjust;
            }
        }
    }

    public int Predict(double[] input)
    {
        double linear = Dot(input, weights) + bias;
        double raw = Activate(linear);
        return activation == ActivationType.Tanh
            ? (raw >= 0.0 ? 1 : 0)
            : (raw >= 0.5 ? 1 : 0);
    }

    public double Accuracy(double[][] x, int[] y)
    {
        int ok = 0;
        for (int i = 0; i < x.Length; i++)
        {
            if (Predict(x[i]) == y[i])
            {
                ok++;
            }
        }

        return (double)ok / x.Length;
    }

    private double Activate(double x)
    {
        return activation switch
        {
            ActivationType.Lineal => x,
            ActivationType.Escalon => x >= 0.0 ? 1.0 : 0.0,
            ActivationType.Sigmoidal => 1.0 / (1.0 + Math.Exp(-x)),
            ActivationType.Relu => Math.Max(0.0, x),
            ActivationType.Softmax => SoftmaxBinary(x),
            ActivationType.Tanh => Math.Tanh(x),
            _ => x
        };
    }

    private double Derivative(double x)
    {
        return activation switch
        {
            ActivationType.Lineal => 1.0,
            ActivationType.Escalon => 1.0,
            ActivationType.Sigmoidal => Activate(x) * (1.0 - Activate(x)),
            ActivationType.Relu => x > 0.0 ? 1.0 : 0.0,
            ActivationType.Softmax => SoftmaxBinary(x) * (1.0 - SoftmaxBinary(x)),
            ActivationType.Tanh => 1.0 - Math.Pow(Math.Tanh(x), 2),
            _ => 1.0
        };
    }

    // Softmax scalar simplificado para clasificacion binaria.
    private static double SoftmaxBinary(double x)
    {
        double ex = Math.Exp(x);
        return ex / (ex + 1.0);
    }

    private static double Dot(double[] a, double[] b)
    {
        double sum = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }
}
