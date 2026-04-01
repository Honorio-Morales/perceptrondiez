using System;
using System.Collections.Generic;

public class Program
{
    public static void Main()
    {
        var resultados = new List<ResultadoCaso>
        {
            EjecutarCasoAnd(),
            EjecutarCasoOr(),
            EjecutarCasoSpam(),
            EjecutarCasoClima(),
            EjecutarCasoFraude(),
            EjecutarCasoRiesgoAcademico()
        };

        Console.WriteLine("=================================================================");
        Console.WriteLine("       EJECUCION DE CASOS DE PRUEBA DEL PERCEPTRON (C#)");
        Console.WriteLine("=================================================================");
        Console.WriteLine($"{"Caso",-26} | {"Activacion",-10} | {"Epocas",-6} | {"Accuracy",-8} | {"Precision",-9} | {"Recall",-7}");
        Console.WriteLine(new string('-', 90));

        foreach (var r in resultados)
        {
            Console.WriteLine($"{r.Nombre,-26} | {r.Activacion,-10} | {r.Epocas,-6} | {r.Accuracy,8:F4} | {r.Precision,9:F4} | {r.Recall,7:F4}");
        }

        Console.WriteLine("=================================================================");
    }

    private static ResultadoCaso EjecutarCasoAnd()
    {
        double[][] x =
        {
            new double[] { 0, 0 },
            new double[] { 0, 1 },
            new double[] { 1, 0 },
            new double[] { 1, 1 }
        };
        int[] y = { 0, 0, 0, 1 };
        return EntrenarYEvaluar("AND Logico", x, y, ActivationType.Escalon, 0.1, 15);
    }

    private static ResultadoCaso EjecutarCasoOr()
    {
        double[][] x =
        {
            new double[] { 0, 0 },
            new double[] { 0, 1 },
            new double[] { 1, 0 },
            new double[] { 1, 1 }
        };
        int[] y = { 0, 1, 1, 1 };
        return EntrenarYEvaluar("OR Logico", x, y, ActivationType.Escalon, 0.1, 15);
    }

    private static ResultadoCaso EjecutarCasoSpam()
    {
        var (x, y) = GenerarDatosSpam(42, 100);
        return EntrenarYEvaluar("Clasificacion Spam", x, y, ActivationType.Sigmoidal, 0.1, 50);
    }

    private static ResultadoCaso EjecutarCasoClima()
    {
        var (x, y) = GenerarDatosClima(43, 100);
        return EntrenarYEvaluar("Prediccion Clima", x, y, ActivationType.Tanh, 0.05, 50);
    }

    private static ResultadoCaso EjecutarCasoFraude()
    {
        var (x, y) = GenerarDatosFraude(44, 100);
        return EntrenarYEvaluar("Deteccion Fraude", x, y, ActivationType.Relu, 0.01, 50);
    }

    private static ResultadoCaso EjecutarCasoRiesgoAcademico()
    {
        var (x, y) = GenerarDatosRiesgo(45, 100);
        return EntrenarYEvaluar("Riesgo Academico", x, y, ActivationType.Sigmoidal, 0.05, 50);
    }

    private static ResultadoCaso EntrenarYEvaluar(string nombre, double[][] x, int[] y, ActivationType activacion, double lr, int epocas)
    {
        var modelo = new Perceptron(learningRate: lr, epochs: epocas, activation: activacion);
        modelo.Train(x, y);
        var (precision, recall) = PrecisionRecall(modelo, x, y);

        return new ResultadoCaso
        {
            Nombre = nombre,
            Activacion = activacion.ToString().ToLowerInvariant(),
            Epocas = epocas,
            Accuracy = modelo.Accuracy(x, y),
            Precision = precision,
            Recall = recall
        };
    }

    private static (double precision, double recall) PrecisionRecall(Perceptron modelo, double[][] x, int[] y)
    {
        int vp = 0;
        int fp = 0;
        int fn = 0;

        for (int i = 0; i < x.Length; i++)
        {
            int pred = modelo.Predict(x[i]);
            if (pred == 1 && y[i] == 1) vp++;
            if (pred == 1 && y[i] == 0) fp++;
            if (pred == 0 && y[i] == 1) fn++;
        }

        double precision = (vp + fp) > 0 ? (double)vp / (vp + fp) : 0.0;
        double recall = (vp + fn) > 0 ? (double)vp / (vp + fn) : 0.0;
        return (precision, recall);
    }

    private static (double[][] x, int[] y) GenerarDatosSpam(int seed, int n)
    {
        var rnd = new Random(seed);
        var x = new double[n][];
        var y = new int[n];

        for (int i = 0; i < n; i++)
        {
            double longitudAsunto = rnd.NextDouble();
            double numLinks = Poisson(rnd, 2.0) / 10.0;
            double tieneOferta = Bernoulli(rnd, 0.5);
            double remitenteDesconocido = Bernoulli(rnd, 0.4);

            double score = longitudAsunto * 0.5 + numLinks * 1.5 + tieneOferta * 2.0 + remitenteDesconocido * 2.0;
            score += Gaussian(rnd, 0.0, 0.5);

            y[i] = score > 2.5 ? 1 : 0;
            x[i] = new[] { longitudAsunto, numLinks, tieneOferta, remitenteDesconocido };
        }

        return (x, y);
    }

    private static (double[][] x, int[] y) GenerarDatosClima(int seed, int n)
    {
        var rnd = new Random(seed);
        var x = new double[n][];
        var y = new int[n];

        for (int i = 0; i < n; i++)
        {
            double temperatura = rnd.NextDouble();
            double humedad = rnd.NextDouble();
            double presion = rnd.NextDouble();

            double score = humedad * 2.0 - presion * 1.5 + temperatura * 0.5 + Gaussian(rnd, 0.0, 0.2);
            y[i] = score > 0.5 ? 1 : 0;
            x[i] = new[] { temperatura, humedad, presion };
        }

        return (x, y);
    }

    private static (double[][] x, int[] y) GenerarDatosFraude(int seed, int n)
    {
        var rnd = new Random(seed);
        var x = new double[n][];
        var y = new int[n];

        for (int i = 0; i < n; i++)
        {
            double monto = rnd.NextDouble();
            double hora = rnd.NextDouble();
            double distancia = rnd.NextDouble();
            double intentos = Poisson(rnd, 1.0) / 5.0;

            double score = monto * 1.5 + distancia * 1.2 + intentos * 2.0 + Gaussian(rnd, 0.0, 0.3);
            y[i] = score > 2.0 ? 1 : 0;
            x[i] = new[] { monto, hora, distancia, intentos };
        }

        return (x, y);
    }

    private static (double[][] x, int[] y) GenerarDatosRiesgo(int seed, int n)
    {
        var rnd = new Random(seed);
        var x = new double[n][];
        var y = new int[n];

        for (int i = 0; i < n; i++)
        {
            double asistencia = rnd.NextDouble();
            double promedio = rnd.NextDouble();
            double entregas = Poisson(rnd, 2.0) / 10.0;
            double horas = rnd.NextDouble();

            double score = (1.0 - asistencia) * 2.0 + (1.0 - promedio) * 2.5 + entregas * 1.5 - horas * 1.0;
            score += Gaussian(rnd, 0.0, 0.2);

            y[i] = score > 2.0 ? 1 : 0;
            x[i] = new[] { asistencia, promedio, entregas, horas };
        }

        return (x, y);
    }

    private static double Bernoulli(Random rnd, double p) => rnd.NextDouble() < p ? 1.0 : 0.0;

    private static int Poisson(Random rnd, double lambda)
    {
        double l = Math.Exp(-lambda);
        int k = 0;
        double p = 1.0;

        do
        {
            k++;
            p *= rnd.NextDouble();
        }
        while (p > l);

        return k - 1;
    }

    private static double Gaussian(Random rnd, double mean, double stddev)
    {
        // Box-Muller para aproximar ruido normal en datos sinteticos.
        double u1 = 1.0 - rnd.NextDouble();
        double u2 = 1.0 - rnd.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return mean + z * stddev;
    }

    private class ResultadoCaso
    {
        public string Nombre { get; set; } = string.Empty;
        public string Activacion { get; set; } = string.Empty;
        public int Epocas { get; set; }
        public double Accuracy { get; set; }
        public double Precision { get; set; }
        public double Recall { get; set; }
    }
}
