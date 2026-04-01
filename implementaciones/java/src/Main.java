import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        List<ResultCase> resultados = new ArrayList<>();
        resultados.add(runAndCase());
        resultados.add(runOrCase());
        resultados.add(runSpamCase());
        resultados.add(runClimaCase());
        resultados.add(runFraudeCase());
        resultados.add(runRiesgoCase());

        System.out.println("=================================================================");
        System.out.println("      EJECUCION DE CASOS DE PRUEBA DEL PERCEPTRON (JAVA)");
        System.out.println("=================================================================");
        System.out.printf("%-26s | %-10s | %-6s | %-8s | %-9s | %-7s%n",
            "Caso", "Activacion", "Epocas", "Accuracy", "Precision", "Recall");
        System.out.println("------------------------------------------------------------------------------------------");

        for (ResultCase r : resultados) {
            System.out.printf("%-26s | %-10s | %-6d | %8.4f | %9.4f | %7.4f%n",
                r.nombre, r.activacion, r.epocas, r.accuracy, r.precision, r.recall);
        }

        System.out.println("=================================================================");
    }

    private static ResultCase runAndCase() {
        double[][] x = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        int[] y = {0, 0, 0, 1};
        return trainAndEvaluate("AND Logico", x, y, Perceptron.Activation.ESCALON, 0.1, 15);
    }

    private static ResultCase runOrCase() {
        double[][] x = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        int[] y = {0, 1, 1, 1};
        return trainAndEvaluate("OR Logico", x, y, Perceptron.Activation.ESCALON, 0.1, 15);
    }

    private static ResultCase runSpamCase() {
        DataSet d = generateSpamData(42, 100);
        return trainAndEvaluate("Clasificacion Spam", d.x, d.y, Perceptron.Activation.SIGMOIDAL, 0.1, 50);
    }

    private static ResultCase runClimaCase() {
        DataSet d = generateClimaData(43, 100);
        return trainAndEvaluate("Prediccion Clima", d.x, d.y, Perceptron.Activation.TANH, 0.05, 50);
    }

    private static ResultCase runFraudeCase() {
        DataSet d = generateFraudeData(44, 100);
        return trainAndEvaluate("Deteccion Fraude", d.x, d.y, Perceptron.Activation.RELU, 0.01, 50);
    }

    private static ResultCase runRiesgoCase() {
        DataSet d = generateRiesgoData(45, 100);
        return trainAndEvaluate("Riesgo Academico", d.x, d.y, Perceptron.Activation.SIGMOIDAL, 0.05, 50);
    }

    private static ResultCase trainAndEvaluate(
        String nombre,
        double[][] x,
        int[] y,
        Perceptron.Activation activacion,
        double learningRate,
        int epocas
    ) {
        Perceptron model = new Perceptron(learningRate, epocas, activacion);
        model.train(x, y);
        Metrics metrics = precisionRecall(model, x, y);

        ResultCase result = new ResultCase();
        result.nombre = nombre;
        result.activacion = activacion.name().toLowerCase();
        result.epocas = epocas;
        result.accuracy = model.accuracy(x, y);
        result.precision = metrics.precision;
        result.recall = metrics.recall;
        return result;
    }

    private static Metrics precisionRecall(Perceptron model, double[][] x, int[] y) {
        int vp = 0;
        int fp = 0;
        int fn = 0;

        for (int i = 0; i < x.length; i++) {
            int pred = model.predict(x[i]);
            if (pred == 1 && y[i] == 1) vp++;
            if (pred == 1 && y[i] == 0) fp++;
            if (pred == 0 && y[i] == 1) fn++;
        }

        Metrics m = new Metrics();
        m.precision = (vp + fp) > 0 ? (double) vp / (vp + fp) : 0.0;
        m.recall = (vp + fn) > 0 ? (double) vp / (vp + fn) : 0.0;
        return m;
    }

    private static DataSet generateSpamData(int seed, int n) {
        Random rnd = new Random(seed);
        double[][] x = new double[n][4];
        int[] y = new int[n];

        for (int i = 0; i < n; i++) {
            double longitudAsunto = rnd.nextDouble();
            double numLinks = poisson(rnd, 2.0) / 10.0;
            double tieneOferta = bernoulli(rnd, 0.5);
            double remitenteDesconocido = bernoulli(rnd, 0.4);

            double score = longitudAsunto * 0.5 + numLinks * 1.5 + tieneOferta * 2.0 + remitenteDesconocido * 2.0;
            score += gaussian(rnd, 0.0, 0.5);

            y[i] = score > 2.5 ? 1 : 0;
            x[i][0] = longitudAsunto;
            x[i][1] = numLinks;
            x[i][2] = tieneOferta;
            x[i][3] = remitenteDesconocido;
        }

        return new DataSet(x, y);
    }

    private static DataSet generateClimaData(int seed, int n) {
        Random rnd = new Random(seed);
        double[][] x = new double[n][3];
        int[] y = new int[n];

        for (int i = 0; i < n; i++) {
            double temperatura = rnd.nextDouble();
            double humedad = rnd.nextDouble();
            double presion = rnd.nextDouble();

            double score = humedad * 2.0 - presion * 1.5 + temperatura * 0.5 + gaussian(rnd, 0.0, 0.2);
            y[i] = score > 0.5 ? 1 : 0;

            x[i][0] = temperatura;
            x[i][1] = humedad;
            x[i][2] = presion;
        }

        return new DataSet(x, y);
    }

    private static DataSet generateFraudeData(int seed, int n) {
        Random rnd = new Random(seed);
        double[][] x = new double[n][4];
        int[] y = new int[n];

        for (int i = 0; i < n; i++) {
            double monto = rnd.nextDouble();
            double hora = rnd.nextDouble();
            double distancia = rnd.nextDouble();
            double intentos = poisson(rnd, 1.0) / 5.0;

            double score = monto * 1.5 + distancia * 1.2 + intentos * 2.0 + gaussian(rnd, 0.0, 0.3);
            y[i] = score > 2.0 ? 1 : 0;

            x[i][0] = monto;
            x[i][1] = hora;
            x[i][2] = distancia;
            x[i][3] = intentos;
        }

        return new DataSet(x, y);
    }

    private static DataSet generateRiesgoData(int seed, int n) {
        Random rnd = new Random(seed);
        double[][] x = new double[n][4];
        int[] y = new int[n];

        for (int i = 0; i < n; i++) {
            double asistencia = rnd.nextDouble();
            double promedio = rnd.nextDouble();
            double entregas = poisson(rnd, 2.0) / 10.0;
            double horas = rnd.nextDouble();

            double score = (1.0 - asistencia) * 2.0 + (1.0 - promedio) * 2.5 + entregas * 1.5 - horas * 1.0;
            score += gaussian(rnd, 0.0, 0.2);

            y[i] = score > 2.0 ? 1 : 0;
            x[i][0] = asistencia;
            x[i][1] = promedio;
            x[i][2] = entregas;
            x[i][3] = horas;
        }

        return new DataSet(x, y);
    }

    private static double bernoulli(Random rnd, double p) {
        return rnd.nextDouble() < p ? 1.0 : 0.0;
    }

    private static int poisson(Random rnd, double lambda) {
        double l = Math.exp(-lambda);
        int k = 0;
        double p = 1.0;

        do {
            k++;
            p *= rnd.nextDouble();
        } while (p > l);

        return k - 1;
    }

    private static double gaussian(Random rnd, double mean, double stddev) {
        // Box-Muller para ruido normal sintetico.
        double u1 = 1.0 - rnd.nextDouble();
        double u2 = 1.0 - rnd.nextDouble();
        double z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        return mean + z * stddev;
    }

    private static class DataSet {
        double[][] x;
        int[] y;

        DataSet(double[][] x, int[] y) {
            this.x = x;
            this.y = y;
        }
    }

    private static class Metrics {
        double precision;
        double recall;
    }

    private static class ResultCase {
        String nombre;
        String activacion;
        int epocas;
        double accuracy;
        double precision;
        double recall;
    }
}
