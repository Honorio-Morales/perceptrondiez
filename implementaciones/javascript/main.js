const { Perceptron } = require("./perceptron");

const resultados = [
  runAndCase(),
  runOrCase(),
  runSpamCase(),
  runClimaCase(),
  runFraudeCase(),
  runRiesgoCase(),
];

console.log("=================================================================");
console.log("   EJECUCION DE CASOS DE PRUEBA DEL PERCEPTRON (JAVASCRIPT)");
console.log("=================================================================");
console.log("Caso                       | Activacion | Epocas | Accuracy | Precision | Recall");
console.log("-------------------------------------------------------------------------");

for (const r of resultados) {
  console.log(
    `${pad(r.nombre, 26)} | ${pad(r.activacion, 10)} | ${pad(String(r.epocas), 6)} | ${r.accuracy.toFixed(4)}   | ${r.precision.toFixed(4)}    | ${r.recall.toFixed(4)}`
  );
}

console.log("=================================================================");

function runAndCase() {
  const x = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ];
  const y = [0, 0, 0, 1];
  return trainAndEvaluate("AND Logico", x, y, "escalon", 0.1, 15);
}

function runOrCase() {
  const x = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ];
  const y = [0, 1, 1, 1];
  return trainAndEvaluate("OR Logico", x, y, "escalon", 0.1, 15);
}

function runSpamCase() {
  const { x, y } = generateSpamData(42, 100);
  return trainAndEvaluate("Clasificacion Spam", x, y, "sigmoidal", 0.1, 50);
}

function runClimaCase() {
  const { x, y } = generateClimaData(43, 100);
  return trainAndEvaluate("Prediccion Clima", x, y, "tanh", 0.05, 50);
}

function runFraudeCase() {
  const { x, y } = generateFraudeData(44, 100);
  return trainAndEvaluate("Deteccion Fraude", x, y, "relu", 0.01, 50);
}

function runRiesgoCase() {
  const { x, y } = generateRiesgoData(45, 100);
  return trainAndEvaluate("Riesgo Academico", x, y, "sigmoidal", 0.05, 50);
}

function trainAndEvaluate(nombre, x, y, activacion, learningRate, epocas) {
  const model = new Perceptron(learningRate, epocas, activacion);
  model.train(x, y);

  const { precision, recall } = precisionRecall(model, x, y);
  return {
    nombre,
    activacion,
    epocas,
    accuracy: model.accuracy(x, y),
    precision,
    recall,
  };
}

function precisionRecall(model, x, y) {
  let vp = 0;
  let fp = 0;
  let fn = 0;

  for (let i = 0; i < x.length; i += 1) {
    const pred = model.predictOne(x[i]);
    if (pred === 1 && y[i] === 1) vp += 1;
    if (pred === 1 && y[i] === 0) fp += 1;
    if (pred === 0 && y[i] === 1) fn += 1;
  }

  return {
    precision: (vp + fp) > 0 ? vp / (vp + fp) : 0,
    recall: (vp + fn) > 0 ? vp / (vp + fn) : 0,
  };
}

function generateSpamData(seed, n) {
  const rng = mulberry32(seed);
  const x = [];
  const y = [];

  for (let i = 0; i < n; i += 1) {
    const longitudAsunto = rng();
    const numLinks = poisson(rng, 2.0) / 10.0;
    const tieneOferta = bernoulli(rng, 0.5);
    const remitenteDesconocido = bernoulli(rng, 0.4);

    let score = longitudAsunto * 0.5 + numLinks * 1.5 + tieneOferta * 2.0 + remitenteDesconocido * 2.0;
    score += gaussian(rng, 0, 0.5);

    y.push(score > 2.5 ? 1 : 0);
    x.push([longitudAsunto, numLinks, tieneOferta, remitenteDesconocido]);
  }

  return { x, y };
}

function generateClimaData(seed, n) {
  const rng = mulberry32(seed);
  const x = [];
  const y = [];

  for (let i = 0; i < n; i += 1) {
    const temperatura = rng();
    const humedad = rng();
    const presion = rng();

    const score = humedad * 2.0 - presion * 1.5 + temperatura * 0.5 + gaussian(rng, 0, 0.2);
    y.push(score > 0.5 ? 1 : 0);
    x.push([temperatura, humedad, presion]);
  }

  return { x, y };
}

function generateFraudeData(seed, n) {
  const rng = mulberry32(seed);
  const x = [];
  const y = [];

  for (let i = 0; i < n; i += 1) {
    const monto = rng();
    const hora = rng();
    const distancia = rng();
    const intentos = poisson(rng, 1.0) / 5.0;

    const score = monto * 1.5 + distancia * 1.2 + intentos * 2.0 + gaussian(rng, 0, 0.3);
    y.push(score > 2.0 ? 1 : 0);
    x.push([monto, hora, distancia, intentos]);
  }

  return { x, y };
}

function generateRiesgoData(seed, n) {
  const rng = mulberry32(seed);
  const x = [];
  const y = [];

  for (let i = 0; i < n; i += 1) {
    const asistencia = rng();
    const promedio = rng();
    const entregas = poisson(rng, 2.0) / 10.0;
    const horas = rng();

    let score = (1 - asistencia) * 2.0 + (1 - promedio) * 2.5 + entregas * 1.5 - horas * 1.0;
    score += gaussian(rng, 0, 0.2);

    y.push(score > 2.0 ? 1 : 0);
    x.push([asistencia, promedio, entregas, horas]);
  }

  return { x, y };
}

function bernoulli(rng, p) {
  return rng() < p ? 1 : 0;
}

function poisson(rng, lambda) {
  const l = Math.exp(-lambda);
  let k = 0;
  let p = 1;

  do {
    k += 1;
    p *= rng();
  } while (p > l);

  return k - 1;
}

function gaussian(rng, mean, stddev) {
  // Box-Muller para ruido normal sintetico.
  const u1 = 1 - rng();
  const u2 = 1 - rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + z * stddev;
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return function random() {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function pad(value, width) {
  return value.length >= width ? value.slice(0, width) : `${value}${" ".repeat(width - value.length)}`;
}
