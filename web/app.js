function val(id) {
  return parseFloat(document.getElementById(id).value || "0");
}

function activate(name, z) {
  if (name === "lineal") return z;
  if (name === "escalon") return z >= 0 ? 1 : 0;
  if (name === "sigmoidal") return 1 / (1 + Math.exp(-z));
  if (name === "relu") return Math.max(0, z);
  if (name === "softmax") {
    const ez = Math.exp(z);
    return ez / (ez + 1);
  }
  if (name === "tanh") return Math.tanh(z);
  return z;
}

function toClass(name, y) {
  if (name === "tanh") return y >= 0 ? 1 : 0;
  return y >= 0.5 ? 1 : 0;
}

function runSimulation() {
  const x1 = val("x1");
  const x2 = val("x2");
  const w1 = val("w1");
  const w2 = val("w2");
  const b = val("b");
  const activation = document.getElementById("activation").value;

  const z = x1 * w1 + x2 * w2 + b;
  const fz = activate(activation, z);
  const cls = toClass(activation, fz);

  document.getElementById("zVal").textContent = z.toFixed(4);
  document.getElementById("fzVal").textContent = fz.toFixed(4);
  document.getElementById("classVal").textContent = String(cls);
}

function applyAndPreset() {
  document.getElementById("activation").value = "escalon";
  document.getElementById("w1").value = "1";
  document.getElementById("w2").value = "1";
  document.getElementById("b").value = "-1.5";
  runSimulation();
}

function applyOrPreset() {
  document.getElementById("activation").value = "escalon";
  document.getElementById("w1").value = "1";
  document.getElementById("w2").value = "1";
  document.getElementById("b").value = "-0.5";
  runSimulation();
}

document.getElementById("runSim").addEventListener("click", runSimulation);
document.getElementById("presetAnd").addEventListener("click", applyAndPreset);
document.getElementById("presetOr").addEventListener("click", applyOrPreset);

runSimulation();
