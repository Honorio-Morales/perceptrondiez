class Perceptron {
  constructor(learningRate = 0.1, epochs = 50, activation = "escalon") {
    if (epochs < 10) {
      throw new Error("epochs must be >= 10");
    }

    this.learningRate = learningRate;
    this.epochs = epochs;
    this.activation = activation;
    this.weights = [];
    this.bias = 0;
  }

  train(X, y) {
    const samples = X.length;
    const features = X[0].length;
    this.weights = Array(features).fill(0);
    this.bias = 0;

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      for (let i = 0; i < samples; i++) {
        const linear = dot(X[i], this.weights) + this.bias;
        const pred = this.activate(linear);
        const error = y[i] - pred;
        const adjust = this.activation === "escalon"
          ? this.learningRate * error
          : this.learningRate * error * this.derivative(linear);

        for (let j = 0; j < features; j++) {
          this.weights[j] += adjust * X[i][j];
        }
        this.bias += adjust;
      }
    }
  }

  predictOne(input) {
    const linear = dot(input, this.weights) + this.bias;
    const raw = this.activate(linear);
    if (this.activation === "tanh") {
      return raw >= 0 ? 1 : 0;
    }
    return raw >= 0.5 ? 1 : 0;
  }

  accuracy(X, y) {
    let ok = 0;
    for (let i = 0; i < X.length; i++) {
      if (this.predictOne(X[i]) === y[i]) {
        ok += 1;
      }
    }
    return ok / X.length;
  }

  activate(x) {
    switch (this.activation) {
      case "lineal":
        return x;
      case "escalon":
        return x >= 0 ? 1 : 0;
      case "sigmoidal":
        return 1 / (1 + Math.exp(-x));
      case "relu":
        return Math.max(0, x);
      case "softmax":
        return softmaxBinary(x);
      case "tanh":
        return Math.tanh(x);
      default:
        return x;
    }
  }

  derivative(x) {
    switch (this.activation) {
      case "lineal":
        return 1;
      case "escalon":
        return 1;
      case "sigmoidal": {
        const s = 1 / (1 + Math.exp(-x));
        return s * (1 - s);
      }
      case "relu":
        return x > 0 ? 1 : 0;
      case "softmax": {
        const s = softmaxBinary(x);
        return s * (1 - s);
      }
      case "tanh": {
        const t = Math.tanh(x);
        return 1 - t * t;
      }
      default:
        return 1;
    }
  }
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

function softmaxBinary(x) {
  const ex = Math.exp(x);
  return ex / (ex + 1);
}

module.exports = { Perceptron };
