import numpy as np

def lineal(x):
    """Función de activación lineal."""
    return x

def derivada_lineal(x):
    """Derivada de la función de activación lineal."""
    return np.ones_like(x)

def escalon(x):
    """Función de activación escalón (Heaviside modificado)."""
    return np.where(x >= 0, 1, 0)

def derivada_escalon(x):
    """
    Derivada de la función escalón.
    Matemáticamente es 0 (excepto en 0 donde es indefinida), pero para
    la regla clásica del perceptrón, retornamos 1 como proxy del gradiente
    para permitir la actualización directa de pesos.
    """
    return np.ones_like(x)

def sigmoidal(x):
    """Función de activación sigmoidal."""
    # Clip para evitar overflow en np.exp
    x_clip = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clip))

def derivada_sigmoidal(x):
    """Derivada de la función sigmoidal."""
    s = sigmoidal(x)
    return s * (1 - s)

def relu(x):
    """Función de activación ReLU (Rectified Linear Unit)."""
    return np.maximum(0, x)

def derivada_relu(x):
    """Derivada de la función ReLU."""
    return np.where(x > 0, 1, 0)

def softmax(x):
    """Función de activación Softmax."""
    # Restar el máximo para estabilidad numérica
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)

def derivada_softmax(x):
    """
    Derivada de la función Softmax pura (diagonal del Jacobiano).
    Para uso simplificado en el perceptrón.
    """
    s = softmax(x)
    return s * (1 - s)

def tanh(x):
    """Función de activación Tangente Hiperbólica."""
    return np.tanh(x)

def derivada_tanh(x):
    """Derivada de la función Tangente Hiperbólica."""
    return 1.0 - np.tanh(x)**2
