import numpy as np
from nucleo.perceptron import Perceptron

def ejecutar():
    """Ejecuta el caso de prueba: Compuerta OR."""
    # Tabla de verdad OR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    
    activacion = 'escalon'
    epocas = 15
    
    # Instanciamos y entrenamos
    modelo = Perceptron(tasa_aprendizaje=0.1, epocas=epocas, funcion_activacion=activacion)
    modelo.entrenar(X, y)
    
    # Evaluamos
    accuracy, precision, recall = modelo.evaluar(X, y)
    predicciones = modelo.predecir(X)
    
    return {
        'nombre': 'OR Lógico',
        'activacion_usada': activacion,
        'epocas': epocas,
        'accuracy': accuracy,
        'precision': precision,
        'datos_prueba': X,
        'predicciones': predicciones,
        'historial_errores': modelo.historial_errores
    }
