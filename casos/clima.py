import os
import numpy as np
from nucleo.perceptron import Perceptron

def generar_datos_clima():
    np.random.seed(43)
    n = 100
    
    temperatura = np.random.uniform(0, 1, n)
    humedad = np.random.uniform(0, 1, n)
    presion = np.random.uniform(0, 1, n)
    
    score = humedad * 2.0 - presion * 1.5 + temperatura * 0.5 + np.random.normal(0, 0.2, n)
    y = np.where(score > 0.5, 1, 0)
    
    X = np.column_stack((temperatura, humedad, presion))
    
    os.makedirs('/home/honorio/IA/perceptron/datos', exist_ok=True)
    datos_csv = np.column_stack((X, y))
    np.savetxt('/home/honorio/IA/perceptron/datos/clima.csv', datos_csv, delimiter=',', 
               header='temperatura,humedad,presion,lluvia', comments='')
               
    return X, y

def ejecutar():
    X, y = generar_datos_clima()
    
    activacion = 'tanh'
    epocas = 50
    modelo = Perceptron(tasa_aprendizaje=0.05, epocas=epocas, funcion_activacion=activacion)
    modelo.entrenar(X, y)
    
    accuracy, precision, recall = modelo.evaluar(X, y)
    predicciones = modelo.predecir(X)
    
    return {
        'nombre': 'Predicción de Clima',
        'activacion_usada': activacion,
        'epocas': epocas,
        'accuracy': accuracy,
        'precision': precision,
        'datos_prueba': X,
        'predicciones': predicciones,
        'historial_errores': modelo.historial_errores
    }
