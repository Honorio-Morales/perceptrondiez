import os
import numpy as np
from nucleo.perceptron import Perceptron

def generar_datos_fraude():
    np.random.seed(44)
    n = 100
    
    monto = np.random.uniform(0, 1, n)
    hora = np.random.uniform(0, 1, n) 
    distancia = np.random.uniform(0, 1, n)
    intentos = np.random.poisson(1, n) / 5.0
    
    score = monto * 1.5 + distancia * 1.2 + intentos * 2.0 + np.random.normal(0, 0.3, n)
    y = np.where(score > 2.0, 1, 0)
    
    X = np.column_stack((monto, hora, distancia, intentos))
    
    os.makedirs('/home/honorio/IA/perceptron/datos', exist_ok=True)
    np.savetxt('/home/honorio/IA/perceptron/datos/fraude.csv', np.column_stack((X, y)), delimiter=',', 
               header='monto,hora,distancia_habitual,intentos_fallidos,es_fraude', comments='')
               
    return X, y

def ejecutar():
    X, y = generar_datos_fraude()
    
    activacion = 'relu'
    epocas = 50
    modelo = Perceptron(tasa_aprendizaje=0.01, epocas=epocas, funcion_activacion=activacion)
    modelo.entrenar(X, y)
    
    accuracy, precision, recall = modelo.evaluar(X, y)
    predicciones = modelo.predecir(X)
    
    return {
        'nombre': 'Detección de Fraude',
        'activacion_usada': activacion,
        'epocas': epocas,
        'accuracy': accuracy,
        'precision': precision,
        'datos_prueba': X,
        'predicciones': predicciones,
        'historial_errores': modelo.historial_errores
    }
