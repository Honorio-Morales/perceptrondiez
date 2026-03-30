import os
import numpy as np
from nucleo.perceptron import Perceptron

def generar_datos_riesgo():
    np.random.seed(45)
    n = 100
    
    asistencia = np.random.uniform(0, 1, n)
    promedio = np.random.uniform(0, 1, n)
    entregas = np.random.poisson(2, n) / 10.0
    horas = np.random.uniform(0, 1, n)
    
    score = (1 - asistencia) * 2.0 + (1 - promedio) * 2.5 + entregas * 1.5 - horas * 1.0 + np.random.normal(0, 0.2, n)
    y = np.where(score > 2.0, 1, 0)
    
    X = np.column_stack((asistencia, promedio, entregas, horas))
    
    os.makedirs('/home/honorio/IA/perceptron/datos', exist_ok=True)
    np.savetxt('/home/honorio/IA/perceptron/datos/riesgo.csv', np.column_stack((X, y)), delimiter=',', 
               header='asistencia,promedio,entregas_pendientes,horas_estudio,en_riesgo', comments='')
               
    return X, y

def ejecutar():
    X, y = generar_datos_riesgo()
    
    activacion = 'sigmoidal'
    epocas = 50
    modelo = Perceptron(tasa_aprendizaje=0.05, epocas=epocas, funcion_activacion=activacion)
    modelo.entrenar(X, y)
    
    accuracy, precision, recall = modelo.evaluar(X, y)
    predicciones = modelo.predecir(X)
    
    return {
        'nombre': 'Riesgo Académico',
        'activacion_usada': activacion,
        'epocas': epocas,
        'accuracy': accuracy,
        'precision': precision,
        'datos_prueba': X,
        'predicciones': predicciones,
        'historial_errores': modelo.historial_errores
    }
