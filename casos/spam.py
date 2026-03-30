import os
import numpy as np
from nucleo.perceptron import Perceptron

def generar_datos_spam():
    np.random.seed(42)
    n = 100
    
    longitud_asunto = np.random.uniform(0, 1, n)
    num_links = np.random.poisson(2, n) / 10.0
    tiene_oferta = np.random.binomial(1, 0.5, n)
    remitente_desconocido = np.random.binomial(1, 0.4, n)
    
    score = longitud_asunto * 0.5 + num_links * 1.5 + tiene_oferta * 2 + remitente_desconocido * 2
    score += np.random.normal(0, 0.5, n)
    
    y = np.where(score > 2.5, 1, 0)
    
    X = np.column_stack((longitud_asunto, num_links, tiene_oferta, remitente_desconocido))
    
    os.makedirs('/home/honorio/IA/perceptron/datos', exist_ok=True)
    datos_csv = np.column_stack((X, y))
    np.savetxt('/home/honorio/IA/perceptron/datos/spam.csv', datos_csv, delimiter=',', 
               header='longitud_asunto,num_links,tiene_oferta,remitente_desconocido,es_spam', comments='')
               
    return X, y

def ejecutar():
    X, y = generar_datos_spam()
    
    activacion = 'sigmoidal'
    epocas = 50
    modelo = Perceptron(tasa_aprendizaje=0.1, epocas=epocas, funcion_activacion=activacion)
    modelo.entrenar(X, y)
    
    accuracy, precision, recall = modelo.evaluar(X, y)
    predicciones = modelo.predecir(X)
    
    return {
        'nombre': 'Clasificación de Spam',
        'activacion_usada': activacion,
        'epocas': epocas,
        'accuracy': accuracy,
        'precision': precision,
        'datos_prueba': X,
        'predicciones': predicciones,
        'historial_errores': modelo.historial_errores
    }
