import numpy as np
from nucleo import activaciones

class Perceptron:
    """Implementación de un Perceptrón desde cero."""
    
    def __init__(self, tasa_aprendizaje=0.01, epocas=50, funcion_activacion='escalon'):
        self.tasa_aprendizaje = tasa_aprendizaje
        if epocas < 10:
            raise ValueError("Las épocas deben ser mayores o iguales a 10.")
        self.epocas = epocas
        self.funcion_activacion = funcion_activacion
        self.pesos = None
        self.sesgo = None
        self.historial_errores = []
        
        # Mapeo de la función de activación solicitada a sus implementaciones en activaciones.py
        self.funciones = {
            'lineal': (activaciones.lineal, activaciones.derivada_lineal),
            'escalon': (activaciones.escalon, activaciones.derivada_escalon),
            'sigmoidal': (activaciones.sigmoidal, activaciones.derivada_sigmoidal),
            'relu': (activaciones.relu, activaciones.derivada_relu),
            'softmax': (activaciones.softmax, activaciones.derivada_softmax),
            'tanh': (activaciones.tanh, activaciones.derivada_tanh)
        }
        
        if self.funcion_activacion not in self.funciones:
            raise ValueError(f"La función de activación '{self.funcion_activacion}' no es soportada.")

    def entrenar(self, X, y):
        """
        Entrena el perceptrón ajustando pesos y sesgo a lo largo del número de épocas indicadas.
        X: matriz de características (n_muestras, n_caracteristicas)
        y: vector de etiquetas reales (n_muestras,)
        """
        n_muestras, n_caracteristicas = X.shape
        self.pesos = np.zeros(n_caracteristicas)
        self.sesgo = 0.0
        self.historial_errores = []
        
        activacion, derivada = self.funciones[self.funcion_activacion]
        
        for epoca in range(self.epocas):
            error_epoca = 0
            for i in range(n_muestras):
                # Salida lineal del perceptrón
                salida_lineal = np.dot(X[i], self.pesos) + self.sesgo
                # Pasamos por la función de activación
                prediccion = activacion(salida_lineal)
                
                # Calculamos el error (real - estimación)
                error = y[i] - prediccion
                
                # Para el perceptrón original (escalón), la regla de actualización es estándar.
                # Para otras funciones, aplicamos la regla delta multiplicada por la derivada.
                if self.funcion_activacion == 'escalon':
                    ajuste = self.tasa_aprendizaje * error
                else:
                    ajuste = self.tasa_aprendizaje * error * derivada(salida_lineal)
                
                # Actualizar pesos y sesgo
                self.pesos += ajuste * X[i]
                self.sesgo += ajuste
                
                # Sumar al error global de la época (error cuadrático)
                error_epoca += error ** 2
            
            # Guardamos el Error Cuadrático Medio de esta época
            self.historial_errores.append(error_epoca / n_muestras)

    def predecir(self, X):
        """
        Genera predicciones para las muestras dadas.
        Retorna un array de 0s y 1s.
        """
        activacion, _ = self.funciones[self.funcion_activacion]
        salida_lineal = np.dot(X, self.pesos) + self.sesgo
        predicciones_crudas = activacion(salida_lineal)
        
        # Binarizar según la naturaleza de la función de activación
        if self.funcion_activacion == 'tanh':
            # tanh produce salidas entre -1 y 1
            return np.where(predicciones_crudas >= 0, 1, 0)
        else:
            # Demás funciones producen salidas principalmente entre 0 y 1 o al menos evaluadas vs 0.5
            return np.where(predicciones_crudas >= 0.5, 1, 0)

    def evaluar(self, X, y):
        """
        Evalúa el modelo contra etiquetas conocidas y retorna métricas clave.
        Retorna: tuple(accuracy, precision, recall)
        """
        predicciones = self.predecir(X)
        
        vp = np.sum((predicciones == 1) & (y == 1))
        fp = np.sum((predicciones == 1) & (y == 0))
        fn = np.sum((predicciones == 0) & (y == 1))
        
        accuracy = np.mean(predicciones == y)
        precision = vp / (vp + fp) if (vp + fp) > 0 else 0.0
        recall = vp / (vp + fn) if (vp + fn) > 0 else 0.0
        
        return accuracy, precision, recall
