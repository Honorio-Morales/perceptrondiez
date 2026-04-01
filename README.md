# Proyecto Perceptron Multilenguaje

Este repositorio contiene una implementacion de Perceptron desde cero y una arquitectura preparada para cumplir la entrega en multiples lenguajes.

## Objetivo

Implementar el perceptron en:
- Python
- C#
- Java
- Otro lenguaje (JavaScript en esta version base)

Aplicado a 6 casos de clasificacion binaria:
1. AND logico
2. OR logico
3. Spam vs no spam
4. Prediccion de clima
5. Deteccion de fraude
6. Riesgo academico

## Estructura del repositorio

- `main.py`, `nucleo/`, `casos/`: implementacion Python actual (funcional)
- `implementaciones/python/`: documentacion de la version Python
- `implementaciones/csharp/`: base del perceptron en C#
- `implementaciones/java/`: base del perceptron en Java
- `implementaciones/javascript/`: base del perceptron en JavaScript
- `web/`: pagina estatica para explicar el proyecto y navegar implementaciones

## Detalle tecnico (neuronas, arquitectura y entrenamiento)

### Cuantas neuronas se usaron

Se uso un **Perceptron monocapa de una sola neurona de salida** por cada caso.

- No hay capas ocultas.
- Cada entrenamiento corresponde a 1 neurona binaria (salida 0/1).
- Como hay 6 casos, se entrenan 6 modelos independientes por lenguaje.

Resumen de conteo:

- 1 neurona por modelo.
- 6 modelos por lenguaje (uno por caso).
- 4 lenguajes en el repositorio (Python, Java, JavaScript, C#).
- Total conceptual: 24 entrenamientos independientes (6 x 4).

### Como se hizo

El flujo de cada implementacion es el mismo:

1. Inicializar pesos en cero y sesgo en cero.
2. Elegir funcion de activacion y numero de epocas (>= 10).
3. Recorrer muestras en cada epoca.
4. Calcular salida lineal: `z = w.x + b`.
5. Aplicar activacion: `y_hat = f(z)`.
6. Calcular error: `error = y - y_hat`.
7. Ajustar pesos y sesgo con tasa de aprendizaje.
8. Evaluar con accuracy, precision y recall.

Regla de actualizacion usada:

- General: `delta_w = lr * (y - y_hat) * f'(z) * x`
- Para escalon: se usa la regla clasica sin derivada explicita.

### Funciones de activacion implementadas

- lineal
- escalon
- sigmoidal
- relu
- softmax (version binaria en este contexto)
- tanh

### Casos y configuracion

1. AND logico: escalon, 15 epocas.
2. OR logico: escalon, 15 epocas.
3. Spam: sigmoidal, 50 epocas.
4. Clima: tanh, 50 epocas.
5. Fraude: relu, 50 epocas.
6. Riesgo academico: sigmoidal, 50 epocas.

### Datos usados

- AND y OR: tablas de verdad.
- Spam, clima, fraude y riesgo: datos sinteticos (100 muestras por caso) con semilla fija para reproducibilidad.

### Metricas

- Accuracy
- Precision
- Recall (en las implementaciones multilenguaje)

### Resultado clave

- AND y OR convergen de forma perfecta (accuracy 1.0).
- En datos sinteticos, el desempeno varia segun caso y activacion.
- Fraude es el caso mas dificil con el perceptron monocapa actual.

### Limitacion principal

El modelo es lineal (una neurona), por lo que no resuelve bien fronteras no linealmente separables. Para mejorar, el siguiente paso natural es un MLP (perceptron multicapa) con retropropagacion.

## Estado actual

- Python: implementado y ejecutable con los 6 casos.
- Java: implementado y ejecutable con los 6 casos.
- JavaScript: implementado y ejecutable con los 6 casos.
- C#: implementado con los 6 casos (requiere entorno dotnet para ejecutar en local).

## Ejecucion rapida de Python

```bash
python main.py
```

## Ejecucion por lenguaje

Python:

```bash
/home/honorio/IA/perceptron/venv/bin/python main.py
```

Java:

```bash
javac implementaciones/java/src/*.java
java -cp implementaciones/java/src Main
```

JavaScript:

```bash
node implementaciones/javascript/main.js
```

C# (si dotnet esta disponible):

```bash
dotnet run
```

## Sitio estatico

Abrir `web/index.html` en navegador para ver el resumen del proyecto y enlaces a implementaciones.
