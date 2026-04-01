# Implementacion C# (base)

Este modulo contiene una base del perceptron para completar la entrega en C# sin librerias de machine learning.

## Archivos

- `Perceptron.cs`
- `Program.cs`

## Estado

- Clase `Perceptron` con 6 funciones de activacion.
- Entrenamiento binario con epocas > 10.
- Casos implementados: AND, OR, spam, clima, fraude y riesgo academico.
- Salida en consola con accuracy, precision y recall por caso.

## Ejecucion sugerida

Usar .NET SDK:

```bash
dotnet new console -n PerceptronApp
dotnet run
```

Copiar o mover los archivos base dentro del proyecto generado.
