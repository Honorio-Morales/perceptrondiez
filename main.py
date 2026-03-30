import os
import sys
import datetime
import importlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Asegurar que el directorio actual está en el path para las importaciones
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    casos = [
        'and_logico',
        'or_logico',
        'spam',
        'clima',
        'fraude',
        'riesgo_academico'
    ]
    
    os.makedirs('/home/honorio/IA/perceptron/resultados', exist_ok=True)
    
    resultados_lista = []
    
    print("=" * 65)
    print("      EJECUCIÓN DE CASOS DE PRUEBA DEL PERCEPTRÓN")
    print("=" * 65)
    
    for nombre_modulo in casos:
        print(f"[*] Ejecutando modelo: {nombre_modulo}...")
        modulo = importlib.import_module(f"casos.{nombre_modulo}")
        resultado = modulo.ejecutar()
        resultados_lista.append(resultado)
        
        # Generar y guardar la gráfica del error
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, resultado['epocas'] + 1), resultado['historial_errores'], marker='o', color='b')
        plt.title(f"Error Cuadrático Medio por Época\nCaso: {resultado['nombre']}")
        plt.xlabel("Épocas")
        plt.ylabel("Error")
        plt.grid(True)
        
        ruta_grafica = f"/home/honorio/IA/perceptron/resultados/{nombre_modulo}.png"
        plt.savefig(ruta_grafica)
        plt.close()
        
    # Imprimir tabla en consola
    print("\n" + "=" * 65)
    print(f"{'Caso':<25} | {'Activación':<12} | {'Épocas':<8} | {'Accuracy':<10}")
    print("-" * 65)
    
    for res in resultados_lista:
        caso_formateado = res['nombre']
        activacion = res['activacion_usada']
        epocas = res['epocas']
        accuracy = res['accuracy']
        
        print(f"{caso_formateado:<25} | {activacion:<12} | {epocas:<8} | {accuracy:.4f}")
    print("=" * 65 + "\n")
    
    # Generar reporte txt
    fecha_hora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ruta_reporte = "/home/honorio/IA/perceptron/resultados/reporte.txt"
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write("============================================================\n")
        f.write("             REPORTE DE EJECUCIÓN DEL PERCEPTRÓN\n")
        f.write("============================================================\n")
        f.write(f"Fecha y Hora: {fecha_hora}\n\n")
        f.write(f"{'Caso':<25} | {'Activación':<12} | {'Épocas':<8} | {'Accuracy':<8} | {'Precisión':<8}\n")
        f.write("-" * 75 + "\n")
        for res in resultados_lista:
            f.write(f"{res['nombre']:<25} | {res['activacion_usada']:<12} | {res['epocas']:<8} | {res['accuracy']:.4f}   | {res['precision']:.4f}\n")
            
    print(f"[+] ¡Proceso finalizado con éxito!")
    print(f"[+] Las gráficas (.png) y 'reporte.txt' se guardaron en: /home/honorio/IA/perceptron/resultados/")

if __name__ == "__main__":
    main()
