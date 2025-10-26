import pandas as pd
import matplotlib.pyplot as plt

def comparar_tiempo_por_bloques(pso_path, secuencial_path):
    # Cargar los archivos CSV
    pso = pd.read_csv(pso_path)
    sec = pd.read_csv(secuencial_path)
    
    # Limitar a las primeras 220000 filas
    pso = pso.iloc[:220000]
    sec = sec.iloc[:220000]
    
    # Eliminar la columna 'tiempo_promedio_s' si existe
    if 'tiempo_promedio_s' in pso.columns:
        pso = pso.drop(columns=['tiempo_promedio_s'])
    if 'tiempo_promedio_s' in sec.columns:
        sec = sec.drop(columns=['tiempo_promedio_s'])
    
    # Detectar columnas relevantes
    col_tweets = pso.columns[0]   # tweets procesados
    col_tiempo = pso.columns[1]   # tiempo tardado
    
    # Definir bloques de 20,000 tweets
    bloque = 20000
    max_tweets = 200000
    bloques = list(range(bloque, max_tweets + bloque, bloque))
    
    # Calcular tiempo total hasta cada bloque
    tiempos_pso = []
    tiempos_sec = []
    
    for limite in bloques:
        tiempos_pso.append(pso[pso[col_tweets] <= limite][col_tiempo].sum())
        tiempos_sec.append(sec[sec[col_tweets] <= limite][col_tiempo].sum())
    
    # Graficar comparación
    plt.figure(figsize=(10,6))
    x = range(len(bloques))
    width = 0.35

    plt.bar([i - width/2 for i in x], tiempos_pso, width, label='PSO')
    plt.bar([i + width/2 for i in x], tiempos_sec, width, label='Secuencial')

    plt.xlabel('Tweets procesados')
    plt.ylabel('Tiempo total tardado (s)')
    plt.title('Comparación de tiempo entre PSO y Secuencial')
    plt.xticks(x, [f'{b}' for b in bloques])
    plt.legend()
    plt.tight_layout()
    plt.show()

# Ejemplo de uso:
comparar_tiempo_por_bloques('tiempoPSO.csv', 'tiempo.csv')
