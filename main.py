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

"""
main.py

Script de Comparación: GA-Paralelo vs Secuencial vs PSO-Paralelo
================================================================

Este script ejecuta experimentos comparativos entre tres enfoques de
vectorización TF-IDF:

1. Secuencial: Ejecución tradicional en un solo hilo (secuencial.py)
2. GA-Paralelo: Balanceo de carga con Algoritmo Genético (GA.py)
3. PSO-Paralelo: Balanceo de carga con Particle Swarm Optimization (PSO.py)

OBJETIVO:
---------
Medir y comparar el rendimiento de los tres enfoques en datasets de
diferentes tamaños para determinar:
- ¿Cuándo vale la pena usar optimización metaheurística?
- ¿Qué speedup se obtiene con cada método?
- ¿Cómo escala cada enfoque?
- ¿GA o PSO ofrece mejor rendimiento?

RESULTADOS GENERADOS:
--------------------
1. comparison_times.csv: Tabla con tiempos de ejecución y métricas
2. comparison_times.png: Gráfica comparativa de tiempos
3. comparison_speedup.png: Gráfica de speedup relativo
4. Logs detallados en consola
"""

import os
import sys
import time
import warnings
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

# Suprimir warnings de sklearn para logs más limpios
warnings.filterwarnings('ignore', category=UserWarning)

# Usar backend no interactivo para generar gráficas sin GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Importar pipelines de vectorización
from GA import vectorize_with_ga_load_balancing, GA_CONFIG
from secuencial import sequential_vectorize

# Importar PSO (si está disponible)
try:
    from PSO import vectorize_with_pso_load_balancing, PSO_CONFIG
    PSO_AVAILABLE = True
except ImportError:
    PSO_AVAILABLE = False
    print("⚠ Módulo PSO no disponible. Se omitirá en comparaciones.")


# ============================================================================
# FUNCIÓN PRINCIPAL DE COMPARACIÓN
# ============================================================================

def compare_pipelines(
    csv_path: str,
    sizes: List[int],
    output_csv: str = "comparison_times.csv",
    output_png: str = "comparison_times.png",
    output_speedup_png: str = "comparison_speedup.png",
    ga_config: Dict[str, Any] = None,
    pso_config: Dict[str, Any] = None,
    enable_pso: bool = True
) -> pd.DataFrame:
    """
    Ejecuta experimentos comparativos entre los tres enfoques
    
    PROCESO COMPLETO:
    ================
    1. Cargar dataset completo una sola vez (eficiencia)
    2. Para cada tamaño de subset:
       a. Crear subset de tamaño específico
       b. Ejecutar Secuencial y medir tiempo (baseline)
       c. Ejecutar GA-Paralelo y medir tiempo
       d. Ejecutar PSO-Paralelo y medir tiempo (si disponible)
       e. Calcular métricas comparativas (speedup)
       f. Guardar resultados
    3. Generar tabla CSV con todos los resultados
    4. Generar gráficas comparativas
    
    MÉTRICAS CALCULADAS:
    -------------------
    - Tiempo absoluto: Segundos de ejecución
    - Speedup: Secuencial_time / Paralelo_time
      * >1: paralelo es más rápido
      * <1: secuencial es más rápido
    
    Args:
        csv_path: Ruta al archivo CSV con el dataset completo
        sizes: Lista de tamaños a probar
        output_csv: Nombre del archivo CSV de salida
        output_png: Nombre del archivo PNG de gráfica de tiempos
        output_speedup_png: Nombre del archivo PNG de gráfica de speedup
        ga_config: Configuración del GA (default: GA_CONFIG)
        pso_config: Configuración del PSO (default: PSO_CONFIG)
        enable_pso: Si True, ejecuta comparación con PSO
    
    Returns:
        DataFrame con resultados comparativos completos
    """
    
    # ========================================================================
    # PASO 1: VALIDAR PARÁMETROS
    # ========================================================================
    if ga_config is None:
        ga_config = GA_CONFIG.copy()
    
    if pso_config is None and PSO_AVAILABLE:
        pso_config = PSO_CONFIG.copy()
    
    # Verificar si PSO está disponible y habilitado
    use_pso = PSO_AVAILABLE and enable_pso

    # ========================================================================
    # PASO 2: CARGAR DATASET
    # ========================================================================
    print("=" * 80)
    print("INICIANDO COMPARACIÓN DE ALGORITMOS DE BALANCEO DE CARGA")
    print("=" * 80)
    print(f"\nAlgoritmos a comparar:")
    print("  1. Secuencial (baseline)")
    print("  2. GA-Paralelo (Genetic Algorithm)")
    if use_pso:
        print("  3. PSO-Paralelo (Particle Swarm Optimization)")
    else:
        print("  3. PSO-Paralelo (NO DISPONIBLE)")
    
    print("\n" + "=" * 80)
    print("CARGANDO DATASET")
    print("=" * 80)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No se encontró {csv_path} en el directorio actual."
        )
    
    # Cargar dataset completo una sola vez
    df_full = pd.read_csv(csv_path)
    
    # Eliminar primera columna si es índice duplicado
    if df_full.columns[0] in ['Unnamed: 0', 'index']:
        df_full = df_full.drop(df_full.columns[0], axis=1)
    
    print(f"✓ Dataset cargado: {len(df_full):,} registros totales")
    print(f"✓ Configuración GA: {ga_config['num_cores']} cores, "
          f"{ga_config['population_size']} población, "
          f"{ga_config['num_generations']} generaciones")
    
    if use_pso:
        print(f"✓ Configuración PSO: {pso_config['num_cores']} cores, "
              f"{pso_config['num_particles']} partículas, "
              f"{pso_config['num_iterations']} iteraciones")
    
    # ========================================================================
    # PASO 3: EJECUTAR EXPERIMENTOS
    # ========================================================================
    results = []

    for size in sizes:
        if size > len(df_full):
            print(f"⚠ Tamaño solicitado ({size:,}) excede datos disponibles ({len(df_full):,}). Usando todos los datos.")
            size = len(df_full)
        print("\n" + "=" * 80)
        print(f"TAMAÑO DEL DATASET: {size:,} tweets")
        print("=" * 80)
        
        # Crear subset del tamaño especificado
        df_subset = df_full.head(size).reset_index(drop=True)

        # Diccionario para almacenar resultados de este experimento
        experiment_result = {
            'size': size,
            'seq_time': np.nan,
            'ga_time': np.nan,
            'pso_time': np.nan,
            'ga_speedup': np.nan,
            'pso_speedup': np.nan,
        }

        # ====================================================================
        # EXPERIMENTO 1: SECUENCIAL (BASELINE)
        # ====================================================================
        print("\n" + "-" * 80)
        print("[SECUENCIAL - BASELINE] Iniciando...")
        print("-" * 80)
        
        try:
            _, seq_total_time, seq_stats = sequential_vectorize(
                df_subset,
                intervalo=20_000
            )
            experiment_result['seq_time'] = seq_total_time
            print(f"✓ [SECUENCIAL] Completado en {seq_total_time:.2f}s")
            
        except Exception as e:
            print(f"✗ [SECUENCIAL] Error: {e}")
            import traceback
            traceback.print_exc()
            continue

        # ====================================================================
        # EXPERIMENTO 2: GA-PARALELO
        # ====================================================================
        print("\n" + "-" * 80)
        print("[GA-PARALELO] Iniciando...")
        print("-" * 80)
        
        try:
            _, ga_total_time, ga_stats = vectorize_with_ga_load_balancing(
                df_subset,
                config=ga_config,
                verbose=True
            )
            experiment_result['ga_time'] = ga_total_time
            print(f"✓ [GA-PARALELO] Completado en {ga_total_time:.2f}s")
            
        except Exception as e:
            print(f"✗ [GA-PARALELO] Error: {e}")
            import traceback
            traceback.print_exc()
            continue        # ====================================================================
        # EXPERIMENTO 3: PSO-PARALELO (si está disponible)
        # ====================================================================
        if use_pso:
            print("\n" + "-" * 80)
            print("[PSO-PARALELO] Iniciando...")
            print("-" * 80)
            
            try:
                _, pso_total_time, pso_stats = vectorize_with_pso_load_balancing(
                    df_subset,
                    config=pso_config
                )
                experiment_result['pso_time'] = pso_total_time
                print(f"✓ [PSO-PARALELO] Completado en {pso_total_time:.2f}s")
                
            except Exception as e:
                print(f"✗ [PSO-PARALELO] Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        # ====================================================================
        # CALCULAR MÉTRICAS COMPARATIVAS
        # ====================================================================
        print("\n" + "-" * 80)
        print("CALCULANDO MÉTRICAS COMPARATIVAS")
        print("-" * 80)
        
        seq_time = experiment_result['seq_time']
        ga_time = experiment_result['ga_time']
        pso_time = experiment_result['pso_time']
        
        # Speedup: Cuánto más rápido es el enfoque paralelo
        if not np.isnan(ga_time) and not np.isnan(seq_time) and seq_time > 0:
            ga_speedup = seq_time / ga_time
            experiment_result['ga_speedup'] = ga_speedup
            print(f"  GA Speedup: {ga_speedup:.2f}x")
        else:
            print("  GA Speedup: N/A")
        
        if use_pso and not np.isnan(pso_time) and not np.isnan(seq_time) and seq_time > 0:
            pso_speedup = seq_time / pso_time
            experiment_result['pso_speedup'] = pso_speedup
            print(f"  PSO Speedup: {pso_speedup:.2f}x")
        else:
            print("  PSO Speedup: N/A")
        
        # Comparación directa: GA vs PSO
        if (use_pso and not np.isnan(ga_time) and not np.isnan(pso_time) and
            ga_time > 0 and pso_time > 0):
            
            if ga_time < pso_time:
                faster_method = "GA"
                improvement = ((pso_time - ga_time) / pso_time) * 100
            else:
                faster_method = "PSO"
                improvement = ((ga_time - pso_time) / ga_time) * 100
            
            print(f"\n  {faster_method} es {improvement:.1f}% más rápido")
        
        # Agregar resultados a la lista
        results.append(experiment_result)
        
        # Mostrar resumen del experimento
        print("\n" + "-" * 80)
        print("RESUMEN DEL EXPERIMENTO")
        print("-" * 80)
        print(f"  Tamaño: {size:,} tweets")
        print(f"  Secuencial: {seq_time:.2f}s")
        print(f"  GA-Paralelo: {ga_time:.2f}s (speedup: {experiment_result['ga_speedup']:.2f}x)")
        if use_pso:
            print(f"  PSO-Paralelo: {pso_time:.2f}s (speedup: {experiment_result['pso_speedup']:.2f}x)")

    # ========================================================================
    # PASO 4: GENERAR DATAFRAME CON RESULTADOS
    # ========================================================================
    print("\n" + "=" * 80)
    print("PROCESANDO RESULTADOS")
    print("=" * 80)
    
    df_results = pd.DataFrame(results)
    
    # Ordenar columnas
    column_order = ['size', 'seq_time', 'ga_time', 'pso_time',
                   'ga_speedup', 'pso_speedup']
    
    column_order = [col for col in column_order if col in df_results.columns]
    df_results = df_results[column_order]
    
    # ========================================================================
    # PASO 5: GUARDAR TABLA CSV
    # ========================================================================
    print(f"\n  Guardando tabla de resultados en {output_csv}...")
    df_results.to_csv(output_csv, index=False, float_format='%.4f')
    print(f"  ✓ Tabla guardada exitosamente")
    
    # Mostrar tabla en consola
    print("\n" + "=" * 80)
    print("TABLA DE RESULTADOS COMPLETA")
    print("=" * 80)
    print(df_results.to_string(index=False))
    
    # ========================================================================
    # PASO 6: GENERAR GRÁFICAS
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERANDO GRÁFICAS")
    print("=" * 80)
    
    # ------------------------------------------------------------------------
    # GRÁFICA 1: Tiempos de Ejecución
    # ------------------------------------------------------------------------
    print(f"  Generando gráfica de tiempos: {output_png}...")
    
    plt.figure(figsize=(12, 7))
    
    # Plotear línea para cada método
    plt.plot(df_results['size'], df_results['seq_time'],
            marker='o', linewidth=2, markersize=8,
            label='Secuencial (Baseline)', color='#e74c3c')
    
    plt.plot(df_results['size'], df_results['ga_time'],
            marker='s', linewidth=2, markersize=8,
            label='GA-Paralelo', color='#3498db')
    
    if use_pso:
        plt.plot(df_results['size'], df_results['pso_time'],
                marker='^', linewidth=2, markersize=8,
                label='PSO-Paralelo', color='#2ecc71')
    
    # Configuración de la gráfica
    plt.xlabel('Tamaño del Dataset (número de tweets)', fontsize=12)
    plt.ylabel('Tiempo de Ejecución (segundos)', fontsize=12)
    plt.title('Comparación de Tiempos: Secuencial vs GA vs PSO',
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Formatear eje X para mostrar números con comas
    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: f'{int(x):,}')
    )
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Gráfica de tiempos guardada: {output_png}")
    
    # ------------------------------------------------------------------------
    # GRÁFICA 2: Speedup
    # ------------------------------------------------------------------------
    print(f"  Generando gráfica de speedup: {output_speedup_png}...")
    
    plt.figure(figsize=(12, 7))
    
    num_cores = ga_config['num_cores']
    
    # Línea de referencia: speedup ideal
    plt.axhline(y=num_cores, color='gray', linestyle='--',
               linewidth=2, alpha=0.7, label=f'Speedup Ideal ({num_cores}x)')
    plt.axhline(y=1, color='black', linestyle='-',
               linewidth=1, alpha=0.5, label='Sin mejora (1x)')
    
    # Plotear speedup de cada método
    plt.plot(df_results['size'], df_results['ga_speedup'],
            marker='s', linewidth=2, markersize=8,
            label='GA-Paralelo', color='#3498db')
    
    if use_pso:
        plt.plot(df_results['size'], df_results['pso_speedup'],
                marker='^', linewidth=2, markersize=8,
                label='PSO-Paralelo', color='#2ecc71')
    
    # Configuración de la gráfica
    plt.xlabel('Tamaño del Dataset (número de tweets)', fontsize=12)
    plt.ylabel('Speedup (veces más rápido que secuencial)', fontsize=12)
    plt.title('Speedup Comparativo: GA vs PSO',
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Formatear eje X
    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: f'{int(x):,}')
    )
    
    plt.tight_layout()
    plt.savefig(output_speedup_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Gráfica de speedup guardada: {output_speedup_png}")
    
    # ========================================================================
    # FINALIZACIÓN
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARACIÓN COMPLETADA")
    print("=" * 80)
    print(f"\nArchivos generados:")
    print(f"  • {output_csv} - Tabla de resultados")
    print(f"  • {output_png} - Gráfica de tiempos")
    print(f"  • {output_speedup_png} - Gráfica de speedup")
    print("\n" + "=" * 80)
    
    return df_results


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == '__main__':
    """
    Script principal de comparación
    
    Ejecuta comparación con batches de 20k, 40k, 60k, ..., 200k tweets
    """
    
    # ------------------------------------------------------------------------
    # CONFIGURACIÓN DEL EXPERIMENTO
    # ------------------------------------------------------------------------
    
    # Archivo de datos
    DATA_FILE = 'Suicide_Detection.csv'
    
    # Tamaños de batches a probar (20k, 40k, 60k, ..., 200k)
    SIZES = list(range(20_000, 200_001, 20_000))
    
    # Configuraciones
    GA_TEST_CONFIG = GA_CONFIG
    PSO_TEST_CONFIG = PSO_CONFIG if PSO_AVAILABLE else None
    
    # ------------------------------------------------------------------------
    # VERIFICAR REQUISITOS
    # ------------------------------------------------------------------------
    
    print("=" * 80)
    print("VERIFICACIÓN DE REQUISITOS")
    print("=" * 80)
    
    # Verificar archivo de datos
    if not os.path.exists(DATA_FILE):
        print(f"\n✗ Error: No se encontró el archivo {DATA_FILE}")
        print(f"  Asegúrate de tener el dataset en el directorio actual.")
        sys.exit(1)
    
    print(f"✓ Archivo de datos encontrado: {DATA_FILE}")
    
    # Verificar módulos
    print(f"✓ Módulo secuencial disponible")
    print(f"✓ Módulo GA disponible")
    
    if PSO_AVAILABLE:
        print(f"✓ Módulo PSO disponible")
    else:
        print(f"⚠ Módulo PSO no disponible (se omitirá)")
    
    print(f"\nConfiguración del experimento:")
    print(f"  • Tamaños a probar: {len(SIZES)} batches")
    print(f"  • Rango: {SIZES[0]:,} - {SIZES[-1]:,} tweets")
    print(f"  • Cores disponibles: {GA_CONFIG['num_cores']}")
    
    # ------------------------------------------------------------------------
    # EJECUTAR COMPARACIÓN
    # ------------------------------------------------------------------------
    
    print("\n¿Deseas continuar con la comparación? (y/n): ", end='')
    response = input().strip().lower()
    
    if response != 'y':
        print("Comparación cancelada.")
        sys.exit(0)
    
    try:
        # Ejecutar comparación completa
        results_df = compare_pipelines(
            csv_path=DATA_FILE,
            sizes=SIZES,
            output_csv='comparison_times.csv',
            output_png='comparison_times.png',
            output_speedup_png='comparison_speedup.png',
            ga_config=GA_TEST_CONFIG,
            pso_config=PSO_TEST_CONFIG,
            enable_pso=PSO_AVAILABLE
        )
        
        print("\n✓ Comparación completada exitosamente")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Comparación interrumpida por el usuario")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n✗ Error durante la comparación: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)        sys.exit(1)        sys.exit(1)        sys.exit(1)