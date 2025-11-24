"""
main.py

Script de Comparación: GA-Paralelo vs Secuencial vs PSO-Paralelo
================================================================
"""

import os
import sys
import warnings
from typing import List, Dict, Any

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

# Importar pipelines
from secuencial import sequential_vectorize
from PSO import vectorize_with_pso_load_balancing, PSO_CONFIG
from GA import vectorize_with_ga_load_balancing, GA_CONFIG
from WOA import vectorize_with_woa_load_balancing, WOA_CONFIG
from HILL_CLIMBING import vectorize_with_hill_climbing, HILL_CLIMBING_CONFIG
from SA import vectorize_with_simulated_annealing, SIMULATED_ANNEALING_CONFIG

# ============================================================================
# FUNCIÓN PRINCIPAL DE COMPARACIÓN
# ============================================================================

def compare_pipelines(
    csv_path: str,
    sizes: List[int],
    output_csv: str = "comparison_times.csv",
    output_png: str = "comparison_times.png",
    output_speedup_png: str = "comparison_speedup.png",
    # geneticos
    ga_config: Dict[str, Any] = GA_CONFIG,
    # bio inspirados
    pso_config: Dict[str, Any] = PSO_CONFIG,
    woa_config: Dict[str, Any] = WOA_CONFIG,
    # local search
    hill_climbing_config: Dict[str, Any] = HILL_CLIMBING_CONFIG,
    simulated_annealing_config: Dict[str, Any] = SIMULATED_ANNEALING_CONFIG,
    verbose: bool = False,
    train_models: bool = False
) -> pd.DataFrame:
    """
    Ejecuta experimentos comparativos entre los todos los enfoques
    
    Args:
        csv_path: Ruta al CSV
        sizes: Lista de tamaños a probar
        output_csv: Archivo CSV de salida
        output_png: Gráfica de tiempos
        output_speedup_png: Gráfica de speedup

        verbose: Si True, muestra evolución detallada
        train_models: Si True, entrena un MLP al final de la vectorizacion
    Returns:
        DataFrame con resultados
    """
    
    # ========================================================================
    # CARGAR DATASET
    # ========================================================================
    print("=" * 80)
    print("INICIANDO COMPARACIÓN DE ALGORITMOS DE BALANCEO DE CARGA")
    print("=" * 80)
    print(f"\nAlgoritmos a comparar:")
    print("  1. Secuencial (baseline)")
    print("  2. GA-Paralelo (Genetic Algorithm)")
    print("  3. PSO-Paralelo (Particle Swarm Optimization)")
    print("  4. WOA-Paralelo (Whale Optimization Algorithm)")
    print("  5. Hill Climbing-Paralelo")
    print("  6. Simulated Annealing-Paralelo")

    
    if verbose:
        print(f"\n⚙️  Modo VERBOSE activado: Se mostrarán detalles de evolución")
    
    print("\n" + "=" * 80)
    print("CARGANDO DATASET")
    print("=" * 80)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró {csv_path}")
    
    df_full = pd.read_csv(csv_path)
    
    if df_full.columns[0] in ['Unnamed: 0', 'index']:
        df_full = df_full.drop(df_full.columns[0], axis=1)
    
    print(f"✓ Dataset cargado: {len(df_full):,} registros totales")
    print(f"✓ Configuración GA: {ga_config['num_cores']} cores, "
          f"{ga_config['population_size']} población, "
          f"{ga_config['num_generations']} generaciones")
    print(f"  (con subtareas: {4 * ga_config['num_cores']} subtareas/tarea para mejor balanceo)")
    
    print(f"✓ Configuración PSO: {pso_config['num_cores']} cores, "
            f"{pso_config['num_particles']} partículas, "
            f"{pso_config['num_iterations']} iteraciones")
    
    # ========================================================================
    # EJECUTAR EXPERIMENTOS
    # ========================================================================
    results = []
    
    for size in sizes:
        if size > len(df_full):
            print(f"⚠ Tamaño solicitado ({size:,}) excede datos disponibles. Usando todos.")
            size = len(df_full)
        
        print("\n" + "=" * 80)
        print(f"TAMAÑO DEL DATASET: {size:,} tweets")
        print("=" * 80)
        
        df_subset = df_full.head(size).reset_index(drop=True)
        
        experiment_result = {
            'size': size,
            'seq_time': np.nan,
            'ga_time': np.nan,
            'pso_time': np.nan,
            'woa_time': np.nan,
            'hill_climbing_time': np.nan,
            'simulated_annealing_time': np.nan,
            'ga_speedup': np.nan,
            'pso_speedup': np.nan,
            'woa_speedup': np.nan,
            'hill_climbing_speedup': np.nan,
            'simulated_annealing_speedup': np.nan,
            'seq_accuracy': np.nan,
            'ga_accuracy': np.nan,
            'pso_accuracy': np.nan,
            'woa_accuracy': np.nan,
            'hill_climbing_accuracy': np.nan,
            'simulated_annealing_accuracy': np.nan
        }
        
        # ====================================================================
        # EXPERIMENTO 1: SECUENCIAL
        # ====================================================================
        print("\n" + "-" * 80)
        print("[SECUENCIAL - BASELINE] Iniciando...")
        print("-" * 80)
        
        try:
            train_now = train_models and (size == sizes[-1])
            _, seq_total_time, seq_stats = sequential_vectorize(
                df_subset,
                intervalo=20_000,
                train_model=train_now
            )
            experiment_result['seq_time'] = seq_total_time
            if train_now and 'mlp_stats' in seq_stats:
                experiment_result['seq_accuracy'] = seq_stats['mlp_stats']['accuracy']
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
            train_now = train_models and (size == sizes[-1])
            _, ga_total_time, ga_stats = vectorize_with_ga_load_balancing(
                df_subset,
                config=ga_config,
                verbose=verbose,
                train_model=train_now
            )
            experiment_result['ga_time'] = ga_total_time
            if train_now and 'mlp_stats' in ga_stats:
                experiment_result['ga_accuracy'] = ga_stats['mlp_stats']['accuracy']
            print(f"✓ [GA-PARALELO] Completado en {ga_total_time:.2f}s")
            
            # Mostrar info de subtareas si está disponible
            if verbose and 'num_subtasks' in ga_stats:
                print(f"  - Tareas: {ga_stats.get('num_tasks', 'N/A')}")
                print(f"  - Subtareas: {ga_stats['num_subtasks']} "
                      f"({ga_stats.get('subtasks_per_task', 'N/A')} por tarea)")
                print(f"  - Tiempo GA: {ga_stats.get('ga_time', 0):.2f}s "
                      f"({ga_stats.get('ga_time', 0)/ga_total_time*100:.1f}%)")
                print(f"  - Tiempo vectorización: {ga_stats.get('vectorization_time', 0):.2f}s "
                      f"({ga_stats.get('vectorization_time', 0)/ga_total_time*100:.1f}%)")
        except Exception as e:
            print(f"✗ [GA-PARALELO] Error: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # ====================================================================
        # EXPERIMENTO 3: PSO-PARALELO
        # ====================================================================
        print("\n" + "-" * 80)
        print("[PSO-PARALELO] Iniciando...")
        print("-" * 80)
        
        try:
            train_now = train_models and (size == sizes[-1])
            
            _, pso_total_time, pso_stats = vectorize_with_pso_load_balancing(
                df_subset,
                config=pso_config,
                verbose=verbose,
                train_model=train_now
            )
            experiment_result['pso_time'] = pso_total_time
            if train_now and 'mlp_stats' in pso_stats:
                experiment_result['pso_accuracy'] = pso_stats['mlp_stats']['accuracy']
                
            print(f"✓ [PSO-PARALELO] Completado en {pso_total_time:.2f}s")
        except Exception as e:
            print(f"✗ [PSO-PARALELO] Error: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # ====================================================================
        # EXPERIMENTO 4: WOA-PARALELO
        # ====================================================================
        print("\n" + "-" * 80)
        print("[WOA-PARALELO] Iniciando...")
        print("-" * 80)
        
        try:
            train_now = train_models and (size == sizes[-1])
            
            _, woa_total_time, woa_stats = vectorize_with_woa_load_balancing(
                df_subset,
                config=woa_config,
                verbose=verbose,
                train_model=train_now
            )
            experiment_result['woa_time'] = woa_total_time
            if train_now and 'mlp_stats' in woa_stats:
                experiment_result['pso_accuracy'] = woa_stats['mlp_stats']['accuracy']
                
            print(f"✓ [WOA-PARALELO] Completado en {woa_total_time:.2f}s")
        except Exception as e:
            print(f"✗ [WOA-PARALELO] Error: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # ====================================================================
        # EXPERIMENTO 5: Hill Climbing-PARALELO
        # ====================================================================
        print("\n" + "-" * 80)
        print("[HILL CLIMBING-PARALELO] Iniciando...")
        print("-" * 80)
        
        try:
            train_now = train_models and (size == sizes[-1])
            
            _, hill_climbing_total_time, hill_climbing_stats = vectorize_with_hill_climbing(
                df_subset,
                config=hill_climbing_config,
                verbose=verbose,
                train_model=train_now
            )
            experiment_result['hill_climbing_time'] = hill_climbing_total_time
            if train_now and 'mlp_stats' in hill_climbing_stats:
                experiment_result['hill_climbing_accuracy'] = hill_climbing_stats['mlp_stats']['accuracy']
                
            print(f"✓ [HILL CLIMBING-PARALELO] Completado en {hill_climbing_total_time:.2f}s")
        except Exception as e:
            print(f"✗ [HILL CLIMBING-PARALELO] Error: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # ====================================================================
        # EXPERIMENTO 6: Simulated Annealing-PARALELO
        # ====================================================================
        print("\n" + "-" * 80)
        print("[Simulated Annealing] Iniciando...")
        print("-" * 80)
        
        try:
            train_now = train_models and (size == sizes[-1])
            
            _, simulated_annealing_total_time, simulated_annealing_stats = vectorize_with_simulated_annealing(
                df_subset,
                config=simulated_annealing_config,
                verbose=verbose,
                train_model=train_now
            )
            experiment_result['simulated_annealing_time'] = simulated_annealing_total_time
            if train_now and 'mlp_stats' in simulated_annealing_stats:
                experiment_result['simulated_annealing_accuracy'] = simulated_annealing_stats['mlp_stats']['accuracy']
                
            print(f"✓ [SIMULATED ANNEALING-PARALELO] Completado en {simulated_annealing_total_time:.2f}s")
        except Exception as e:
            print(f"✗ [SIMULATED ANNEALING-PARALELO] Error: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # ====================================================================
        # CALCULAR MÉTRICAS
        # ====================================================================
        print("\n" + "-" * 80)
        print("CALCULANDO MÉTRICAS COMPARATIVAS")
        print("-" * 80)
        
        seq_time = experiment_result['seq_time']
        ga_time = experiment_result['ga_time']
        pso_time = experiment_result['pso_time']
        woa_time = experiment_result['woa_time']
        hill_climbing_time = experiment_result['hill_climbing_time']
        simulated_annealing_time = experiment_result['simulated_annealing_time']
        
        if not np.isnan(ga_time) and not np.isnan(seq_time) and seq_time > 0:
            ga_speedup = seq_time / ga_time
            experiment_result['ga_speedup'] = ga_speedup
            print(f"  GA Speedup: {ga_speedup:.2f}x")
        else:
            print("  GA Speedup: N/A")
        
        if not np.isnan(pso_time) and not np.isnan(seq_time) and seq_time > 0:
            pso_speedup = seq_time / pso_time
            experiment_result['pso_speedup'] = pso_speedup
            print(f"  PSO Speedup: {pso_speedup:.2f}x")
        else:
            print("  PSO Speedup: N/A")
            
        if not np.isnan(woa_time) and not np.isnan(seq_time) and seq_time > 0:
            woa_speedup = seq_time / woa_time
            experiment_result['woa_speedup'] = woa_speedup
            print(f"  WOA Speedup: {woa_speedup:.2f}x")
        else:
            print("  WOA Speedup: N/A")
            
        if not np.isnan(simulated_annealing_time) and not np.isnan(seq_time) and seq_time > 0:
            simulated_annealing_speedup = seq_time /  simulated_annealing_time
            experiment_result['simulated_annealing_speedup'] =  simulated_annealing_speedup
            print(f"  SIMULATED ANNEALING Speedup: { simulated_annealing_speedup:.2f}x")
        else:
            print("  SIMULATED ANNEALING Speedup: N/A")
            
        results.append(experiment_result)
        
        print("\n" + "-" * 80)
        print("RESUMEN DEL EXPERIMENTO")
        print("-" * 80)
        print(f"  Tamaño: {size:,} tweets")
        print(f"  Secuencial: {seq_time:.2f}s")
        print(f"  GA-Paralelo: {ga_time:.2f}s (speedup: {experiment_result['ga_speedup']:.2f}x)")
        print(f"  PSO-Paralelo: {pso_time:.2f}s (speedup: {experiment_result['pso_speedup']:.2f}x)")
        print(f"  WOA-Paralelo: {woa_time:.2f}s (speedup: {experiment_result['woa_speedup']:.2f}x)")
        print(f"  SIMULATED ANNEALING-Paralelo: {simulated_annealing_time:.2f}s (speedup: {experiment_result['simulated_annealing_speedup']:.2f}x)")
    
    # ========================================================================
    # GENERAR DATAFRAME
    # ========================================================================
    print("\n" + "=" * 80)
    print("PROCESANDO RESULTADOS")
    print("=" * 80)
    
    df_results = pd.DataFrame(results)
    
    column_order = ['size', 'seq_time', 'ga_time', 'pso_time','woa_time','hill_climbing_time','simulated_annealing_time',
                   'ga_speedup', 'pso_speedup','woa_speedup','hill_climbing_speedup','simulated_annealing_speedup']
    column_order = [col for col in column_order if col in df_results.columns]
    df_results = df_results[column_order]
    
    # ========================================================================
    # GUARDAR CSV
    # ========================================================================
    print(f"\n  Guardando tabla de resultados en {output_csv}...")
    df_results.to_csv(output_csv, index=False, float_format='%.4f')
    print(f"  ✓ Tabla guardada exitosamente")
    
    print("\n" + "=" * 80)
    print("TABLA DE RESULTADOS COMPLETA")
    print("=" * 80)
    print(df_results.to_string(index=False))
    
    # ========================================================================
    # GENERAR GRÁFICAS
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERANDO GRÁFICAS")
    print("=" * 80)
    
    # Gráfica 1: Tiempos
    print(f"  Generando gráfica de tiempos: {output_png}...")
    
    plt.figure(figsize=(12, 7))
    
    plt.plot(df_results['size'], df_results['seq_time'],
            marker='o', linewidth=2, markersize=8,
            label='Secuencial (Baseline)', color='#e74c3c')
    
    plt.plot(df_results['size'], df_results['ga_time'],
            marker='s', linewidth=2, markersize=8,
            label='GA-Paralelo', color='#3498db')
    
    plt.plot(df_results['size'], df_results['pso_time'],
            marker='^', linewidth=2, markersize=8,
            label='PSO-Paralelo', color='#2ecc71')
    
    plt.plot(df_results['size'], df_results['woa_time'],
            marker='X', linewidth=2, markersize=8,
            label='WOA-Paralelo', color="#fc4eab")
    
    plt.plot(df_results['size'], df_results['hill_climbing_time'],
            marker='*', linewidth=2, markersize=8,
            label='HILL CLIMBING-Paralelo', color="#75ffed")
    
    plt.plot(df_results['size'], df_results['simulated_annealing_time'],
            marker='D', linewidth=2, markersize=8,
            label='SIMULATED ANNEALING-Paralelo', color="#3f1fc0")
    
    plt.xlabel('Tamaño del Dataset (número de tweets)', fontsize=12)
    plt.ylabel('Tiempo de Ejecución (segundos)', fontsize=12)
    plt.title('Comparación de Tiempos: Secuencial, GA, PSO, WOA, Hill Climbing ',
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: f'{int(x):,}')
    )
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Gráfica de tiempos guardada: {output_png}")
    
    # Gráfica 2: Speedup
    print(f"  Generando gráfica de speedup: {output_speedup_png}...")
    
    plt.figure(figsize=(12, 7))
    
    num_cores = ga_config['num_cores']
    
    plt.axhline(y=num_cores, color='gray', linestyle='--',
               linewidth=2, alpha=0.7, label=f'Speedup Ideal ({num_cores}x)')
    plt.axhline(y=1, color='black', linestyle='-',
               linewidth=1, alpha=0.5, label='Sin mejora (1x)')
    
    plt.plot(df_results['size'], df_results['ga_speedup'],
            marker='s', linewidth=2, markersize=8,
            label='GA-Paralelo', color='#3498db')
    
    plt.plot(df_results['size'], df_results['pso_speedup'],
            marker='^', linewidth=2, markersize=8,
            label='PSO-Paralelo', color='#2ecc71')
    
    plt.plot(df_results['size'], df_results['woa_speedup'],
            marker='x', linewidth=2, markersize=8,
            label='WOA-Paralelo', color='#fc4eab')
    
    plt.plot(df_results['size'], df_results['hill_climbing_speedup'],
            marker='*', linewidth=2, markersize=8,
            label='Hill Climbing-Paralelo', color='#75ffed')
    
    plt.plot(df_results['size'], df_results['simulated_annealing_time'],
            marker='D', linewidth=2, markersize=8,
            label='SIMULATED ANNEALING-Paralelo', color="#3f1fc0")
    
    plt.xlabel('Tamaño del Dataset (número de tweets)', fontsize=12)
    plt.ylabel('Speedup (veces más rápido que secuencial)', fontsize=12)
    plt.title('Speedup Comparativo: GA, PSO, WOA, Hill Climbing',
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    
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
    """Script principal de comparación"""
    
    # Configuración
    DATA_FILE = 'Suicide_Detection.csv'
    SIZES = list(range(20_000, 200_001, 20_000))
    
    # Verificar requisitos
    print("=" * 80)
    print("VERIFICACIÓN DE REQUISITOS")
    print("=" * 80)
    
    if not os.path.exists(DATA_FILE):
        print(f"\n✗ Error: No se encontró el archivo {DATA_FILE}")
        sys.exit(1)
    
    print(f"✓ Archivo de datos encontrado: {DATA_FILE}")
    print(f"\nConfiguración del experimento:")
    print(f"  • Tamaños a probar: {len(SIZES)} batches")
    print(f"  • Rango: {SIZES[0]:,} - {SIZES[-1]:,} tweets")
    print(f"  • Cores disponibles: {GA_CONFIG['num_cores']}")
    
    #Preguntar sobre entrenamiento de modelo
    print(f"\n¿Deseas entrenar modelos MLP en el último batch? (y/n): ", end='')
    train_response = input().strip().lower()
    TRAIN_MODELS = (train_response == 'y')
    
    # Preguntar por modo verbose
    print(f"\n¿Deseas ver la evolución detallada? (y/n): ", end='')
    verbose_response = input().strip().lower()
    VERBOSE_MODE = (verbose_response == 'y')
    
    try:
        results_df = compare_pipelines(
            csv_path=DATA_FILE,
            sizes=SIZES,
            output_csv='comparison_times.csv',
            output_png='comparison_times.png',
            output_speedup_png='comparison_speedup.png',
            ga_config=GA_CONFIG,
            pso_config=PSO_CONFIG,
            woa_config=WOA_CONFIG,
            hill_climbing_config=HILL_CLIMBING_CONFIG,
            simulated_annealing_config=SIMULATED_ANNEALING_CONFIG,
            verbose=VERBOSE_MODE,
            train_models=TRAIN_MODELS
        )
        
        print("\n✓ Comparación completada exitosamente")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Comparación interrumpida por el usuario")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n✗ Error durante la comparación: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)