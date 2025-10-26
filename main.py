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

# Importar pipelines
from GA import vectorize_with_ga_load_balancing, GA_CONFIG
from secuencial import sequential_vectorize

# Importar PSO
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
    enable_pso: bool = True,
    verbose: bool = False,
    train_models: bool = False
) -> pd.DataFrame:
    """
    Ejecuta experimentos comparativos entre los tres enfoques
    
    Args:
        csv_path: Ruta al CSV
        sizes: Lista de tamaños a probar
        output_csv: Archivo CSV de salida
        output_png: Gráfica de tiempos
        output_speedup_png: Gráfica de speedup
        ga_config: Configuración GA
        pso_config: Configuración PSO
        enable_pso: Si True, ejecuta PSO
        verbose: Si True, muestra evolución detallada
    
    Returns:
        DataFrame con resultados
    """
    
    if ga_config is None:
        ga_config = GA_CONFIG.copy()
    
    if pso_config is None and PSO_AVAILABLE:
        pso_config = PSO_CONFIG.copy()
    
    use_pso = PSO_AVAILABLE and enable_pso
    
    # ========================================================================
    # CARGAR DATASET
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
    
    if use_pso:
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
            'ga_speedup': np.nan,
            'pso_speedup': np.nan,
            'seq_accuracy': np.nan,
            'ga_accuracy': np.nan,
            'pso_accuracy': np.nan
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
            if train_now and 'mlp_stats' in ga_stats:
                experiment_result['ga_accuracy'] = ga_stats['mlp_stats']['accuracy']
            print(f"✓ [GA-PARALELO] Completado en {ga_total_time:.2f}s")
        except Exception as e:
            print(f"✗ [GA-PARALELO] Error: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # ====================================================================
        # EXPERIMENTO 3: PSO-PARALELO
        # ====================================================================
        if use_pso:
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
                if train_now and 'mlp_stats' in pso_stats:
                    experiment_result['pso_accuracy'] = pso_stats['mlp_stats']['accuracy']
                    
                print(f"✓ [PSO-PARALELO] Completado en {pso_total_time:.2f}s")
            except Exception as e:
                print(f"✗ [PSO-PARALELO] Error: {e}")
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
        
        if (use_pso and not np.isnan(ga_time) and not np.isnan(pso_time) and
            ga_time > 0 and pso_time > 0):
            
            if ga_time < pso_time:
                faster_method = "GA"
                improvement = ((pso_time - ga_time) / pso_time) * 100
            else:
                faster_method = "PSO"
                improvement = ((ga_time - pso_time) / ga_time) * 100
            
            print(f"\n  {faster_method} es {improvement:.1f}% más rápido")
        
        results.append(experiment_result)
        
        print("\n" + "-" * 80)
        print("RESUMEN DEL EXPERIMENTO")
        print("-" * 80)
        print(f"  Tamaño: {size:,} tweets")
        print(f"  Secuencial: {seq_time:.2f}s")
        print(f"  GA-Paralelo: {ga_time:.2f}s (speedup: {experiment_result['ga_speedup']:.2f}x)")
        if use_pso:
            print(f"  PSO-Paralelo: {pso_time:.2f}s (speedup: {experiment_result['pso_speedup']:.2f}x)")
    
    # ========================================================================
    # GENERAR DATAFRAME
    # ========================================================================
    print("\n" + "=" * 80)
    print("PROCESANDO RESULTADOS")
    print("=" * 80)
    
    df_results = pd.DataFrame(results)
    
    column_order = ['size', 'seq_time', 'ga_time', 'pso_time',
                   'ga_speedup', 'pso_speedup']
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
    
    if use_pso:
        plt.plot(df_results['size'], df_results['pso_time'],
                marker='^', linewidth=2, markersize=8,
                label='PSO-Paralelo', color='#2ecc71')
    
    plt.xlabel('Tamaño del Dataset (número de tweets)', fontsize=12)
    plt.ylabel('Tiempo de Ejecución (segundos)', fontsize=12)
    plt.title('Comparación de Tiempos: Secuencial vs GA vs PSO',
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
    
    if use_pso:
        plt.plot(df_results['size'], df_results['pso_speedup'],
                marker='^', linewidth=2, markersize=8,
                label='PSO-Paralelo', color='#2ecc71')
    
    plt.xlabel('Tamaño del Dataset (número de tweets)', fontsize=12)
    plt.ylabel('Speedup (veces más rápido que secuencial)', fontsize=12)
    plt.title('Speedup Comparativo: GA vs PSO',
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
    SIZES = list(range(180_000, 200_001, 20_000))
    
    GA_TEST_CONFIG = GA_CONFIG
    PSO_TEST_CONFIG = PSO_CONFIG if PSO_AVAILABLE else None
    
    # Verificar requisitos
    print("=" * 80)
    print("VERIFICACIÓN DE REQUISITOS")
    print("=" * 80)
    
    if not os.path.exists(DATA_FILE):
        print(f"\n✗ Error: No se encontró el archivo {DATA_FILE}")
        sys.exit(1)
    
    print(f"✓ Archivo de datos encontrado: {DATA_FILE}")
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
    
    #Preguntar sobre entrenamiento de modelo
    print(f"\n¿Deseas entrenar modelos MLP en el último batch? (y/n): ", end='')
    train_response = input().strip().lower()
    TRAIN_MODELS = (train_response == 'y')
    
    # Preguntar por modo verbose
    print(f"\n¿Deseas ver la evolución detallada? (y/n): ", end='')
    verbose_response = input().strip().lower()
    VERBOSE_MODE = (verbose_response == 'y')
    
    print(f"\n¿Deseas continuar con la comparación? (y/n): ", end='')
    response = input().strip().lower()
    
    if response != 'y':
        print("Comparación cancelada.")
        sys.exit(0)
    
    try:
        results_df = compare_pipelines(
            csv_path=DATA_FILE,
            sizes=SIZES,
            output_csv='comparison_times.csv',
            output_png='comparison_times.png',
            output_speedup_png='comparison_speedup.png',
            ga_config=GA_TEST_CONFIG,
            pso_config=PSO_TEST_CONFIG,
            enable_pso=PSO_AVAILABLE,
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