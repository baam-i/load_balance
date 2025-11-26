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
from WOA import vectorize_with_woa_load_balancing, WOA_CONFIG

# ============================================================================
# FUNCIÓN PRINCIPAL DE COMPARACIÓN
# ============================================================================

def compare_pipelines(
    csv_path: str,
    sizes: List[int],
    output_csv: str = "comparison_times_WOA.csv",
    output_png: str = "comparison_times_WOA.png",
    output_speedup_png: str = "comparison_speedup_WOA.png",
    
    # bio inspirados
    
    woa_config: Dict[str, Any] = WOA_CONFIG,
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
        woa_config: Configuración WOA

        verbose: Si True, muestra evolución detallada
    
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
    print("  2. WOA-Paralelo (Whale Optimization Algorithm)")
    
    woa_config
    
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
    num_cores = woa_config['num_cores']

    print(f"✓ Configuración WOA: {num_cores} cores, "
          f"{woa_config['num_whales']} población, "
          f"{woa_config['num_iterations']} iteraciones")
    print(f"  (con subtareas: {4 * num_cores} subtareas/tarea para mejor balanceo)")
        
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
            'woa_speedup': np.nan,
            'seq_accuracy': np.nan,
            'woa_accuracy': np.nan
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
        # EXPERIMENTO 2: WOA-PARALELO
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
                experiment_result['woa_accuracy'] = woa_stats['mlp_stats']['accuracy']
                
            print(f"✓ [WOA-PARALELO] Completado en {woa_total_time:.2f}s")
        except Exception as e:
            print(f"✗ [WOA-PARALELO] Error: {e}")
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
        woa_time = experiment_result['woa_time']
           
        if not np.isnan(woa_time) and not np.isnan(seq_time) and seq_time > 0:
            woa_speedup = seq_time / woa_time
            experiment_result['woa_speedup'] = woa_speedup
            print(f"  WOA Speedup: {woa_speedup:.2f}x")
        else:
            print("  WOA Speedup: N/A")
        
        results.append(experiment_result)
        
        print("\n" + "-" * 80)
        print("RESUMEN DEL EXPERIMENTO")
        print("-" * 80)
        print(f"  Tamaño: {size:,} tweets")
        print(f"  Secuencial: {seq_time:.2f}s")
        print(f"  WOA-Paralelo: {woa_time:.2f}s (speedup: {experiment_result['woa_speedup']:.2f}x)")
    
    # ========================================================================
    # GENERAR DATAFRAME
    # ========================================================================
    print("\n" + "=" * 80)
    print("PROCESANDO RESULTADOS")
    print("=" * 80)
    
    df_results = pd.DataFrame(results)
    
    column_order = ['size', 'seq_time','woa_time','woa_speedup']
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
    
    plt.plot(df_results['size'], df_results['woa_time'],
            marker='X', linewidth=2, markersize=8,
            label='WOA-Paralelo', color="#028206")
    
    plt.xlabel('Tamaño del Dataset (número de tweets)', fontsize=12)
    plt.ylabel('Tiempo de Ejecución (segundos)', fontsize=12)
    plt.title('Comparación de Tiempos: Secuencial, WOA',
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
    
    num_cores = woa_config['num_cores']
    
    plt.axhline(y=num_cores, color='gray', linestyle='--',
               linewidth=2, alpha=0.7, label=f'Speedup Ideal ({num_cores}x)')
    plt.axhline(y=1, color='black', linestyle='-',
               linewidth=1, alpha=0.5, label='Sin mejora (1x)')
    
    plt.plot(df_results['size'], df_results['woa_speedup'],
            marker='X', linewidth=2, markersize=8,
            label='WOA-Paralelo', color="#028206")
    
    plt.xlabel('Tamaño del Dataset (número de tweets)', fontsize=12)
    plt.ylabel('Speedup (veces más rápido que secuencial)', fontsize=12)
    plt.title('Speedup Comparativo: WOA',
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
    #print(f"  • Cores disponibles: {woa_sta}")
    
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
            output_csv='comparison_times_WOA.csv',
            output_png='comparison_times_WOA.png',
            output_speedup_png='comparison_speedup_WOA.png',
            woa_config=WOA_CONFIG,
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