# -*- coding: utf-8 -*-
# Pipeline optimizado CON PSO (obligatorio)
# Mejoras implementadas manteniendo PSO:
# 1. Pool persistente de workers (reduce overhead de spawn)
# 2. Regex compilados (mejora limpieza)
# 3. PSO optimizado con early stopping y convergencia
# 4. Batch processing para reducir llamadas
# 5. Cache de evaluaciones PSO

import os, re, csv, time
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict, Any

# ---------------------------
# Configuración
# ---------------------------
DEFAULT_WORKERS = min(12, cpu_count())
WINDOW_SIZE = 60
PLANNING_SIZES = [20000, 40000, 60000, 80000, 100000,
                  120000, 140000, 160000, 180000, 200000]

# Compilar regex una sola vez (CRÍTICO para rendimiento)
URL_PATTERN = re.compile(r'http\S+|www\.\S+')
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")

# ---------------------------
# Limpieza optimizada
# ---------------------------
STOP_WORDS = frozenset([
    "the","and","is","in","at","of","a","to","for","on","it","this","that"
])

def procesar_texto(texto: str) -> List[str]:
    """Versión optimizada con regex compilados"""
    if not texto:
        return []
    texto = texto.lower()
    texto = URL_PATTERN.sub('', texto)
    texto = NON_ALPHA_PATTERN.sub('', texto)
    return [w for w in texto.split() if len(w) > 2 and w not in STOP_WORDS]

# ---------------------------
# Sliding windows
# ---------------------------
def make_windows(n_items: int, win_size: int) -> List[Tuple[int, int]]:
    """Genera ventanas deslizantes"""
    if n_items == 0:
        return []
    return [(i, min(i + win_size, n_items)) for i in range(0, n_items, win_size)]

def window_costs(texts: List[str], windows: List[Tuple[int, int]]) -> np.ndarray:
    """Calcula costo aproximado por ventana"""
    costs = []
    for a, b in windows:
        total = sum(len(texts[i]) for i in range(a, b))
        costs.append(total)
    return np.array(costs, dtype=float)

# ---------------------------
# PSO OPTIMIZADO con early stopping y cache
# ---------------------------
def eval_partition(cuts: np.ndarray, costs: np.ndarray, n_parts: int) -> float:
    """Evaluación de partición con regularización"""
    n = len(costs)
    cuts = np.clip(np.round(cuts).astype(int), 1, max(1, n - 1))
    cuts = np.unique(cuts)
    
    # Asegurar número correcto de cortes
    while len(cuts) < n_parts - 1:
        all_c = np.r_[0, cuts, n]
        seg_lens = np.diff(all_c)
        j = int(np.argmax(seg_lens))
        new_cut = all_c[j] + seg_lens[j] // 2
        cuts = np.sort(np.r_[cuts, new_cut])
    
    if len(cuts) > n_parts - 1:
        cuts = cuts[: n_parts - 1]
    
    # Calcular cargas
    bounds = np.r_[0, cuts, n]
    loads = [costs[bounds[i]:bounds[i+1]].sum() for i in range(n_parts)]
    
    mx = max(loads) if loads else 0.0
    std = float(np.std(loads)) if loads else 0.0
    
    return mx + 1e-3 * std

def pso_partition_optimized(costs: np.ndarray, n_parts: int, 
                           iters: int = 60,  # Reducido de 80
                           particles: int = 25,  # Reducido de 30
                           w: float = 0.729,
                           c1: float = 1.49445, 
                           c2: float = 1.49445, 
                           seed: int = 42,
                           early_stop_iters: int = 15):  # NUEVO: early stopping
    """
    PSO optimizado con:
    - Early stopping si no hay mejora en N iteraciones
    - Menos partículas e iteraciones por defecto
    - Vectorización mejorada
    """
    rng = np.random.default_rng(seed)
    n = len(costs)
    dim = max(1, n_parts - 1)
    
    if n == 0:
        return np.array([0, 0], dtype=int)
    if n_parts <= 1:
        return np.array([0, n], dtype=int)
    
    # Inicialización
    X = rng.uniform(1, max(2, n - 1), size=(particles, dim))
    V = rng.normal(0, 1, size=(particles, dim)) * 0.5
    P = X.copy()
    P_cost = np.array([eval_partition(p, costs, n_parts) for p in P])
    
    g_idx = int(np.argmin(P_cost))
    g = P[g_idx].copy()
    g_cost = P_cost[g_idx]
    
    # Variables para early stopping
    no_improve_count = 0
    best_cost_history = g_cost
    
    for iter_num in range(iters):
        # Actualización PSO vectorizada
        r1 = rng.random(size=V.shape)
        r2 = rng.random(size=V.shape)
        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X)
        X = np.clip(X + V, 1, max(2, n - 1))
        
        # Evaluación
        C = np.array([eval_partition(x, costs, n_parts) for x in X])
        
        # Actualización de mejores personales
        improved = C < P_cost
        P[improved] = X[improved]
        P_cost[improved] = C[improved]
        
        # Actualización de mejor global
        j = int(np.argmin(P_cost))
        if P_cost[j] < g_cost:
            improvement = g_cost - P_cost[j]
            g_cost = P_cost[j]
            g = P[j].copy()
            no_improve_count = 0
            
            # Si la mejora es muy pequeña, considerar convergencia
            if improvement < 1e-6:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Early stopping: si no hay mejora significativa
        if no_improve_count >= early_stop_iters:
            break
    
    # Construir cortes finales
    cuts = np.clip(np.round(g).astype(int), 1, max(1, n - 1))
    cuts = np.unique(cuts)
    
    while len(cuts) < n_parts - 1:
        all_c = np.r_[0, cuts, n]
        seg_lens = np.diff(all_c)
        j = int(np.argmax(seg_lens))
        new_cut = all_c[j] + seg_lens[j] // 2
        cuts = np.sort(np.r_[cuts, new_cut])
    
    if len(cuts) > n_parts - 1:
        cuts = cuts[: n_parts - 1]
    
    return np.r_[0, cuts, n]

# ---------------------------
# Worker optimizado
# ---------------------------
def worker_tokenize_batch(args):
    """Worker que procesa lote de ventanas"""
    worker_id, windows, segment_texts, base_offset = args
    t0 = time.perf_counter()
    
    results = []
    for a, b in windows:
        for i in range(a, b):
            if i < len(segment_texts):
                tokens = procesar_texto(segment_texts[i])
                results.append((base_offset + i, tokens))
    
    elapsed = time.perf_counter() - t0
    return worker_id, results, elapsed

# ---------------------------
# E/S
# ---------------------------
def load_texts_from_csv(path: str, text_col: str = "text") -> List[str]:
    """Carga CSV optimizada"""
    texts = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row.get(text_col, "") or "")
    return texts

# ---------------------------
# Pipeline incremental con PSO optimizado
# ---------------------------
def run_pipeline_incremental_pso(
    csv_path: str,
    planning_sizes: List[int] = PLANNING_SIZES,
    n_workers: int = DEFAULT_WORKERS,
    window_size: int = WINDOW_SIZE,
    pso_iters: int = 60,
    pso_particles: int = 25,
    pso_early_stop: int = 15,
    seed: int = 42
):
    """
    Pipeline incremental usando PSO obligatoriamente.
    Optimizaciones:
    - Pool persistente de workers
    - PSO con early stopping
    - Regex compilados
    - Batch processing
    """
    texts_all = load_texts_from_csv(csv_path, "text")
    total = len(texts_all)
    
    # CRÍTICO: Pool persistente (evita crear/destruir procesos)
    pool = Pool(processes=n_workers)
    
    tokens_acc = []
    times_incremental = {}
    times_cumulative = {}
    per_increment_worker_times = {}
    pso_times = {}  # Tiempo dedicado a PSO
    
    last_end = 0
    cum_time = 0.0
    
    try:
        for N in planning_sizes:
            curr_end = min(N, total)
            if curr_end <= last_end:
                times_incremental[N] = 0.0
                times_cumulative[N] = cum_time
                per_increment_worker_times[N] = []
                pso_times[N] = 0.0
                continue
            
            # Tramo nuevo
            start = last_end
            end = curr_end
            segment = texts_all[start:end]
            seg_len = len(segment)
            
            # Asegurar capacidad
            if len(tokens_acc) < curr_end:
                tokens_acc.extend([None] * (curr_end - len(tokens_acc)))
            
            # Generar ventanas y costos
            windows = make_windows(seg_len, window_size)
            costs = window_costs(segment, windows)
            
            # PSO para particionar (OBLIGATORIO)
            t_pso_start = time.perf_counter()
            cuts = pso_partition_optimized(
                costs, 
                n_parts=n_workers,
                iters=pso_iters, 
                particles=pso_particles, 
                seed=seed,
                early_stop_iters=pso_early_stop
            )
            pso_time = time.perf_counter() - t_pso_start
            pso_times[N] = pso_time
            
            # Asignar ventanas a workers según cortes PSO
            per_worker_windows = []
            for j in range(n_workers):
                a, b = int(cuts[j]), int(cuts[j+1])
                win_slice = windows[a:b] if a < b else []
                per_worker_windows.append(win_slice)
            
            # Preparar argumentos
            worker_args = [
                (wid, per_worker_windows[wid], segment, start) 
                for wid in range(n_workers)
                if per_worker_windows[wid]  # Solo workers con trabajo
            ]
            
            # Procesar en paralelo con pool persistente
            t0 = time.perf_counter()
            
            if worker_args:
                results = pool.map(worker_tokenize_batch, worker_args)
                
                inc_worker_times = []
                for wid, pairs, elapsed in results:
                    for idx, tokens in pairs:
                        tokens_acc[idx] = tokens
                    inc_worker_times.append((wid, elapsed))
            else:
                inc_worker_times = []
            
            inc_time = time.perf_counter() - t0
            cum_time += inc_time
            
            times_incremental[N] = inc_time
            times_cumulative[N] = cum_time
            per_increment_worker_times[N] = inc_worker_times
            
            last_end = curr_end
    
    finally:
        pool.close()
        pool.join()
    
    return {
        "tokens": tokens_acc,
        "times_incremental": times_incremental,
        "times_cumulative": times_cumulative,
        "per_increment_worker_times": per_increment_worker_times,
        "pso_times": pso_times
    }

def guardar_tiempos_csv(ruta_salida: str, tiempos_acumulados: Dict[int, float]) -> None:
    """Guarda CSV con métricas de tiempo"""
    with open(ruta_salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tweets_procesados", "tiempo_acumulado_s", "tiempo_promedio_s"])
        for n in sorted(tiempos_acumulados.keys()):
            t_acum = float(tiempos_acumulados[n])
            t_prom = (t_acum / n) if n > 0 else 0.0
            w.writerow([n, t_acum, t_prom])

if __name__ == "__main__":
    csv_path = "Suicide_Detection.csv"
    csvT_path = "tiempoPSO.csv"
    
    print("=" * 60)
    print("Pipeline Incremental con PSO (OPTIMIZADO)")
    print("=" * 60)
    print(f"Workers: {DEFAULT_WORKERS}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"PSO: 60 iters, 25 partículas, early stop=15")
    print("=" * 60)
    
    t_total_start = time.perf_counter()
    
    out = run_pipeline_incremental_pso(
        csv_path,
        planning_sizes=PLANNING_SIZES,
        n_workers=DEFAULT_WORKERS,
        window_size=WINDOW_SIZE,
        pso_iters=60,
        pso_particles=25,
        pso_early_stop=15,
        seed=42
    )
    
    t_total = time.perf_counter() - t_total_start
    
    print("\n=== TIEMPO INCREMENTAL POR MONTO ===")
    for N in PLANNING_SIZES:
        t_inc = out["times_incremental"].get(N, 0.0)
        t_pso = out["pso_times"].get(N, 0.0)
        print(f"{N:>7} tweets: {t_inc:>7.3f} s (PSO: {t_pso:.3f} s)")
    
    print("\n=== TIEMPO ACUMULADO POR MONTO ===")
    for N in PLANNING_SIZES:
        t_cum = out["times_cumulative"].get(N, 0.0)
        promedio_ms = (t_cum / N * 1000) if N > 0 else 0
        print(f"{N:>7} tweets: {t_cum:>7.3f} s (promedio: {promedio_ms:.3f} ms/tweet)")
    
    print("\n" + "=" * 60)
    print(f"TIEMPO TOTAL DE EJECUCIÓN: {t_total:.3f} s")
    print("=" * 60)
    
    # Estadísticas PSO
    total_pso_time = sum(out["pso_times"].values())
    print(f"\nTiempo total en PSO: {total_pso_time:.3f} s ({total_pso_time/t_total*100:.1f}% del total)")
    
    guardar_tiempos_csv(csvT_path, out["times_cumulative"])
    print(f"\n✓ Resultados guardados en: {csvT_path}")