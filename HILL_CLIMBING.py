"""
Hill Climbing.py

Balanceo de Carga Dinámico Basado en Hill Climbing
====================================================

VERSIÓN MEJORADA: Enfoque en UTILIZACIÓN de cores con subtareas aleatorias
Similar a GA.py, PSO.py y WOA.py, pero usando Hill Climbing como metaheurística.

Hill Climbing es un algoritmo de búsqueda local que:
1. Comienza con una solución inicial
2. Genera vecinos mediante pequeñas modificaciones
3. Se mueve al mejor vecino si mejora la solución actual
4. Se detiene cuando no encuentra mejores vecinos (óptimo local)
5. Puede usar reinicios aleatorios para escapar de óptimos locales
"""

import os, re, csv, time
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import pandas as pd

from Task import *

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

AVAILABLE_CORES = cpu_count()

HILL_CLIMBING_CONFIG = {
    'max_iterations': 100,      # Máximo número de iteraciones por búsqueda
    'num_restarts': 5,          # Número de reinicios aleatorios
    'neighbor_size': 20,        # Cantidad de vecinos a generar
    'early_stop_iters': 15,     # Iteraciones sin mejora para detener
    'perturbation_rate': 0.3,   # Tasa de perturbación para generar vecinos
    'num_cores': AVAILABLE_CORES
}

# Compilar regex una sola vez
URL_PATTERN = re.compile(r'http\S+|www\.\S+')
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")

# ============================================================================
# FUNCIONES DE VISUALIZACIÓN - ENFOCADAS EN UTILIZACIÓN
# ============================================================================

def print_utilization_stats(mapping: TaskMapping, tasks, 
                           processor_states: List[ProcessorState],
                           num_cores: int, show_details: bool = True):
    """
    Muestra estadísticas detalladas de utilización de cores.
    Analiza cómo se distribuye la carga entre los procesadores.
    """
    print("\n" + "="*80)
    print("ANÁLISIS DE UTILIZACIÓN DE CORES (Hill climbing)")
    print("="*80)
    
    # Calcular cargas finales por core
    final_loads = []
    task_counts = []
    
    for proc_id in range(num_cores):
        current_load = processor_states[proc_id].total_load()
        task_ids = mapping.get_processor_tasks(proc_id)
        new_load = sum(tasks[tid].size for tid in task_ids if tid < len(tasks))
        final_loads.append(current_load + new_load)
        task_counts.append(len(task_ids))
    
    # Métricas globales
    max_load = max(final_loads) if final_loads else 1.0
    min_load = min(final_loads) if final_loads else 0.0
    avg_load = sum(final_loads) / num_cores if num_cores > 0 else 0.0
    total_load = sum(final_loads)
    
    # Calcular utilizaciones relativas (como porcentaje del máximo)
    utilizations = [(load / max_load * 100) if max_load > 0 else 0.0 
                    for load in final_loads]
    
    min_util = min(utilizations)
    max_util = max(utilizations)
    avg_util = sum(utilizations) / num_cores
    
    # Eficiencia del sistema (qué tan cerca está el promedio del máximo)
    efficiency = (avg_load / max_load * 100) if max_load > 0 else 0.0
    
    # Uniformidad (basado en coeficiente de variación)
    std_dev = (sum((u - avg_util) ** 2 for u in utilizations) / num_cores) ** 0.5
    coef_variation = (std_dev / avg_util) if avg_util > 0 else 0.0
    
    # Clasificar cores por rango de utilización
    idle_cores = sum(1 for u in utilizations if u < 50)
    underutilized_cores = sum(1 for u in utilizations if 50 <= u < 80)
    optimal_cores = sum(1 for u in utilizations if 80 <= u < 100)
    saturated_cores = sum(1 for u in utilizations if u >= 100)
    
    print(f"\nRESUMEN DE UTILIZACIÓN:")
    print(f"  Cores disponibles:      {num_cores}")
    print(f"  Tareas asignadas:       {len(tasks)}")
    print(f"  Carga total:            {total_load:,}")
    print(f"  Fitness:                {mapping.fitness_value:.4f}")
    
    print(f"\nMÉTRICAS DE UTILIZACIÓN:")
    print(f"  Utilización mínima:     {min_util:.1f}%")
    print(f"  Utilización máxima:     {max_util:.1f}%")
    print(f"  Utilización promedio:   {avg_util:.1f}%")
    print(f"  Eficiencia del sistema: {efficiency:.1f}%")
    print(f"  Coef. variación:        {coef_variation:.3f}")
    
    print(f"\nDISTRIBUCIÓN DE CORES:")
    print(f"  Ociosos (<50%):         {idle_cores:2d} cores ({idle_cores/num_cores*100:.1f}%)")
    print(f"  Subutilizados (50-80%): {underutilized_cores:2d} cores ({underutilized_cores/num_cores*100:.1f}%)")
    print(f"  Óptimos (80-100%):      {optimal_cores:2d} cores ({optimal_cores/num_cores*100:.1f}%)")
    print(f"  Saturados (>100%):      {saturated_cores:2d} cores ({saturated_cores/num_cores*100:.1f}%)")
    
    # Evaluar calidad global del balanceo
    if optimal_cores >= num_cores * 0.8:
        status = "EXCELENTE - Utilización óptima"
    elif optimal_cores >= num_cores * 0.6:
        status = "BUENO - Mayoría bien utilizada"
    elif efficiency > 70:
        status = "ACEPTABLE - Mejorable"
    else:
        status = "POBRE - Requiere optimización"
    
    print(f"\nEVALUACIÓN:             {status}")
    
    # Mostrar detalle por core si se solicita
    if show_details:
        print(f"\nDETALLE POR CORE:")
        print(f"  {'Core':<6} {'Tareas':<8} {'Carga':<12} {'Utilización':<15} {'Barra Visual':<30}")
        print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*15} {'-'*30}")
        
        for proc_id in range(num_cores):
            load = final_loads[proc_id]
            tasks_count = task_counts[proc_id]
            util = utilizations[proc_id]
            
            # Crear barra visual de utilización
            bar_length = int(min(util, 100) / 5)  # 0-100% mapea a 0-20 chars
            bar = '█' * bar_length + '░' * (20 - bar_length)
            
            # Clasificar por utilización
            if util < 50:
                status_icon = "[IDLE]"
            elif util < 80:
                status_icon = "[LOW]"
            elif util <= 100:
                status_icon = "[OPT]"
            else:
                status_icon = "[HIGH]"
            
            print(f"  {status_icon} {proc_id:<4} {tasks_count:<8} {load:<12,} "
                  f"{util:>6.1f}%          {bar}")
        
        # Mostrar distribución de tareas si hay pocos cores
        if num_cores <= 8 and len(tasks) <= 100:
            print(f"\nTAREAS ASIGNADAS:")
            for proc_id in range(num_cores):
                task_ids = mapping.get_processor_tasks(proc_id)
                if task_ids:
                    ids_str = ", ".join(str(tid) for tid in task_ids[:15])
                    if len(task_ids) > 15:
                        ids_str += f", ... (+{len(task_ids)-15} más)"
                    print(f"  Core {proc_id:2d}: [{ids_str}]")
    
    print("="*80 + "\n")


def track_hill_climbing_evolution(current_fitness: float, best_fitness: float, 
                                  iteration: int, restart: int):
    """
    Muestra el progreso del algoritmo Hill Climbing.
    """
    # Crear barra de progreso basada en el mejor fitness
    bar_length = int(best_fitness * 20)
    bar = '█' * bar_length + '░' * (20 - bar_length)
    
    improvement = "↑" if current_fitness > best_fitness else "→"
    
    print(f"  Restart {restart}, Iter {iteration:3d}: "
          f"Current={current_fitness:.4f} {improvement} "
          f"Best={best_fitness:.4f} "
          f"[{bar}]")


# ============================================================================
# HILL CLIMBING - ENFOCADO EN UTILIZACIÓN
# ============================================================================

class HillClimbingLoadBalancer:
    """
    Hill Climbing para Balanceo de Carga Dinámico.
    VERSIÓN MEJORADA: Maximiza utilización uniforme de cores.
    
    El algoritmo funciona en las siguientes fases:
    
    1. INICIALIZACIÓN:
       - Genera una solución inicial usando estrategia greedy
       - Evalúa el fitness inicial
    
    2. BÚSQUEDA LOCAL:
       - Genera vecinos mediante perturbaciones
       - Evalúa fitness de cada vecino
       - Se mueve al mejor vecino si mejora la solución actual
       - Se detiene si no hay mejora (óptimo local)
    
    3. REINICIOS ALEATORIOS:
       - Si se alcanza un óptimo local, reinicia desde nueva posición
       - Esto ayuda a explorar diferentes regiones del espacio de búsqueda
       - Mantiene el mejor resultado global encontrado
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_processors = config['num_cores']
        self.max_iterations = config['max_iterations']
        self.num_restarts = config['num_restarts']
        self.neighbor_size = config['neighbor_size']
        self.early_stop_iters = config['early_stop_iters']
        self.perturbation_rate = config['perturbation_rate']
    
    def initialize_solution(self, num_tasks: int,
                           processor_states: List[ProcessorState]) -> TaskMapping:
        """
        Genera solución inicial usando estrategia greedy.
        Asigna cada tarea al procesador menos cargado.
        """
        mapping = TaskMapping(self.num_processors)
        
        # Calcular cargas actuales
        current_loads = [ps.total_load() for ps in processor_states]
        
        # Asignar cada tarea al procesador menos cargado
        for task_id in range(num_tasks):
            # Considerar tanto carga actual como tareas ya asignadas en este mapeo
            loads_in_mapping = [
                current_loads[p] + len(mapping.get_processor_tasks(p))
                for p in range(self.num_processors)
            ]
            least_loaded = np.argmin(loads_in_mapping)
            mapping.assign_task(least_loaded, task_id)
        
        return mapping
    
    def calculate_fitness(self, mapping: TaskMapping, tasks,
                         processor_states: List[ProcessorState]) -> float:
        """
        Calcula fitness enfocado en MAXIMIZAR UTILIZACIÓN UNIFORME.
        
        La función evalúa:
        1. Utilización mínima: Asegura que ningún core esté ocioso
        2. Eficiencia global: Mide uso general del sistema
        3. Uniformidad: Penaliza desbalances grandes
        4. Throughput: Capacidad de procesamiento del sistema
        """
        num_tasks = len(tasks)
        mapping.validate_and_fix(num_tasks)

        # Calcular cargas finales por procesador
        final_loads = []
        
        for proc_id in range(self.num_processors):
            current_load = processor_states[proc_id].total_load()
            new_task_indices = mapping.get_processor_tasks(proc_id)
            new_load = sum(
                tasks[tid].size for tid in new_task_indices 
                if 0 <= tid < num_tasks
            )
            final_loads.append(current_load + new_load)

        # Calcular métricas base
        max_load = max(final_loads) if final_loads else 1.0
        min_load = min(final_loads) if final_loads else 0.0
        avg_load = sum(final_loads) / self.num_processors if self.num_processors > 0 else 0.0
        total_load = sum(final_loads)

        # COMPONENTE 1: UTILIZACIÓN MÍNIMA
        # Asegura que todos los cores trabajen
        if max_load > 0:
            min_utilization = min_load / max_load
        else:
            min_utilization = 0.0

        # COMPONENTE 2: EFICIENCIA GLOBAL
        # Mide qué tan cerca está el promedio del máximo
        if max_load > 0:
            efficiency = avg_load / max_load
        else:
            efficiency = 0.0

        # COMPONENTE 3: UNIFORMIDAD
        # Penaliza variaciones grandes entre cores
        if avg_load > 0:
            std_dev = (sum((load - avg_load) ** 2 for load in final_loads) / self.num_processors) ** 0.5
            coef_variation = std_dev / avg_load
            uniformity = 1.0 / (1.0 + coef_variation)
        else:
            uniformity = 0.0

        # COMPONENTE 4: THROUGHPUT
        # Capacidad de procesamiento del sistema
        if max_load > 0:
            throughput_score = avg_load / max_load
        else:
            throughput_score = 0.0

        # FITNESS FINAL: Combinación ponderada
        fitness = (
            0.35 * min_utilization +
            0.25 * efficiency +
            0.30 * uniformity +
            0.10 * throughput_score
        )
        
        # Bonificación si mayoría de cores están bien utilizados
        cores_well_utilized = sum(1 for load in final_loads if load >= 0.8 * avg_load)
        if cores_well_utilized >= 0.85 * self.num_processors:
            fitness *= 1.2
        
        # Penalización si algún core está muy subutilizado
        if min_utilization < 0.5:
            fitness *= 0.7

        return max(0.0, min(1.0, fitness))
    
    def generate_neighbors(self, current_mapping: TaskMapping, 
                          num_tasks: int) -> List[TaskMapping]:
        """
        Genera vecinos mediante perturbaciones aleatorias.
        
        Estrategias de perturbación:
        1. Mover una tarea aleatoria a otro procesador
        2. Intercambiar dos tareas entre procesadores
        3. Mover bloque de tareas
        """
        neighbors = []
        
        for _ in range(self.neighbor_size):
            neighbor = current_mapping.copy()
            
            # Seleccionar estrategia de perturbación
            strategy = np.random.choice(['move', 'swap', 'block'])
            
            if strategy == 'move':
                # ESTRATEGIA 1: Mover una tarea a otro procesador
                # Seleccionar procesador origen con tareas
                procs_with_tasks = [
                    p for p in range(self.num_processors)
                    if len(neighbor.get_processor_tasks(p)) > 0
                ]
                
                if procs_with_tasks:
                    src_proc = np.random.choice(procs_with_tasks)
                    src_tasks = neighbor.get_processor_tasks(src_proc)
                    
                    if src_tasks:
                        # Seleccionar tarea aleatoria
                        task_idx = np.random.randint(0, len(src_tasks))
                        task_id = src_tasks[task_idx]
                        
                        # Seleccionar procesador destino diferente
                        dst_proc = np.random.randint(0, self.num_processors)
                        while dst_proc == src_proc and self.num_processors > 1:
                            dst_proc = np.random.randint(0, self.num_processors)
                        
                        # Mover tarea
                        neighbor.assignment[src_proc].pop(task_idx)
                        neighbor.assignment[dst_proc].append(task_id)
            
            elif strategy == 'swap':
                # ESTRATEGIA 2: Intercambiar dos tareas entre procesadores
                procs_with_tasks = [
                    p for p in range(self.num_processors)
                    if len(neighbor.get_processor_tasks(p)) > 0
                ]
                
                if len(procs_with_tasks) >= 2:
                    proc1, proc2 = np.random.choice(procs_with_tasks, 2, replace=False)
                    tasks1 = neighbor.get_processor_tasks(proc1)
                    tasks2 = neighbor.get_processor_tasks(proc2)
                    
                    if tasks1 and tasks2:
                        idx1 = np.random.randint(0, len(tasks1))
                        idx2 = np.random.randint(0, len(tasks2))
                        
                        # Intercambiar
                        neighbor.assignment[proc1][idx1], neighbor.assignment[proc2][idx2] = \
                            neighbor.assignment[proc2][idx2], neighbor.assignment[proc1][idx1]
            
            else:  # block
                # ESTRATEGIA 3: Mover bloque de tareas
                procs_with_tasks = [
                    p for p in range(self.num_processors)
                    if len(neighbor.get_processor_tasks(p)) > 1
                ]
                
                if procs_with_tasks:
                    src_proc = np.random.choice(procs_with_tasks)
                    src_tasks = neighbor.get_processor_tasks(src_proc)
                    
                    # Seleccionar tamaño de bloque (máximo 30% de las tareas)
                    block_size = min(
                        max(1, int(len(src_tasks) * self.perturbation_rate)),
                        len(src_tasks) - 1
                    )
                    
                    # Seleccionar procesador destino
                    dst_proc = np.random.randint(0, self.num_processors)
                    while dst_proc == src_proc and self.num_processors > 1:
                        dst_proc = np.random.randint(0, self.num_processors)
                    
                    # Mover bloque
                    for _ in range(block_size):
                        if neighbor.assignment[src_proc]:
                            task_id = neighbor.assignment[src_proc].pop(0)
                            neighbor.assignment[dst_proc].append(task_id)
            
            neighbor.validate_and_fix(num_tasks)
            neighbors.append(neighbor)
        
        return neighbors
    
    def climb(self, tasks, processor_states: List[ProcessorState],
             verbose: bool = False) -> TaskMapping:
        """
        Ejecuta una búsqueda local desde la solución inicial.
        Se mueve al mejor vecino mientras haya mejora.
        """
        num_tasks = len(tasks)
        
        if num_tasks == 0:
            return TaskMapping(self.num_processors)
        
        # Solución inicial
        current_solution = self.initialize_solution(num_tasks, processor_states)
        current_fitness = self.calculate_fitness(current_solution, tasks, processor_states)
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        no_improve_count = 0
        
        # Búsqueda local
        for iteration in range(self.max_iterations):
            # Generar vecinos
            neighbors = self.generate_neighbors(current_solution, num_tasks)
            
            # Evaluar todos los vecinos
            neighbor_fitness = [
                self.calculate_fitness(neighbor, tasks, processor_states)
                for neighbor in neighbors
            ]
            
            # Encontrar mejor vecino
            best_neighbor_idx = int(np.argmax(neighbor_fitness))
            best_neighbor = neighbors[best_neighbor_idx]
            best_neighbor_fitness = neighbor_fitness[best_neighbor_idx]
            
            # Si el mejor vecino mejora la solución actual, moverse
            if best_neighbor_fitness > current_fitness:
                current_solution = best_neighbor
                current_fitness = best_neighbor_fitness
                
                # Actualizar mejor global si corresponde
                if current_fitness > best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # No hay mejora, estamos en óptimo local
                no_improve_count += 1
            
            # Early stopping si no hay mejora
            if no_improve_count >= self.early_stop_iters:
                if verbose:
                    print(f"  Local optimum reached at iteration {iteration + 1}")
                break
        
        return best_solution, best_fitness
    
    def optimize(self, tasks, processor_states: List[ProcessorState],
                verbose: bool = False) -> TaskMapping:
        """
        Optimización Hill Climbing con reinicios aleatorios.
        
        Ejecuta múltiples búsquedas locales desde diferentes puntos
        iniciales para intentar encontrar mejor solución global.
        """
        num_tasks = len(tasks)
        
        if num_tasks == 0:
            return TaskMapping(self.num_processors)
        
        # Mejor solución global
        global_best_solution = None
        global_best_fitness = -1.0
        
        if verbose:
            print(f"\nHill Climbing Evolution:")
        
        # Ejecutar múltiples reinicios
        for restart in range(self.num_restarts):
            if verbose:
                print(f"\n  Restart {restart + 1}/{self.num_restarts}")
            
            # Búsqueda local desde solución inicial
            if restart == 0:
                # Primer intento: usar solución greedy
                current_solution = self.initialize_solution(num_tasks, processor_states)
            else:
                # Reinicios subsiguientes: solución aleatoria
                current_solution = TaskMapping(self.num_processors)
                for task_id in range(num_tasks):
                    proc_id = np.random.randint(0, self.num_processors)
                    current_solution.assign_task(proc_id, task_id)
            
            current_fitness = self.calculate_fitness(current_solution, tasks, processor_states)
            
            best_solution = current_solution.copy()
            best_fitness = current_fitness
            
            no_improve_count = 0
            
            # Búsqueda local
            for iteration in range(self.max_iterations):
                # Generar y evaluar vecinos
                neighbors = self.generate_neighbors(current_solution, num_tasks)
                neighbor_fitness = [
                    self.calculate_fitness(neighbor, tasks, processor_states)
                    for neighbor in neighbors
                ]
                
                # Mejor vecino
                best_neighbor_idx = int(np.argmax(neighbor_fitness))
                best_neighbor = neighbors[best_neighbor_idx]
                best_neighbor_fitness = neighbor_fitness[best_neighbor_idx]
                
                # Moverse si hay mejora
                if best_neighbor_fitness > current_fitness:
                    current_solution = best_neighbor
                    current_fitness = best_neighbor_fitness
                    
                    if current_fitness > best_fitness:
                        best_solution = current_solution.copy()
                        best_fitness = current_fitness
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
                
                if verbose and iteration % 10 == 0:
                    track_hill_climbing_evolution(
                        current_fitness, best_fitness, iteration, restart + 1
                    )
                
                # Early stopping
                if no_improve_count >= self.early_stop_iters:
                    break
            
            # Actualizar mejor global
            if best_fitness > global_best_fitness:
                global_best_solution = best_solution.copy()
                global_best_fitness = best_fitness
            
            if verbose:
                print(f"  Restart {restart + 1} completed: fitness={best_fitness:.4f}")
        
        # Configurar fitness en mejor solución
        global_best_solution.fitness_value = global_best_fitness
        global_best_solution.validate_and_fix(num_tasks)
        
        return global_best_solution


# ============================================================================
# FUNCIÓN PRINCIPAL DE VECTORIZACIÓN
# ============================================================================

def vectorize_with_hill_climbing(
    df,
    config: Dict[str, Any] = HILL_CLIMBING_CONFIG,
    verbose: bool = False,
    train_model: bool = False
) -> Tuple[Any, float, Dict[str, Any]]:
    """
    Vectorización TF-IDF con balanceo de carga basado en Hill Climbing.
    VERSIÓN ENFOCADA EN UTILIZACIÓN DE CORES CON SUBTAREAS ALEATORIAS.
    
    Proceso:
    1. Preparar datos y vocabulario TF-IDF
    2. Dividir dataset en tareas principales
    3. Subdividir cada tarea en subtareas de tamaño aleatorio
    4. Procesar ventanas de subtareas usando Hill Climbing para balanceo
    5. Vectorizar en paralelo según asignación Hill Climbing
    6. Actualizar cargas acumulativas de procesadores
    7. Repetir hasta procesar todas las subtareas
    8. Opcionalmente entrenar modelo MLP
    """
    
    num_cores = config['num_cores']
    texts = df["text"].tolist()
    total_texts = len(texts)
    
    print(f"  Usando {num_cores} cores para procesamiento paralelo")
    
    # Inicializar vectorizador TF-IDF
    vectorizer = TfidfVectorizer(
        tokenizer=None,
        lowercase=False,
        max_features=1000
    )
    
    print("  Ajustando vocabulario...")
    fit_start = time.time()
    vectorizer.fit(texts)
    fit_time = time.time() - fit_start
    print(f"  Vocabulario listo ({fit_time:.2f}s)")
    
    # Dividir dataset en tareas principales
    chunk_size = calculate_optimal_chunk_size(total_texts, num_cores)
    print(f"  Chunk size óptimo: {chunk_size}")
    
    tasks = []
    for i in range(0, total_texts, chunk_size):
        end_idx = min(i + chunk_size, total_texts)
        chunk = texts[i:end_idx]
        original_indices = list(range(i, end_idx))
        
        task = Task(
            texts=chunk,
            size=estimate_task_complexity(chunk),
            original_indices=original_indices
        )
        tasks.append(task)
    
    num_tasks_total = len(tasks)
    print(f"  Total de tareas: {num_tasks_total}")
    
    # Subdividir en subtareas con tamaños aleatorios
    num_subtasks_per_task = 4 * num_cores
    print(f"  Subtareas por tarea: {num_subtasks_per_task}")
    print(f"\n  Creando subtareas con tamaños aleatorios...")
    
    all_subtasks = []
    all_subtask_text_counts = []
    
    for task_id, task in enumerate(tasks):
        subtasks = create_subtasks_from_task(task, num_subtasks_per_task, task_id)
        all_subtasks.extend(subtasks)
        
        text_counts = [len(st.texts) for st in subtasks]
        all_subtask_text_counts.extend(text_counts)
        
        if task_id == 0:
            print(f"\n  Tarea {task_id} (ejemplo):")
            print(f"    Textos en tarea: {len(task.texts)}")
            print(f"    Subtareas creadas: {len(subtasks)}")
            print(f"    Tamaños (textos): min={min(text_counts)}, "
                  f"max={max(text_counts)}, "
                  f"avg={sum(text_counts)/len(text_counts):.1f}")
    
    num_subtasks_total = len(all_subtasks)
    print(f"\n  Total de subtareas: {num_subtasks_total}")
    print(f"  Estadísticas de tamaños:")
    print(f"    Min textos: {min(all_subtask_text_counts)}")
    print(f"    Max textos: {max(all_subtask_text_counts)}")
    print(f"    Promedio: {sum(all_subtask_text_counts)/len(all_subtask_text_counts):.1f}")
    
    # Inicializar estados de procesador (comienzan con carga cero)
    processor_states = [
        ProcessorState(processor_id=i, current_load=0.0, queue=[])
        for i in range(num_cores)
    ]
    
    # Inicializar Hill Climbing
    hc = HillClimbingLoadBalancer(config)
    
    # Estadísticas de ejecución
    stats: Dict[str, Any] = {
        'total_texts': total_texts,
        'num_tasks': num_tasks_total,
        'num_subtasks': num_subtasks_total,
        'num_cores': num_cores,
        'hc_restarts': config['num_restarts'],
        'hc_iterations': config['max_iterations'],
        'chunk_size': chunk_size,
        'subtasks_per_task': num_subtasks_per_task,
        'hc_time': 0.0,
        'vectorization_time': 0.0,
        'total_time': 0.0
    }
    
    start_total = time.time()
    
    indexed_vectors = []
    processed_subtasks = 0
    window_count = 0
    
    # Procesar subtareas en ventanas
    window_size = num_subtasks_per_task * 2
    
    while processed_subtasks < num_subtasks_total:
        window_start = processed_subtasks
        window_end = min(processed_subtasks + window_size, num_subtasks_total)
        window_subtasks = all_subtasks[window_start:window_end]
        
        print(f"\n  Procesando ventana {window_count + 1} "
              f"(subtareas {window_start}-{window_end})...")
        
        # Mostrar estado actual de utilización
        if verbose or window_count == 0:
            current_loads = [ps.total_load() for ps in processor_states]
            max_current = max(current_loads) if current_loads else 1.0
            if max_current > 0:
                utilizations = [load / max_current * 100 for load in current_loads]
                print(f"  Utilización actual: "
                      f"min={min(utilizations):.1f}%, "
                      f"max={max(utilizations):.1f}%, "
                      f"avg={sum(utilizations)/len(utilizations):.1f}%")
        
        # Ejecutar Hill Climbing para encontrar mejor asignación
        hc_start = time.time()
        best_mapping = hc.optimize(window_subtasks, processor_states, verbose=verbose)
        hc_time = time.time() - hc_start
        stats['hc_time'] += hc_time
        
        print(f"  (HC: {hc_time:.2f}s, fitness: {best_mapping.fitness_value:.4f})", 
              end=" ")
        
        if verbose:
            print()
            print_utilization_stats(best_mapping, window_subtasks, 
                                   processor_states, num_cores, show_details=True)
        
        # Ejecutar vectorización en paralelo según asignación Hill Climbing
        vec_start = time.time()
        
        # Preparar trabajo para cada procesador
        processor_work = [[] for _ in range(num_cores)]
        processor_indices = [[] for _ in range(num_cores)]
        
        for proc_id in range(num_cores):
            subtask_indices = best_mapping.get_processor_tasks(proc_id)
            for local_sid in subtask_indices:
                if 0 <= local_sid < len(window_subtasks):
                    subtask = window_subtasks[local_sid]
                    processor_work[proc_id].append(subtask)
                    processor_indices[proc_id].extend(subtask.original_indices)
        
        # Crear argumentos para workers
        work_args = [
            (
                [text for subtask in proc_subtasks for text in subtask.texts],
                vectorizer,
                proc_indices
            )
            for proc_subtasks, proc_indices in zip(processor_work, processor_indices)
            if proc_subtasks
        ]
        
        # Ejecutar vectorización en paralelo
        if work_args:
            with Pool(processes=num_cores) as pool:
                chunk_results = pool.map(vectorize_chunk, work_args)
                
                for result_list in chunk_results:
                    if result_list:
                        indexed_vectors.extend(result_list)
        
        vec_time = time.time() - vec_start
        stats['vectorization_time'] += vec_time
        
        if not verbose:
            print(f"(Vec: {vec_time:.2f}s)")
        else:
            print(f"\n  Vectorización completada en {vec_time:.2f}s")
        
        # Actualizar cargas acumulativas (NO resetear - importante para decisiones futuras)
        for proc_id in range(num_cores):
            subtask_indices = best_mapping.get_processor_tasks(proc_id)
            added_load = sum(
                window_subtasks[sid].size 
                for sid in subtask_indices 
                if sid < len(window_subtasks)
            )
            processor_states[proc_id].current_load += added_load
        
        # Mostrar nuevas cargas
        if verbose or window_count == 0:
            new_loads = [ps.total_load() for ps in processor_states]
            max_new = max(new_loads) if new_loads else 1.0
            if max_new > 0:
                new_utilizations = [load / max_new * 100 for load in new_loads]
                print(f"  Utilización actualizada: "
                      f"min={min(new_utilizations):.1f}%, "
                      f"max={max(new_utilizations):.1f}%, "
                      f"avg={sum(new_utilizations)/len(new_utilizations):.1f}%")
        
        processed_subtasks = window_end
        window_count += 1
    
    # Reconstruir matriz X completa
    print(f"  Reconstruyendo matriz...")
    
    vectors_list = [vec for _, vec in indexed_vectors]
    X = vstack(vectors_list)
    
    print(f"  OK: Matriz construida: {X.shape} ({len(indexed_vectors)} vectores)")
    
    total_time = time.time() - start_total
    stats['total_time'] = total_time
    
    print(f"\n  Resumen de tiempos:")
    print(f"  - Total: {total_time:.2f}s")
    print(f"  - Hill Climbing: {stats['hc_time']:.2f}s "
          f"({stats['hc_time']/total_time*100:.1f}%)")
    print(f"  - Vectorización: {stats['vectorization_time']:.2f}s "
          f"({stats['vectorization_time']/total_time*100:.1f}%)")
    
    # Emparejamiento con etiquetas si se va a entrenar modelo
    if train_model and 'class' in df.columns:
        print(f"\n{'='*70}")
        print(f"EMPAREJAMIENTO VECTOR-ETIQUETA")
        print(f"{'='*70}")
        
        y_original = df['class'].values
        
        # Crear array de etiquetas alineado con vectores
        y_aligned = np.zeros(len(indexed_vectors), dtype=y_original.dtype)
        
        for i, (original_idx, _) in enumerate(indexed_vectors):
            y_aligned[i] = y_original[original_idx]
        
        print(f"  OK: Etiquetas emparejadas: {len(y_aligned)}")
        print(f"  - Forma de X: {X.shape}")
        print(f"  - Forma de y: {y_aligned.shape}")
        if X.shape is not None and y_aligned.shape is not None:
            print(f"  - Coinciden?: "
                  f"{'SI' if X.shape[0] == y_aligned.shape[0] else 'NO'}")
        
        # Mostrar distribución de clases
        unique, counts = np.unique(y_aligned, return_counts=True)
        print(f"\n  Distribución de clases:")
        for label, count in zip(unique, counts):
            print(f"     Clase {label}: {count} "
                  f"({count/len(y_aligned)*100:.1f}%)")
        
        # Verificar primeras muestras
        print(f"\n  Verificando primeras 5 muestras:")
        for i in range(min(5, len(indexed_vectors))):
            original_idx, _ = indexed_vectors[i]
            text_preview = (texts[original_idx][:50] + "..." 
                          if len(texts[original_idx]) > 50 
                          else texts[original_idx])
            print(f"     Vector[{i}] -> Original[{original_idx}] "
                  f"-> Clase={y_aligned[i]}")
            print(f"         Texto: {text_preview}")
        
        print(f"{'='*70}\n")
        
        # Entrenar y evaluar modelo MLP
        mlp_stats = train_and_evaluate_mlp(X, y_aligned, 
                                          method_name="HillClimbing-Paralelo")
        stats['mlp_stats'] = mlp_stats
    
    return X, total_time, stats


# ============================================================================
# ENTRENAMIENTO Y EVALUACIÓN DE MODELO
# ============================================================================

def train_and_evaluate_mlp(X, y, method_name: str = "Método") -> Dict[str, Any]:
    """
    Entrena un MLPClassifier y evalúa su rendimiento.
    Genera matriz de confusión y reporte de clasificación.
    """
    print(f"\n{'='*70}")
    print(f"ENTRENAMIENTO DE RED NEURONAL MLP ({method_name})")
    print(f"{'='*70}")
    
    print("  Dividiendo datos (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Train: {X_train.shape[0]} muestras")
    print(f"  Test:  {X_test.shape[0]} muestras")
    
    print("\n  Entrenando MLP...")
    mlp_start = time.time()
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=50,
        random_state=42,
        verbose=False
    )
    
    mlp.fit(X_train, y_train)
    mlp_time = time.time() - mlp_start
    
    print(f"  Entrenamiento completado en {mlp_time:.2f}s")
    
    print("\n  Realizando predicciones...")
    y_pred = mlp.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nRESULTADOS:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n  Matriz de Confusión:")
    print(f"  {cm}")
    
    # Generar y guardar gráfico de matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Suicida', 'Suicida'],
                yticklabels=['No Suicida', 'Suicida'])
    plt.title(f'Matriz de Confusión - {method_name}')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.tight_layout()
    
    filename = f'confusion_matrix_{method_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Matriz de confusión guardada: {filename}")
    
    print(f"\n  Reporte de Clasificación:")
    report = classification_report(y_test, y_pred, 
                                   target_names=['No Suicida', 'Suicida'])
    print(report)
    
    print(f"{'='*70}\n")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'train_time': mlp_time,
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0]
    }


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    """Pruebas del módulo Hill Climbing con enfoque en utilización"""
    print("="*70)
    print("HILL CLIMBING LOAD BALANCER - ENFOQUE EN UTILIZACIÓN DE CORES")
    print("="*70)
    print(f"Cores disponibles: {AVAILABLE_CORES}")
    
    print("\nCargando datos...")
    df_test = pd.read_csv('Suicide_Detection.csv').head(20000)
    print(f"   Dataset: {len(df_test)} textos")
    
    if 'class' in df_test.columns:
        class_dist = df_test['class'].value_counts()
        print(f"\nDistribución de clases:")
        for label, count in class_dist.items():
            print(f"   Clase {label}: {count} ({count/len(df_test)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("INICIANDO VECTORIZACIÓN CON HILL CLIMBING")
    print("="*70)
    
    X, tiempo, stats = vectorize_with_hill_climbing(
        df_test,
        config=HILL_CLIMBING_CONFIG,
        verbose=True,
        train_model=True
    )
    
    print("\n" + "="*70)
    print("RESULTADO FINAL")
    print("="*70)
    print(f"  Textos procesados:      {X.shape[0]:,}")
    print(f"  Dimensiones del vector: {X.shape[1]:,}")
    print(f"  Tiempo total:           {tiempo:.2f}s")
    print(f"  Tiempo Hill Climbing:   {stats['hc_time']:.2f}s")
    print(f"  Tiempo vectorización:   {stats['vectorization_time']:.2f}s")
    print(f"  Cores utilizados:       {stats['num_cores']}")
    print(f"  Tareas creadas:         {stats['num_tasks']}")
    print(f"  Subtareas creadas:      {stats['num_subtasks']}")
    
    if 'mlp_stats' in stats:
        print(f"\nRESULTADOS DEL MODELO:")
        print(f"  Accuracy:               {stats['mlp_stats']['accuracy']:.4f}")
        print(f"  Tiempo entrenamiento:   "
              f"{stats['mlp_stats']['train_time']:.2f}s")
    
    print("="*70)