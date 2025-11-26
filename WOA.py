"""
WOA.py

Balanceo de Carga Dinámico Basado en Whale Optimization Algorithm (WOA)
========================================================================

VERSIÓN MEJORADA: Enfoque en UTILIZACIÓN de cores con subtareas aleatorias
Similar a GA.py y PSO.py, pero usando WOA como metaheurística de optimización.

El WOA se inspira en el comportamiento de caza de las ballenas jorobadas:
1. Encircling prey: Las ballenas rodean a su presa
2. Bubble-net attacking: Ataque en espiral con burbujas
3. Search for prey: Búsqueda exploratoria de nuevas presas
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

WOA_CONFIG = {
    'num_whales': 20,          # Tamaño de la población de ballenas
    'num_iterations': 30,       # Número de iteraciones del algoritmo
    'b': 1,                     # Constante para definir forma de espiral logarítmica
    'early_stop_iters': 15,     # Iteraciones sin mejora para detener
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
    Muestra estadísticas de UTILIZACIÓN de cores
    """
    print("\n" + "="*80)
    print("⚡ ANÁLISIS DE UTILIZACIÓN DE CORES (WOA)")
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
    
    # Calcular utilizaciones relativas
    utilizations = [(load / max_load * 100) if max_load > 0 else 0.0 
                    for load in final_loads]
    
    min_util = min(utilizations)
    max_util = max(utilizations)
    avg_util = sum(utilizations) / num_cores
    
    # Calcular eficiencia del sistema
    efficiency = (avg_load / max_load * 100) if max_load > 0 else 0.0
    
    # Calcular uniformidad
    std_dev = (sum((u - avg_util) ** 2 for u in utilizations) / num_cores) ** 0.5
    coef_variation = (std_dev / avg_util) if avg_util > 0 else 0.0
    
    # Contar cores por rango de utilización
    idle_cores = sum(1 for u in utilizations if u < 50)
    underutilized_cores = sum(1 for u in utilizations if 50 <= u < 80)
    optimal_cores = sum(1 for u in utilizations if 80 <= u < 100)
    saturated_cores = sum(1 for u in utilizations if u >= 100)
    
    print(f"\n📊 RESUMEN DE UTILIZACIÓN:")
    print(f"  Cores disponibles:      {num_cores}")
    print(f"  Tareas asignadas:       {len(tasks)}")
    print(f"  Carga total:            {total_load:,}")
    print(f"  Fitness:                {mapping.fitness_value:.4f}")
    
    print(f"\n⚡ MÉTRICAS DE UTILIZACIÓN:")
    print(f"  Utilización mínima:     {min_util:.1f}%")
    print(f"  Utilización máxima:     {max_util:.1f}%")
    print(f"  Utilización promedio:   {avg_util:.1f}%")
    print(f"  Eficiencia del sistema: {efficiency:.1f}%")
    print(f"  Coef. variación:        {coef_variation:.3f}")
    
    print(f"\n🎯 DISTRIBUCIÓN DE CORES:")
    print(f"  🔴 Ociosos (<50%):      {idle_cores:2d} cores ({idle_cores/num_cores*100:.1f}%)")
    print(f"  🟡 Subutilizados (50-80%): {underutilized_cores:2d} cores ({underutilized_cores/num_cores*100:.1f}%)")
    print(f"  🟢 Óptimos (80-100%):   {optimal_cores:2d} cores ({optimal_cores/num_cores*100:.1f}%)")
    print(f"  🔥 Saturados (>100%):   {saturated_cores:2d} cores ({saturated_cores/num_cores*100:.1f}%)")
    
    # Clasificar rendimiento
    if optimal_cores >= num_cores * 0.8:
        status = "✅ EXCELENTE - Utilización óptima"
    elif optimal_cores >= num_cores * 0.6:
        status = "🟢 BUENO - Mayoría bien utilizada"
    elif efficiency > 70:
        status = "🟡 ACEPTABLE - Mejorable"
    else:
        status = "🔴 POBRE - Requiere optimización"
    
    print(f"\n💯 EVALUACIÓN:          {status}")
    
    # Mostrar detalle por core
    if show_details:
        print(f"\n📋 DETALLE POR CORE:")
        print(f"  {'Core':<6} {'Tareas':<8} {'Carga':<12} {'Utilización':<15} {'Barra Visual':<30}")
        print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*15} {'-'*30}")
        
        for proc_id in range(num_cores):
            load = final_loads[proc_id]
            tasks_count = task_counts[proc_id]
            util = utilizations[proc_id]
            
            # Barra visual
            bar_length = int(min(util, 100) / 5)  # 0-100% -> 0-20 chars
            bar = '█' * bar_length + '░' * (20 - bar_length)
            
            # Clasificar por utilización
            if util < 50:
                color = "🔴"  # Ocioso
            elif util < 80:
                color = "🟡"  # Subutilizado
            elif util <= 100:
                color = "🟢"  # Óptimo
            else:
                color = "🔥"  # Saturado
            
            print(f"  {color} {proc_id:<4} {tasks_count:<8} {load:<12,} "
                  f"{util:>6.1f}%          {bar}")
        
        # Mostrar distribución de tareas si hay pocos cores
        if num_cores <= 8 and len(tasks) <= 100:
            print(f"\n🔍 TAREAS ASIGNADAS:")
            for proc_id in range(num_cores):
                task_ids = mapping.get_processor_tasks(proc_id)
                if task_ids:
                    ids_str = ", ".join(str(tid) for tid in task_ids[:15])
                    if len(task_ids) > 15:
                        ids_str += f", ... (+{len(task_ids)-15} más)"
                    print(f"  Core {proc_id:2d}: [{ids_str}]")
    
    print("="*80 + "\n")


def track_woa_evolution(whale_fitness: List[float], iteration: int, 
                       best_global_fitness: float):
    """Muestra el progreso del WOA con énfasis en utilización"""
    avg_fitness = sum(whale_fitness) / len(whale_fitness)
    worst_fitness = min(whale_fitness)
    
    # Crear barra de progreso
    progress = best_global_fitness
    bar_length = int(progress * 20)
    bar = '#' * bar_length + '.' * (20 - bar_length)
    
    print(f"  Iter {iteration:2d}: "
          f"Best={best_global_fitness:.4f} "
          f"Avg={avg_fitness:.4f} "
          f"[{bar}]")


# ============================================================================
# WHALE OPTIMIZATION ALGORITHM - ENFOCADO EN UTILIZACIÓN
# ============================================================================

class WOALoadBalancer:
    """
    WOA para Balanceo de Carga Dinámico
    VERSION MEJORADA: Maximiza utilización uniforme de cores
    
    El WOA simula el comportamiento de caza de las ballenas jorobadas:
    
    1. ENCIRCLING PREY (Rodear presa):
       Las ballenas identifican la posición de la presa (mejor solución)
       y rodean alrededor de ella. Se actualiza la posición usando:
       D = |C * X*(t) - X(t)|
       X(t+1) = X*(t) - A * D
       
    2. BUBBLE-NET ATTACKING (Ataque con red de burbujas):
       Dos estrategias alternadas:
       a) Shrinking encircling: A decrece de 2 a 0
       b) Spiral updating: Movimiento en espiral hacia la presa
          X(t+1) = D' * e^(bl) * cos(2*pi*l) + X*(t)
       
    3. SEARCH FOR PREY (Búsqueda de presa):
       Cuando |A| >= 1, las ballenas exploran aleatoriamente
       buscando mejores posiciones
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_processors = config['num_cores']
        self.num_whales = config['num_whales']
        self.num_iterations = config['num_iterations']
        self.b = config['b']  # Constante para la forma de la espiral logarítmica
        self.early_stop_iters = config['early_stop_iters']
    
    def initialize_population(self, num_tasks: int,
                             processor_states: List[ProcessorState]) -> List[TaskMapping]:
        """
        Inicializa población de ballenas con estrategias conscientes de utilización.
        Cada ballena representa una posible asignación de tareas a procesadores.
        """
        population = []
        
        # Calcular cargas actuales y probabilidades inversas
        current_loads = [ps.total_load() for ps in processor_states]
        max_load = max(current_loads) if current_loads else 1.0
        
        if max_load > 0:
            inverse_loads = [max_load - load for load in current_loads]
            total_inverse = sum(inverse_loads)
            
            if total_inverse > 0:
                probabilities = [inv / total_inverse for inv in inverse_loads]
            else:
                probabilities = [1.0 / self.num_processors] * self.num_processors
        else:
            probabilities = [1.0 / self.num_processors] * self.num_processors
        
        for i in range(self.num_whales):
            mapping = TaskMapping(self.num_processors)
            
            if i == 0:
                # ESTRATEGIA 1: GREEDY - Asignar al menos cargado
                for task_idx in range(num_tasks):
                    loads_in_mapping = [
                        current_loads[p] + len(mapping.get_processor_tasks(p))
                        for p in range(self.num_processors)
                    ]
                    least_loaded = np.argmin(loads_in_mapping)
                    mapping.assign_task(least_loaded, task_idx)
                    
            elif i == 1:
                # ESTRATEGIA 2: WEIGHTED RANDOM
                for task_idx in range(num_tasks):
                    processor = np.random.choice(
                        self.num_processors, 
                        p=probabilities
                    )
                    mapping.assign_task(processor, task_idx)
                    
            elif i == 2:
                # ESTRATEGIA 3: ROUND-ROBIN desde menos cargado
                sorted_procs = np.argsort(current_loads)
                for task_idx in range(num_tasks):
                    processor = sorted_procs[task_idx % self.num_processors]
                    mapping.assign_task(processor, task_idx)
                    
            else:
                # ESTRATEGIA 4+: MIXTO
                for task_idx in range(num_tasks):
                    if np.random.random() < 0.7:
                        processor = np.random.choice(
                            self.num_processors,
                            p=probabilities
                        )
                    else:
                        processor = np.random.randint(0, self.num_processors)
                    mapping.assign_task(processor, task_idx)
            
            population.append(mapping)
        
        return population
    
    def calculate_fitness(self, mapping: TaskMapping, tasks,
                         processor_states: List[ProcessorState]) -> float:
        """
        Calcula fitness enfocado en MAXIMIZAR UTILIZACIÓN UNIFORME.
        
        La función evalúa qué tan bien distribuidas están las tareas
        considerando tanto la utilización individual de cada core
        como la uniformidad global del sistema.
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

        # Calcular métricas de utilización
        max_load = max(final_loads) if final_loads else 1.0
        min_load = min(final_loads) if final_loads else 0.0
        avg_load = sum(final_loads) / self.num_processors if self.num_processors > 0 else 0.0
        total_load = sum(final_loads)

        # COMPONENTE 1: UTILIZACIÓN MÍNIMA
        # Asegura que ningún core esté ocioso
        if max_load > 0:
            min_utilization = min_load / max_load
        else:
            min_utilization = 0.0

        # COMPONENTE 2: EFICIENCIA GLOBAL
        # Mide qué tan cerca está el promedio del máximo
        ideal_load_per_core = total_load / self.num_processors
        if ideal_load_per_core > 0:
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

        # COMPONENTE 4: THROUGHPUT POTENCIAL
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
    
    def mapping_to_position(self, mapping: TaskMapping, num_tasks: int) -> np.ndarray:
        """
        Convierte un TaskMapping a un vector de posición continuo.
        
        Para WOA necesitamos representar las soluciones discretas (asignaciones)
        como vectores continuos. Cada elemento del vector representa a qué
        procesador está asignada cada tarea.
        """
        position = np.zeros(num_tasks, dtype=float)
        for proc_id in range(self.num_processors):
            for task_id in mapping.get_processor_tasks(proc_id):
                if task_id < num_tasks:
                    position[task_id] = float(proc_id)
        return position
    
    def position_to_mapping(self, position: np.ndarray) -> TaskMapping:
        """
        Convierte un vector de posición continuo a TaskMapping.
        
        Redondea cada valor a un entero válido de procesador
        y construye el mapeo correspondiente.
        """
        mapping = TaskMapping(self.num_processors)
        for task_id, proc_float in enumerate(position):
            proc_id = int(np.round(proc_float))
            proc_id = max(0, min(proc_id, self.num_processors - 1))
            mapping.assign_task(proc_id, task_id)
        return mapping
    
    def optimize(self, tasks,
                processor_states: List[ProcessorState],
                verbose: bool = False) -> TaskMapping:
        """
        Optimización WOA principal con enfoque en utilización.
        
        FASES DEL ALGORITMO:
        
        1. INICIALIZACIÓN:
           - Crear población inicial de ballenas (soluciones)
           - Evaluar fitness de cada ballena
           - Identificar la mejor ballena (líder)
        
        2. ITERACIÓN PRINCIPAL:
           Para cada ballena, decidir estrategia basada en parámetros:
           
           a) Si p < 0.5: BUBBLE-NET ATTACKING
              - Si |A| < 1: Shrinking encircling (contraer cerco)
                Acercarse a la mejor solución actual
              - Si |A| >= 1: Spiral updating (movimiento espiral)
                Moverse en espiral hacia la mejor solución
           
           b) Si p >= 0.5: SEARCH FOR PREY
              Exploración aleatoria buscando nuevas regiones
              
        3. ACTUALIZACIÓN:
           - Evaluar nuevas posiciones
           - Actualizar mejor solución global
           - Decrementar parámetro a linealmente de 2 a 0
        """
        num_tasks = len(tasks)
        
        if num_tasks == 0:
            return TaskMapping(self.num_processors)
        
        # Inicializar población de ballenas
        whales = self.initialize_population(num_tasks, processor_states)
        
        # Convertir a posiciones continuas para operaciones matemáticas
        positions = np.array([
            self.mapping_to_position(whale, num_tasks) 
            for whale in whales
        ])
        
        # Evaluar fitness inicial
        fitness_values = [
            self.calculate_fitness(mapping, tasks, processor_states)
            for mapping in whales
        ]
        
        # Identificar mejor ballena (líder del grupo)
        best_idx = int(np.argmax(fitness_values))
        best_position = positions[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        
        if verbose:
            print(f"\nEvolucion del WOA:")
            track_woa_evolution(fitness_values, 0, best_fitness)
        
        # Variables para early stopping
        no_improve_count = 0
        
        # CICLO PRINCIPAL DE WOA
        for iteration in range(self.num_iterations):
            # Parámetro a decrece linealmente de 2 a 0
            # Controla la transición entre exploración y explotación
            a = 2.0 - iteration * (2.0 / self.num_iterations)
            
            # Actualizar cada ballena
            for i in range(self.num_whales):
                # Parámetros aleatorios para el movimiento
                r1 = np.random.random()
                r2 = np.random.random()
                
                # A: Coeficiente de oscilación, controla explotación vs exploración
                A = 2.0 * a * r1 - a
                
                # C: Coeficiente aleatorio, da énfasis aleatorio a la distancia
                C = 2.0 * r2
                
                # p: Probabilidad para elegir entre espiral o cerco
                p = np.random.random()
                
                # l: Parámetro para definir forma de espiral logarítmica
                l = np.random.uniform(-1, 1)
                
                if p < 0.5:
                    # ===== BUBBLE-NET ATTACKING =====
                    if abs(A) < 1:
                        # SHRINKING ENCIRCLING MECHANISM
                        # Las ballenas contraen el cerco alrededor de la presa
                        # Se acercan a la mejor solución actual
                        D = abs(C * best_position - positions[i])
                        positions[i] = best_position - A * D
                    else:
                        # SEARCH FOR PREY (exploración)
                        # Seleccionar una ballena aleatoria como referencia
                        rand_idx = np.random.randint(0, self.num_whales)
                        X_rand = positions[rand_idx]
                        D = abs(C * X_rand - positions[i])
                        positions[i] = X_rand - A * D
                else:
                    # ===== SPIRAL UPDATING POSITION =====
                    # Movimiento en espiral hacia la mejor solución
                    # Simula el patrón de ataque helicoidal de las ballenas
                    D_prime = abs(best_position - positions[i])
                    
                    # Ecuación espiral: X(t+1) = D' * e^(bl) * cos(2*pi*l) + X*(t)
                    positions[i] = (
                        D_prime * np.exp(self.b * l) * np.cos(2 * np.pi * l) + 
                        best_position
                    )
                
                # Asegurar que las posiciones estén en rango válido [0, num_processors-1]
                positions[i] = np.clip(
                    positions[i], 
                    0, 
                    self.num_processors - 1
                )
            
            # Convertir posiciones a mapeos y evaluar
            whales = [self.position_to_mapping(pos) for pos in positions]
            fitness_values = [
                self.calculate_fitness(mapping, tasks, processor_states)
                for mapping in whales
            ]
            
            # Actualizar mejor solución global
            current_best_idx = int(np.argmax(fitness_values))
            if fitness_values[current_best_idx] > best_fitness:
                improvement = fitness_values[current_best_idx] - best_fitness
                best_position = positions[current_best_idx].copy()
                best_fitness = fitness_values[current_best_idx]
                no_improve_count = 0
                
                if improvement < 1e-6:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            if verbose:
                track_woa_evolution(fitness_values, iteration + 1, best_fitness)
            
            # Early stopping si no hay mejora
            if no_improve_count >= self.early_stop_iters:
                if verbose:
                    print(f"  Early stopping en iteracion {iteration + 1}")
                break
        
        # Construir mejor mapeo final
        best_mapping = self.position_to_mapping(best_position)
        best_mapping.fitness_value = best_fitness
        best_mapping.validate_and_fix(num_tasks)
        
        return best_mapping


# ============================================================================
# FUNCIÓN PRINCIPAL DE VECTORIZACIÓN
# ============================================================================

def vectorize_with_woa_load_balancing(
    df,
    config: Dict[str, Any] = WOA_CONFIG,
    verbose: bool = False,
    train_model: bool = False
) -> Tuple[Any, float, Dict[str, Any]]:
    """
    Vectorización TF-IDF con balanceo de carga basado en WOA
    VERSION ENFOCADA EN UTILIZACION DE CORES CON SUBTAREAS ALEATORIAS
    
    PROCESO:
    1. Preparar datos y vocabulario TF-IDF
    2. Dividir dataset en tareas principales
    3. Subdividir cada tarea en subtareas de tamaño aleatorio
    4. Procesar ventanas de subtareas usando WOA para balanceo
    5. Vectorizar en paralelo según asignación WOA
    6. Actualizar cargas acumulativas de procesadores
    7. Repetir hasta procesar todas las subtareas
    """
    
    num_cores = config['num_cores']
    texts = df["text"].tolist()
    total_texts = len(texts)
    
    print(f"  Usando {num_cores} cores para procesamiento paralelo")
    
    # Inicializar vectorizador
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
    
    # Dividir dataset en tareas
    chunk_size = calculate_optimal_chunk_size(total_texts, num_cores)
    print(f"  Chunk size optimo: {chunk_size}")
    
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
    
    # Subdividir en subtareas
    num_subtasks_per_task = 4 * num_cores
    print(f"  Subtareas por tarea: {num_subtasks_per_task}")
    print(f"\n  Creando subtareas con tamanos aleatorios...")
    
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
            print(f"    Tamanos (textos): min={min(text_counts)}, "
                  f"max={max(text_counts)}, "
                  f"avg={sum(text_counts)/len(text_counts):.1f}")
    
    num_subtasks_total = len(all_subtasks)
    print(f"\n  Total de subtareas: {num_subtasks_total}")
    print(f"  Estadisticas de tamanos:")
    print(f"    Min textos: {min(all_subtask_text_counts)}")
    print(f"    Max textos: {max(all_subtask_text_counts)}")
    print(f"    Promedio: {sum(all_subtask_text_counts)/len(all_subtask_text_counts):.1f}")
    
    # Inicializar estados de procesador
    # Cada procesador inicia con carga cero
    processor_states = [
        ProcessorState(processor_id=i, current_load=0.0)
        for i in range(num_cores)
    ]
    
    # Inicializar WOA
    woa = WOALoadBalancer(config)
    
    # Estadísticas
    stats: Dict[str, Any] = {
        'total_texts': total_texts,
        'num_tasks': num_tasks_total,
        'num_subtasks': num_subtasks_total,
        'num_cores': num_cores,
        'woa_iterations': config['num_iterations'],
        'woa_whales': config['num_whales'],
        'chunk_size': chunk_size,
        'subtasks_per_task': num_subtasks_per_task,
        'woa_time': 0.0,
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
                print(f"  Utilizacion actual: "
                      f"min={min(utilizations):.1f}%, "
                      f"max={max(utilizations):.1f}%, "
                      f"avg={sum(utilizations)/len(utilizations):.1f}%")
        
        # Ejecutar WOA para encontrar mejor asignación
        woa_start = time.time()
        best_mapping = woa.optimize(window_subtasks, processor_states, verbose=verbose)
        woa_time = time.time() - woa_start
        stats['woa_time'] += woa_time
        
        print(f"  (WOA: {woa_time:.2f}s, fitness: {best_mapping.fitness_value:.4f})", 
              end=" ")
        
        if verbose:
            print()
            print_utilization_stats(best_mapping, window_subtasks, 
                                   processor_states, num_cores, show_details=True)
        
        # Ejecutar vectorización según asignación WOA
        vec_start = time.time()
        
        processor_work = [[] for _ in range(num_cores)]
        processor_indices = [[] for _ in range(num_cores)]
        
        for proc_id in range(num_cores):
            subtask_indices = best_mapping.get_processor_tasks(proc_id)
            for local_sid in subtask_indices:
                if 0 <= local_sid < len(window_subtasks):
                    subtask = window_subtasks[local_sid]
                    processor_work[proc_id].append(subtask)
                    processor_indices[proc_id].extend(subtask.original_indices)
        
        work_args = [
            (
                [text for subtask in proc_subtasks for text in subtask.texts],
                vectorizer,
                proc_indices
            )
            for proc_subtasks, proc_indices in zip(processor_work, processor_indices)
            if proc_subtasks
        ]
        
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
            print(f"\n  Vectorizacion completada en {vec_time:.2f}s")
        
        # Actualizar cargas acumulativas (NO resetear)
        # Las cargas se mantienen para decisiones futuras
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
                print(f"  Utilizacion actualizada: "
                      f"min={min(new_utilizations):.1f}%, "
                      f"max={max(new_utilizations):.1f}%, "
                      f"avg={sum(new_utilizations)/len(new_utilizations):.1f}%")
        
        processed_subtasks = window_end
        window_count += 1
    
    # Reconstruir matriz X
    print(f"  Reconstruyendo matriz...")
    
    vectors_list = [vec for _, vec in indexed_vectors]
    X = vstack(vectors_list)
    
    print(f"  OK: Matriz construida: {X.shape} ({len(indexed_vectors)} vectores)")
    
    total_time = time.time() - start_total
    stats['total_time'] = total_time
    
    print(f"\n  Resumen de tiempos:")
    print(f"  - Total: {total_time:.2f}s")
    print(f"  - WOA: {stats['woa_time']:.2f}s "
          f"({stats['woa_time']/total_time*100:.1f}%)")
    print(f"  - Vectorizacion: {stats['vectorization_time']:.2f}s "
          f"({stats['vectorization_time']/total_time*100:.1f}%)")
    
    # Emparejamiento con etiquetas
    if train_model and 'class' in df.columns:
        print(f"\n{'='*70}")
        print(f"EMPAREJAMIENTO VECTOR-ETIQUETA")
        print(f"{'='*70}")
        
        y_original = df['class'].values
        
        y_aligned = np.zeros(len(indexed_vectors), dtype=y_original.dtype)
        
        for i, (original_idx, _) in enumerate(indexed_vectors):
            y_aligned[i] = y_original[original_idx]
        
        print(f"  OK: Etiquetas emparejadas: {len(y_aligned)}")
        print(f"  - Forma de X: {X.shape}")
        print(f"  - Forma de y: {y_aligned.shape}")
        if X.shape is not None and y_aligned.shape is not None:
            print(f"  - Coinciden?: "
                  f"{'SI' if X.shape[0] == y_aligned.shape[0] else 'NO'}")
        
        unique, counts = np.unique(y_aligned, return_counts=True)
        print(f"\n  Distribucion de clases:")
        for label, count in zip(unique, counts):
            print(f"     Clase {label}: {count} "
                  f"({count/len(y_aligned)*100:.1f}%)")
        
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
        
        mlp_stats = train_and_evaluate_mlp(X, y_aligned, 
                                          method_name="WOA-Paralelo")
        stats['mlp_stats'] = mlp_stats
    
    return X, total_time, stats


# ============================================================================
# ENTRENAMIENTO Y EVALUACIÓN DE MODELO
# ============================================================================

def train_and_evaluate_mlp(X, y, method_name: str = "Metodo") -> Dict[str, Any]:
    """Entrena un MLPClassifier y muestra matriz de confusión"""
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
    
    print(f"\n  Matriz de Confusion:")
    print(f"  {cm}")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Suicida', 'Suicida'],
                yticklabels=['No Suicida', 'Suicida'])
    plt.title(f'Matriz de Confusion - {method_name}')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.tight_layout()
    
    filename = f'confusion_matrix_{method_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Matriz de confusion guardada: {filename}")
    
    print(f"\n  Reporte de Clasificacion:")
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
    """Pruebas del módulo WOA con enfoque en utilización"""
    print("="*70)
    print("WOA LOAD BALANCER - ENFOQUE EN UTILIZACION DE CORES")
    print("="*70)
    print(f"Cores disponibles: {AVAILABLE_CORES}")
    
    print("\nCargando datos...")
    df_test = pd.read_csv('Suicide_Detection.csv').head(20000)
    print(f"   Dataset: {len(df_test)} textos")
    
    if 'class' in df_test.columns:
        class_dist = df_test['class'].value_counts()
        print(f"\nDistribucion de clases:")
        for label, count in class_dist.items():
            print(f"   Clase {label}: {count} ({count/len(df_test)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("INICIANDO VECTORIZACION CON WOA")
    print("="*70)
    
    X, tiempo, stats = vectorize_with_woa_load_balancing(
        df_test,
        config=WOA_CONFIG,
        verbose=True,
        train_model=True
    )
    
    print("\n" + "="*70)
    print("RESULTADO FINAL")
    print("="*70)
    print(f"  Textos procesados:      {X.shape[0]:,}")
    print(f"  Dimensiones del vector: {X.shape[1]:,}")
    print(f"  Tiempo total:           {tiempo:.2f}s")
    print(f"  Tiempo WOA:             {stats['woa_time']:.2f}s")
    print(f"  Tiempo vectorizacion:   {stats['vectorization_time']:.2f}s")
    print(f"  Cores utilizados:       {stats['num_cores']}")
    print(f"  Tareas creadas:         {stats['num_tasks']}")
    print(f"  Subtareas creadas:      {stats['num_subtasks']}")
    
    if 'mlp_stats' in stats:
        print(f"\nRESULTADOS DEL MODELO:")
        print(f"  Accuracy:               {stats['mlp_stats']['accuracy']:.4f}")
        print(f"  Tiempo entrenamiento:   "
              f"{stats['mlp_stats']['train_time']:.2f}s")
    
    print("="*70)