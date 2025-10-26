"""
PSO.py

Balanceo de Carga Dinámico Basado en Particle Swarm Optimization (PSO)
======================================================================

Similar a GA.py, pero usando PSO como metaheurística de optimización.
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
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

STOP_WORDS = frozenset([
    "the","and","is","in","at","of","a","to","for","on","it","this","that"
])

AVAILABLE_CORES = cpu_count()

PSO_CONFIG = {
    'num_particles': 25,
    'num_iterations': 60,
    'w': 0.729,
    'c1': 1.49445,
    'c2': 1.49445,
    'early_stop_iters': 15,
    'num_cores': AVAILABLE_CORES
}

# Compilar regex una sola vez
URL_PATTERN = re.compile(r'http\S+|www\.\S+')
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")

# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class Task:
    """Representa una tarea de vectorización"""
    task_id: int
    texts: List[str]
    size: int

@dataclass
class ProcessorState:
    """Estado actual de un procesador"""
    processor_id: int
    current_load: float
    queue: List[Task]
    
    def total_load(self) -> float:
        return self.current_load + sum(t.size for t in self.queue)

class TaskMapping:
    """Representa un mapeo de tareas a procesadores"""
    
    def __init__(self, num_processors: int):
        self.num_processors = num_processors
        self.assignment: List[List[int]] = [[] for _ in range(num_processors)]
        self.fitness_value: float = 0.0
    
    def assign_task(self, processor_id: int, task_id: int):
        self.assignment[processor_id].append(task_id)
    
    def get_processor_tasks(self, processor_id: int) -> List[int]:
        return self.assignment[processor_id]
    
    def copy(self) -> 'TaskMapping':
        new_mapping = TaskMapping(self.num_processors)
        new_mapping.assignment = [queue[:] for queue in self.assignment]
        new_mapping.fitness_value = self.fitness_value
        return new_mapping
    
    def validate_and_fix(self, num_tasks: int):
        for proc_id in range(self.num_processors):
            self.assignment[proc_id] = [
                tid for tid in self.assignment[proc_id]
                if 0 <= tid < num_tasks
            ]

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def procesar_texto(texto: str) -> List[str]:
    """Limpieza optimizada con regex compilados"""
    if not texto:
        return []
    texto = texto.lower()
    texto = URL_PATTERN.sub('', texto)
    texto = NON_ALPHA_PATTERN.sub('', texto)
    return [w for w in texto.split() if len(w) > 2 and w not in STOP_WORDS]

def estimate_task_complexity(texts: List[str]) -> int:
    """Estima el tiempo de procesamiento"""
    if not texts:
        return 1
    base_cost = len(texts) * 100
    length_cost = sum(len(text) for text in texts)
    return max(1, base_cost + length_cost)

def vectorize_chunk(args: Tuple[List[str], TfidfVectorizer]) -> Any:
    """Worker para vectorizar un chunk"""
    texts, vectorizer = args
    if not texts:
        return None
    return vectorizer.transform(texts)

def calculate_optimal_chunk_size(total_texts: int, num_cores: int) -> int:
    """Calcula chunk_size óptimo"""
    ideal_tasks_per_core = 4
    target_total_tasks = num_cores * ideal_tasks_per_core
    chunk_size = total_texts // target_total_tasks
    
    if total_texts < 10000:
        chunk_size = max(200, min(1000, chunk_size))
    elif total_texts < 50000:
        chunk_size = max(500, min(2000, chunk_size))
    else:
        chunk_size = max(1000, min(5000, chunk_size))
    
    return chunk_size

def calculate_optimal_window_size(total_tasks: int, num_cores: int) -> int:
    """Calcula window_size óptimo"""
    if total_tasks <= num_cores * 5:
        return total_tasks
    elif total_tasks <= num_cores * 15:
        return (total_tasks + 1) // 2
    else:
        window_size = num_cores * 5
        return min(window_size, total_tasks)

# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def print_distribution_stats(mapping: TaskMapping, tasks: List[Task], 
                            num_cores: int, show_details: bool = True):
    """
    Muestra estadísticas detalladas de la distribución de tareas
    """
    print("\n" + "="*70)
    print("📊 ANÁLISIS DE DISTRIBUCIÓN DE CARGA (PSO)")
    print("="*70)
    
    # Calcular cargas por core
    loads = []
    task_counts = []
    
    for proc_id in range(num_cores):
        task_ids = mapping.get_processor_tasks(proc_id)
        total_load = sum(tasks[tid].size for tid in task_ids if tid < len(tasks))
        loads.append(total_load)
        task_counts.append(len(task_ids))
    
    # Estadísticas globales
    min_load = min(loads) if loads else 0
    max_load = max(loads) if loads else 0
    avg_load = sum(loads) / num_cores if num_cores > 0 else 0
    total_load = sum(loads)
    
    # Calcular desbalance
    if max_load > 0:
        desbalance = (max_load - min_load) / max_load * 100
    else:
        desbalance = 0.0
    
    # Calcular desviación estándar
    variance = sum((load - avg_load) ** 2 for load in loads) / num_cores
    std_dev = variance ** 0.5
    coef_variation = (std_dev / avg_load * 100) if avg_load > 0 else 0
    
    # Mostrar resumen
    print(f"\n📈 RESUMEN GENERAL:")
    print(f"  Total de tareas:        {len(tasks)}")
    print(f"  Número de cores:        {num_cores}")
    print(f"  Carga total:            {total_load:,}")
    print(f"  Carga promedio:         {avg_load:,.0f}")
    print(f"  Carga mínima:           {min_load:,}")
    print(f"  Carga máxima:           {max_load:,}")
    print(f"  Fitness:                {mapping.fitness_value:.4f}")
    
    print(f"\n⚖️  MÉTRICAS DE BALANCE:")
    print(f"  Desbalance:             {desbalance:.2f}%")
    print(f"  Desviación estándar:    {std_dev:,.0f}")
    print(f"  Coef. variación:        {coef_variation:.2f}%")
    
    # Clasificar balance
    if desbalance < 5:
        balance_status = "✅ EXCELENTE"
    elif desbalance < 15:
        balance_status = "🟢 BUENO"
    elif desbalance < 30:
        balance_status = "🟡 REGULAR"
    else:
        balance_status = "🔴 MALO"
    
    print(f"  Estado:                 {balance_status}")
    
    # Mostrar detalle por core si se solicita
    if show_details:
        print(f"\n📋 DETALLE POR CORE:")
        print(f"  {'Core':<6} {'Tareas':<8} {'Carga':<12} {'% Carga':<10} {'Barra':<30}")
        print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*10} {'-'*30}")
        
        for proc_id in range(num_cores):
            load = loads[proc_id]
            tasks_count = task_counts[proc_id]
            load_pct = (load / avg_load * 100) if avg_load > 0 else 0
            
            # Crear barra visual
            bar_length = int(min(load_pct / 100 * 20, 20))
            bar = '█' * bar_length + '░' * (20 - bar_length)
            
            # Colorear según desviación
            if load < avg_load * 0.9:
                color = "🔵"
            elif load > avg_load * 1.1:
                color = "🔴"
            else:
                color = "🟢"
            
            print(f"  {color} {proc_id:<4} {tasks_count:<8} {load:<12,} "
                  f"{load_pct:>6.1f}%    {bar}")
        
        # Mostrar task IDs si hay pocos cores
        if num_cores <= 8:
            print(f"\n🔍 TAREAS ASIGNADAS:")
            for proc_id in range(num_cores):
                task_ids = mapping.get_processor_tasks(proc_id)
                if task_ids:
                    ids_str = ", ".join(str(tid) for tid in task_ids[:10])
                    if len(task_ids) > 10:
                        ids_str += f", ... (+{len(task_ids)-10} más)"
                    print(f"  Core {proc_id:2d}: [{ids_str}]")
    
    print("="*70 + "\n")

def track_pso_evolution(swarm_fitness: List[float], iteration: int, 
                       best_global_fitness: float):
    """Muestra el progreso del PSO"""
    avg_fitness = sum(swarm_fitness) / len(swarm_fitness)
    worst_fitness = min(swarm_fitness)
    
    # Crear barra de progreso
    progress = best_global_fitness
    bar_length = int(progress * 20)
    bar = '█' * bar_length + '░' * (20 - bar_length)
    
    print(f"  Iter {iteration:2d}: "
          f"Best={best_global_fitness:.4f} "
          f"Avg={avg_fitness:.4f} "
          f"Worst={worst_fitness:.4f} "
          f"[{bar}]")

# ============================================================================
# PARTICLE SWARM OPTIMIZATION
# ============================================================================

class PSOLoadBalancer:
    """PSO para Balanceo de Carga Dinámico"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_processors = config['num_cores']
        self.num_particles = config['num_particles']
        self.num_iterations = config['num_iterations']
        self.w = config['w']
        self.c1 = config['c1']
        self.c2 = config['c2']
        self.early_stop_iters = config['early_stop_iters']
        self.H = 1.2
        self.L = 0.8
    
    def initialize_swarm(self, num_tasks: int) -> List[TaskMapping]:
        """Inicializa enjambre de partículas"""
        swarm = []
        
        for i in range(self.num_particles):
            mapping = TaskMapping(self.num_processors)
            
            if i == 0:
                # Partícula 1: Round-robin
                for task_idx in range(num_tasks):
                    processor = task_idx % self.num_processors
                    mapping.assign_task(processor, task_idx)
            else:
                # Resto: Aleatorio
                for task_idx in range(num_tasks):
                    processor = np.random.randint(0, self.num_processors)
                    mapping.assign_task(processor, task_idx)
            
            swarm.append(mapping)
        
        return swarm
    
    def calculate_fitness(self, mapping: TaskMapping, tasks: List[Task],
                         processor_states: List[ProcessorState]) -> float:
        """Función de fitness (igual que en GA)"""
        num_tasks = len(tasks)
        mapping.validate_and_fix(num_tasks)
        
        completion_times = []
        
        for proc_id in range(self.num_processors):
            current_load = processor_states[proc_id].current_load
            task_indices = mapping.get_processor_tasks(proc_id)
            
            new_load = sum(
                tasks[tid].size for tid in task_indices 
                if 0 <= tid < num_tasks
            )
            
            completion_times.append(current_load + new_load)
        
        maxspan = max(completion_times) if completion_times else 1.0
        if maxspan == 0:
            maxspan = 1.0
        
        total_work = sum(tasks[i].size for i in range(num_tasks))
        ideal_maxspan = total_work / self.num_processors
        
        if ideal_maxspan > 0:
            maxspan_score = ideal_maxspan / maxspan
        else:
            maxspan_score = 0.0
        
        maxspan_score = min(1.0, maxspan_score)
        
        total_load = sum(completion_times)
        avg_utilization = total_load / (maxspan * self.num_processors)
        
        avg_load = total_load / self.num_processors if self.num_processors > 0 else 1.0
        heavy_threshold = self.H * avg_load
        light_threshold = self.L * avg_load
        
        acceptable_queues = sum(
            1 for ct in completion_times
            if light_threshold <= ct <= heavy_threshold
        )
        
        acceptable_ratio = acceptable_queues / self.num_processors if self.num_processors > 0 else 0
        
        fitness = (0.4 * maxspan_score + 
                   0.3 * avg_utilization + 
                   0.3 * acceptable_ratio)
        
        if acceptable_ratio > 0.7:
            fitness *= 1.1
        
        if acceptable_ratio < 0.3:
            fitness *= 0.9
        
        return max(0.0, min(1.0, fitness))
    
    def optimize(self, tasks: List[Task],
                processor_states: List[ProcessorState],
                verbose: bool = False) -> TaskMapping:
        """
        Optimización PSO principal
        
        Args:
            tasks: Lista de tareas
            processor_states: Estados de procesadores
            verbose: Si True, muestra progreso
        
        Returns:
            Mejor mapeo encontrado
        """
        num_tasks = len(tasks)
        
        if num_tasks == 0:
            return TaskMapping(self.num_processors)
        
        # Inicializar enjambre
        swarm = self.initialize_swarm(num_tasks)
        
        # Velocidades (inicialmente cero)
        velocities = [np.zeros(num_tasks, dtype=int) for _ in range(self.num_particles)]
        
        # Evaluar fitness inicial
        fitness_values = [
            self.calculate_fitness(mapping, tasks, processor_states)
            for mapping in swarm
        ]
        
        # Mejores personales
        personal_best = [mapping.copy() for mapping in swarm]
        personal_best_fitness = fitness_values.copy()
        
        # Mejor global
        best_idx = int(np.argmax(fitness_values))
        global_best = swarm[best_idx].copy()
        global_best_fitness = fitness_values[best_idx]
        
        if verbose:
            print(f"\n🐝 Evolución del PSO:")
            track_pso_evolution(fitness_values, 0, global_best_fitness)
        
        # Variables para early stopping
        no_improve_count = 0
        
        # Ciclo PSO
        for iteration in range(self.num_iterations):
            # Actualizar cada partícula
            for i in range(self.num_particles):
                # Convertir mapeo a vector
                position = np.zeros(num_tasks, dtype=int)
                for proc_id in range(self.num_processors):
                    for task_id in swarm[i].get_processor_tasks(proc_id):
                        if task_id < num_tasks:
                            position[task_id] = proc_id
                
                personal_vec = np.zeros(num_tasks, dtype=int)
                for proc_id in range(self.num_processors):
                    for task_id in personal_best[i].get_processor_tasks(proc_id):
                        if task_id < num_tasks:
                            personal_vec[task_id] = proc_id
                
                global_vec = np.zeros(num_tasks, dtype=int)
                for proc_id in range(self.num_processors):
                    for task_id in global_best.get_processor_tasks(proc_id):
                        if task_id < num_tasks:
                            global_vec[task_id] = proc_id
                
                # Actualizar velocidad
                r1 = np.random.random()
                r2 = np.random.random()
                
                cognitive = (personal_vec - position) * self.c1 * r1
                social = (global_vec - position) * self.c2 * r2
                
                velocities[i] = (self.w * velocities[i] + 
                                cognitive + social).astype(int)
                
                # Limitar velocidad
                velocities[i] = np.clip(velocities[i], -2, 2)
                
                # Actualizar posición
                new_position = position + velocities[i]
                new_position = np.clip(new_position, 0, self.num_processors - 1)
                
                # Convertir vector a mapeo
                new_mapping = TaskMapping(self.num_processors)
                for task_id, proc_id in enumerate(new_position):
                    new_mapping.assign_task(int(proc_id), task_id)
                
                swarm[i] = new_mapping
            
            # Evaluar fitness
            fitness_values = [
                self.calculate_fitness(mapping, tasks, processor_states)
                for mapping in swarm
            ]
            
            # Actualizar mejores personales
            for i in range(self.num_particles):
                if fitness_values[i] > personal_best_fitness[i]:
                    personal_best[i] = swarm[i].copy()
                    personal_best_fitness[i] = fitness_values[i]
            
            # Actualizar mejor global
            best_idx = int(np.argmax(fitness_values))
            if fitness_values[best_idx] > global_best_fitness:
                improvement = fitness_values[best_idx] - global_best_fitness
                global_best = swarm[best_idx].copy()
                global_best_fitness = fitness_values[best_idx]
                no_improve_count = 0
                
                if improvement < 1e-6:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            if verbose:
                track_pso_evolution(fitness_values, iteration + 1, global_best_fitness)
            
            # Early stopping
            if no_improve_count >= self.early_stop_iters:
                if verbose:
                    print(f"  Early stopping en iteración {iteration + 1}")
                break
        
        global_best.fitness_value = global_best_fitness
        global_best.validate_and_fix(num_tasks)
        
        return global_best

# ============================================================================
# FUNCIÓN PRINCIPAL DE VECTORIZACIÓN
# ============================================================================

def vectorize_with_pso_load_balancing(
    df,
    config: Dict[str, Any] = None,
    verbose: bool = False
) -> Tuple[Any, float, Dict[str, Any]]:
    """
    Vectorización TF-IDF con balanceo de carga basado en PSO
    
    Args:
        df: DataFrame con columna 'text'
        config: Configuración del PSO
        verbose: Si True, muestra información detallada
    
    Returns:
        Tupla (X, tiempo_total, stats)
    """
    if config is None:
        config = PSO_CONFIG.copy()
    
    num_cores = config['num_cores']
    texts = df["text"].tolist()
    total_texts = len(texts)
    
    print(f"  Usando {num_cores} cores para procesamiento paralelo")
    
    # Inicializar vectorizador
    vectorizer = TfidfVectorizer(
        tokenizer=procesar_texto,
        lowercase=False,
        max_features=1000
    )
    
    print("  Ajustando vocabulario...")
    fit_start = time.time()
    vectorizer.fit(texts)
    fit_time = time.time() - fit_start
    print(f"  Vocabulario listo ({fit_time:.2f}s)")
    
    # Dividir en tareas
    chunk_size = calculate_optimal_chunk_size(total_texts, num_cores)
    print(f"  Chunk size óptimo: {chunk_size}")
    
    tasks = []
    for i in range(0, total_texts, chunk_size):
        chunk = texts[i:i + chunk_size]
        task = Task(
            task_id=len(tasks),
            texts=chunk,
            size=estimate_task_complexity(chunk)
        )
        tasks.append(task)
    
    num_tasks_total = len(tasks)
    print(f"  Total de tareas: {num_tasks_total}")
    
    # Calcular ventana
    window_size = calculate_optimal_window_size(num_tasks_total, num_cores)
    print(f"  Tamaño de ventana óptimo: {window_size}")
    
    # Inicializar estados
    processor_states = [
        ProcessorState(processor_id=i, current_load=0.0, queue=[])
        for i in range(num_cores)
    ]
    
    # Inicializar PSO
    pso = PSOLoadBalancer(config)
    
    # Estadísticas
    stats: Dict[str, Any] = {
        'total_texts': total_texts,
        'num_tasks': num_tasks_total,
        'num_cores': num_cores,
        'pso_iterations': config['num_iterations'],
        'pso_particles': config['num_particles'],
        'chunk_size': chunk_size,
        'window_size': window_size,
        'pso_time': 0.0,
        'vectorization_time': 0.0,
        'total_time': 0.0
    }
    
    start_total = time.time()
    
    # Procesamiento por ventanas
    total_windows = (num_tasks_total + window_size - 1) // window_size
    vectorized_chunks = []
    processed_tasks = 0
    window_count = 0
    
    print(f"  Procesando {total_windows} ventanas...")
    
    while processed_tasks < num_tasks_total:
        window_count += 1
        
        window_end = min(processed_tasks + window_size, num_tasks_total)
        window_tasks = tasks[processed_tasks:window_end]
        
        print(f"    Ventana {window_count}/{total_windows}: {len(window_tasks)} tareas", end=" ")
        
        # Ejecutar PSO
        pso_start = time.time()
        best_mapping = pso.optimize(window_tasks, processor_states, verbose=verbose)
        pso_time = time.time() - pso_start
        stats['pso_time'] += pso_time
        
        print(f"(PSO: {pso_time:.2f}s, fitness: {best_mapping.fitness_value:.4f})", end=" ")
        
        # Mostrar distribución si verbose
        if verbose:
            print()
            print_distribution_stats(best_mapping, window_tasks, num_cores,
                                    show_details=True)
        
        # Ejecutar vectorización
        vec_start = time.time()
        
        processor_work = [[] for _ in range(num_cores)]
        for proc_id in range(num_cores):
            task_indices = best_mapping.get_processor_tasks(proc_id)
            for local_tid in task_indices:
                if 0 <= local_tid < len(window_tasks):
                    processor_work[proc_id].append(window_tasks[local_tid])
        
        work_args = [
            ([text for task in proc_tasks for text in task.texts], vectorizer)
            for proc_tasks in processor_work
            if proc_tasks
        ]
        
        if work_args:
            with Pool(processes=num_cores) as pool:
                chunk_results = pool.map(vectorize_chunk, work_args)
                vectorized_chunks.extend([r for r in chunk_results if r is not None])
        
        vec_time = time.time() - vec_start
        stats['vectorization_time'] += vec_time
        
        if not verbose:
            print(f"(Vec: {vec_time:.2f}s)")
        else:
            print(f"\n  Vectorización completada en {vec_time:.2f}s")
        
        # Actualizar estados
        for proc_id in range(num_cores):
            processor_states[proc_id].current_load = 0.0
            processor_states[proc_id].queue.clear()
        
        processed_tasks = window_end
    
    # Combinar resultados
    print("  Combinando resultados...")
    if vectorized_chunks:
        X = vstack(vectorized_chunks)
    else:
        X = vectorizer.transform(texts)
    
    total_time = time.time() - start_total
    stats['total_time'] = total_time
    
    # Mostrar resumen
    print(f"\n  Resumen de tiempos:")
    print(f"  - Total: {total_time:.2f}s")
    print(f"  - PSO: {stats['pso_time']:.2f}s ({stats['pso_time']/total_time*100:.1f}%)")
    print(f"  - Vectorización: {stats['vectorization_time']:.2f}s ({stats['vectorization_time']/total_time*100:.1f}%)")
    
    return X, total_time, stats

# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    """Pruebas del módulo PSO"""
    import pandas as pd
    
    print("Prueba de PSO Load Balancer")
    print(f"Cores disponibles: {AVAILABLE_CORES}")
    
    # Cargar datos de prueba
    df_test = pd.read_csv('Suicide_Detection.csv').head(10000)
    
    # Ejecutar vectorización con PSO
    X, tiempo, stats = vectorize_with_pso_load_balancing(df_test, verbose=True)
    
    print(f"\nResultado: {X.shape[0]} textos vectorizados en {tiempo:.2f}s")
