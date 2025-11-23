"""
GA.py

Balanceo de Carga Dinámico Basado en Algoritmos Genéticos para Vectorización de Texto
================================================================================

Basado en: "Observations on Using Genetic Algorithms for Dynamic Load-Balancing"
por Zomaya & Teh (2001)

CONCEPTO PRINCIPAL DEL ARTÍCULO:
--------------------------------
El artículo propone usar Algoritmos Genéticos (GA) para asignar tareas 
dinámicamente a procesadores en sistemas paralelos. Los objetivos son:

1. MINIMIZAR el tiempo total de ejecución (makespan)
2. MAXIMIZAR la utilización de procesadores 
3. BALANCEAR la carga entre procesadores

COMPONENTES CLAVE (del artículo):
---------------------------------
- Ventana deslizante (Sección 4.1): Procesar tareas conforme llegan
- Política de umbrales (Sección 5.4): Determinar procesadores sobrecargados
- Cromosoma bidimensional (Sección 4.2): Mapeo de tareas a procesadores
- Función de fitness triple (Sección 4.3-4.7): Combinar múltiples objetivos
- Operadores genéticos (Sección 4.8): Selección, cruce y mutación

APLICACIÓN EN ESTE CÓDIGO:
--------------------------
Aplicamos el GA del artículo al problema de vectorización TF-IDF de textos,
donde cada "tarea" es un chunk de textos a vectorizar, y cada "procesador"
es un core de CPU en un sistema multicore.
"""

import numpy as np
import pandas as pd
import time
import re
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
# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

# Palabras vacías para preprocesamiento de texto
STOP_WORDS = {"the", "and", "is", "in", "at", "of", "a", "to", "for", "on",
              "it", "this", "that"}

# Detectar todos los cores disponibles en el sistema
AVAILABLE_CORES = cpu_count()

# Parámetros del Algoritmo Genético
GA_CONFIG = {
    'population_size': 30,
    'num_generations': 10,
    'heavy_multiplier': 1.2,
    'light_multiplier': 0.8,
    'mutation_rate': 0.15,
    'num_cores': AVAILABLE_CORES
}


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class Task:
    """
    Representa una tarea de vectorización (chunk de textos)
    
    Attributes:
        task_id: Identificador único de la tarea
        texts: Lista de textos a procesar
        size: Complejidad estimada
    """
    texts: List[str]
    size: int
    original_indices: List[int]


@dataclass
class ProcessorState:
    """
    Estado actual de un procesador en el sistema
    
    Attributes:
        processor_id: Identificador del procesador
        current_load: Carga actualmente en ejecución
        queue: Cola de tareas asignadas
    """
    processor_id: int
    current_load: float
    queue: List[Task]

    def total_load(self) -> float:
        """Calcula la carga total del procesador"""
        return self.current_load + sum(t.size for t in self.queue)


class TaskMapping:
    """
    Representa un cromosoma - mapeo completo de tareas a procesadores
    
    Del artículo (Sección 4.2): "Codificación bidimensional"
    """

    def __init__(self, num_processors: int):
        """Inicializa un mapeo vacío"""
        self.num_processors = num_processors
        self.assignment: List[List[int]] = [[] for _ in range(num_processors)]
        self.fitness_value: float = 0.0

    def assign_task(self, processor_id: int, task_id: int):
        """Asigna una tarea a un procesador específico"""
        self.assignment[processor_id].append(task_id)

    def get_processor_tasks(self, processor_id: int) -> List[int]:
        """Obtiene todas las tareas asignadas a un procesador"""
        return self.assignment[processor_id]

    def copy(self) -> 'TaskMapping':
        """Crea una copia profunda del mapeo"""
        new_mapping = TaskMapping(self.num_processors)
        new_mapping.assignment = [queue[:] for queue in self.assignment]
        new_mapping.fitness_value = self.fitness_value
        return new_mapping

    def validate_and_fix(self, num_tasks: int):
        """Asegura que todos los IDs de tarea sean válidos"""
        for proc_id in range(self.num_processors):
            self.assignment[proc_id] = [
                tid for tid in self.assignment[proc_id]
                if 0 <= tid < num_tasks
            ]


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def process_text(text: str) -> List[str]:
    """
    Preprocesamiento de texto con expresiones regulares
    
    Args:
        text: Texto crudo a procesar
        
    Returns:
        Lista de tokens limpios
    """
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]
    return tokens


def estimate_task_complexity(texts: List[str]) -> int:
    """
    Estima el tiempo de procesamiento para un chunk de textos
    
    Args:
        texts: Lista de textos en el chunk
        
    Returns:
        Complejidad estimada
    """
    if not texts:
        return 1
    
    base_cost = len(texts) * 100
    length_cost = sum(len(text) for text in texts)
    
    return max(1, base_cost + length_cost)


def vectorize_chunk(args: Tuple[List[str], TfidfVectorizer, List[int]]) -> List[Tuple[int, Any]]:
    """
    Worker para vectorizar un chunk
    
    Returns:
        Lista de tuplas (indice_original, vector_fila)
    """
    texts, vectorizer, original_indices = args
    
    if not texts or not original_indices:
        return []
    
    # Vectorizar el chunk completo
    X_chunk = vectorizer.transform(texts)
    
    # Retornar lista de tuplas (índice_original, vector)
    result = []
    for i, original_idx in enumerate(original_indices):
        result.append((original_idx, X_chunk.getrow(i)))
    
    return result


def calculate_optimal_chunk_size(total_texts: int, num_cores: int) -> int:
    chunk_size = total_texts // 10
    return chunk_size

# ============================================================================
# ALGORITMO GENÉTICO
# ============================================================================

class GeneticLoadBalancer:
    """
    Algoritmo Genético para Balanceo de Carga Dinámico
    
    Del artículo: "Un algoritmo de balanceo de carga dinámico se desarrolla
    mediante el cual asignaciones de tareas óptimas o casi óptimas pueden
    'evolucionar' durante la operación del sistema."
    """

    def __init__(self, config: Dict[str, Any]):
        """Inicializa el algoritmo genético con configuración dada"""
        self.config = config
        self.num_processors = self.config['num_cores']
        self.population_size = self.config['population_size']
        self.num_generations = self.config['num_generations']
        self.H = self.config['heavy_multiplier']
        self.L = self.config['light_multiplier']
        self.mutation_rate = self.config['mutation_rate']

    def initialize_population(self, num_tasks: int) -> List[TaskMapping]:
        """
        Inicializa población con diferentes estrategias de asignación
        
        Args:
            num_tasks: Número total de tareas a asignar
            
        Returns:
            Lista de mappings (cromosomas) iniciales
        """
        population = []

        for i in range(self.population_size):
            mapping = TaskMapping(self.num_processors)
            if i == 0:
                # Estrategia 1: Round-robin puro
                for task_idx in range(num_tasks):
                    processor = task_idx % self.num_processors
                    mapping.assign_task(processor, task_idx)
                    
            elif i == 1:
                # Estrategia 2: Asignación balanceada por bloques
                tasks_per_proc = num_tasks // self.num_processors
                for proc_id in range(self.num_processors):
                    start = proc_id * tasks_per_proc
                    end = start + tasks_per_proc if proc_id < self.num_processors - 1 else num_tasks
                    for task_idx in range(start, end):
                        mapping.assign_task(proc_id, task_idx)
                        
            else:
                # Estrategia 3: Asignación completamente aleatoria
                for task_idx in range(num_tasks):
                    processor = np.random.randint(0, self.num_processors)
                    mapping.assign_task(processor, task_idx)
            population.append(mapping)
        return population

    def calculate_fitness(self, mapping: TaskMapping, tasks: List[Task],
                        processor_states: List[ProcessorState]) -> float:
        """
        Versión CORREGIDA con normalización relativa
        """
        num_tasks = len(tasks)
        mapping.validate_and_fix(num_tasks)

        # ====================================================================
        # PASO 1: Calcular tiempos de finalización
        # ====================================================================
        completion_times = []
        
        for proc_id in range(self.num_processors):
            current_load = processor_states[proc_id].current_load
            task_indices = mapping.get_processor_tasks(proc_id)
            
            new_load = sum(
                tasks[tid].size for tid in task_indices 
                if 0 <= tid < num_tasks
            )
            
            completion_times.append(current_load + new_load)

        # ====================================================================
        # COMPONENTE 1: MAXSPAN (NORMALIZACIÓN RELATIVA)
        # ====================================================================
        maxspan = max(completion_times) if completion_times else 1.0
        
        # Calcular el MEJOR maxspan posible (distribución perfecta)
        total_work = sum(tasks[i].size for i in range(num_tasks))
        ideal_maxspan = total_work / self.num_processors
        
        # Normalizar: si maxspan = ideal → score = 1.0
        #             si maxspan = 2*ideal → score = 0.5
        if ideal_maxspan > 0:
            maxspan_score = ideal_maxspan / maxspan
        else:
            maxspan_score = 0.0
        
        # Limitar a [0, 1]
        maxspan_score = min(1.0, maxspan_score)

        # ====================================================================
        # COMPONENTE 2: UTILIZACIÓN PROMEDIO
        # ====================================================================
        total_load = sum(completion_times)
        avg_utilization = total_load / (maxspan * self.num_processors)

        # ====================================================================
        # COMPONENTE 3: POLÍTICA DE UMBRALES
        # ====================================================================
        avg_load = total_load / self.num_processors if self.num_processors > 0 else 1.0
        heavy_threshold = self.H * avg_load
        light_threshold = self.L * avg_load
        
        acceptable_queues = sum(
            1 for ct in completion_times
            if light_threshold <= ct <= heavy_threshold
        )
        
        acceptable_ratio = acceptable_queues / self.num_processors if self.num_processors > 0 else 0

        # ====================================================================
        # FUNCIÓN DE FITNESS CORREGIDA
        # ====================================================================
        fitness = (0.4 * maxspan_score + 
                0.3 * avg_utilization + 
                0.3 * acceptable_ratio)
        
        # Bonificación si la mayoría está bien balanceada
        if acceptable_ratio > 0.8:
            fitness *= 1.2
        
        # Penalización si el desbalance es extremo
        if acceptable_ratio < 0.4:
            fitness *= 0.8

        return max(0.0, min(1.0, fitness))  # Forzar rango [0, 1]

    def roulette_wheel_selection(self, population: List[TaskMapping],
                                 fitness_values: List[float]) -> TaskMapping:
        """
        Selección por ruleta (Roulette Wheel Selection)
        
        Args:
            population: Lista de cromosomas
            fitness_values: Fitness de cada cromosoma
            
        Returns:
            Cromosoma seleccionado
        """
        total_fitness = sum(fitness_values)

        if total_fitness == 0:
            idx = np.random.randint(0, len(population))
            return population[idx]

        probabilities = [f / total_fitness for f in fitness_values]
        r = np.random.random()
        cumsum = 0

        for i, prob in enumerate(probabilities):
            cumsum += prob
            if r <= cumsum:
                return population[i]

        return population[-1]

    def cycle_crossover(self, parent1: TaskMapping,
                       parent2: TaskMapping, num_tasks: int) -> Tuple[TaskMapping, TaskMapping]:
        """
        Cruce por ciclo (Cycle Crossover)
        
        Args:
            parent1: Primer padre
            parent2: Segundo padre
            num_tasks: Número total de tareas
            
        Returns:
            Tupla (hijo1, hijo2)
        """
        child1 = TaskMapping(self.num_processors)
        child2 = TaskMapping(self.num_processors)

        # Aplanar asignaciones a 1D
        p1_flat = []
        p2_flat = []

        for proc_tasks in parent1.assignment:
            p1_flat.extend(proc_tasks)
        for proc_tasks in parent2.assignment:
            p2_flat.extend(proc_tasks)

        if len(p1_flat) == 0 or len(p1_flat) != len(p2_flat):
            return parent1.copy(), parent2.copy()

        n = len(p1_flat)
        mask = np.random.random(n) < 0.5

        c1_flat = [p1_flat[i] if mask[i] else p2_flat[i] for i in range(n)]
        c2_flat = [p2_flat[i] if mask[i] else p1_flat[i] for i in range(n)]

        for i, task_id in enumerate(c1_flat):
            if 0 <= task_id < num_tasks:
                proc = i % self.num_processors
                child1.assign_task(proc, task_id)

        for i, task_id in enumerate(c2_flat):
            if 0 <= task_id < num_tasks:
                proc = i % self.num_processors
                child2.assign_task(proc, task_id)

        return child1, child2

    def swap_mutation(self, mapping: TaskMapping, num_tasks: int) -> TaskMapping:
        """
        Mutación por intercambio (Swap Mutation)
        
        Args:
            mapping: Cromosoma a mutar
            num_tasks: Número total de tareas
            
        Returns:
            Cromosoma mutado
        """
        mutated = mapping.copy()

        if np.random.random() > self.mutation_rate:
            return mutated

        mutated.validate_and_fix(num_tasks)

        processors_with_tasks = [
            i for i in range(self.num_processors)
            if len(mutated.get_processor_tasks(i)) > 0
        ]

        if len(processors_with_tasks) < 2:
            return mutated

        proc1 = np.random.choice(processors_with_tasks)
        processors_with_tasks_2 = [p for p in processors_with_tasks if p != proc1]
        
        if not processors_with_tasks_2:
            return mutated
            
        proc2 = np.random.choice(processors_with_tasks_2)

        tasks1 = mutated.get_processor_tasks(proc1)
        tasks2 = mutated.get_processor_tasks(proc2)

        if len(tasks1) > 0 and len(tasks2) > 0:
            idx1 = np.random.randint(0, len(tasks1))
            idx2 = np.random.randint(0, len(tasks2))

            task1 = tasks1[idx1]
            task2 = tasks2[idx2]

            mutated.assignment[proc1][idx1] = task2
            mutated.assignment[proc2][idx2] = task1

        return mutated

    def evolve(self, tasks: List[Task],
            processor_states: List[ProcessorState],
            verbose: bool = False) -> TaskMapping:
        """
        Ciclo principal de evolución del Algoritmo Genético
        
        Args:
            tasks: Lista de todas las tareas a asignar
            processor_states: Estado actual de cada procesador
            verbose: Si True, muestra progreso generación por generación
            
        Returns:
            Mejor mapeo encontrado
        """
        num_tasks = len(tasks)

        if num_tasks == 0:
            return TaskMapping(self.num_processors)

        # Inicializar población
        population = self.initialize_population(num_tasks)

        # Evaluar fitness inicial
        fitness_values = [
            self.calculate_fitness(mapping, tasks, processor_states)
            for mapping in population
        ]

        if verbose:
            print(f"\n🧬 Evolución del Algoritmo Genético:")
            track_ga_evolution(population, fitness_values, 0)

        # Ciclo de evolución
        for gen in range(self.num_generations):
            new_population = []

            # Elitismo: mantener los 2 mejores
            sorted_indices = np.argsort(fitness_values)[::-1]
            new_population.append(population[sorted_indices[0]].copy())
            if len(population) > 1:
                new_population.append(population[sorted_indices[1]].copy())

            # Generar nueva población
            while len(new_population) < self.population_size:
                # Selección
                parent1 = self.roulette_wheel_selection(population, fitness_values)
                parent2 = self.roulette_wheel_selection(population, fitness_values)

                # Cruce
                child1, child2 = self.cycle_crossover(parent1, parent2, num_tasks)

                # Mutación
                child1 = self.swap_mutation(child1, num_tasks)
                child2 = self.swap_mutation(child2, num_tasks)

                new_population.extend([child1, child2])

            # Recortar al tamaño exacto
            population = new_population[:self.population_size]

            # Re-evaluar fitness
            fitness_values = [
                self.calculate_fitness(mapping, tasks, processor_states)
                for mapping in population
            ]
            
            if verbose:
                track_ga_evolution(population, fitness_values, gen + 1)

        # Retornar mejor solución
        best_idx = np.argmax(fitness_values)
        best_mapping = population[best_idx]
        best_mapping.fitness_value = fitness_values[best_idx]

        best_mapping.validate_and_fix(num_tasks)

        return best_mapping

# ============================================================================
# FUNCIONES DE VISUALIZACIÓN Y DEBUG
# ============================================================================

def print_distribution_stats(mapping: TaskMapping, tasks: List[Task], 
                            num_cores: int, show_details: bool = True):
    """
    Muestra estadísticas detalladas de la distribución de tareas
    
    Args:
        mapping: Mapeo de tareas a procesadores
        tasks: Lista de todas las tareas
        num_cores: Número de cores
        show_details: Si True, muestra detalle por core
    """
    print("\n" + "="*70)
    print("📊 ANÁLISIS DE DISTRIBUCIÓN DE CARGA")
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
            bar_length = int(load_pct / 100 * 20)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            
            # Colorear según desviación
            if load < avg_load * 0.9:
                color = "🔵"  # Subutilizado
            elif load > avg_load * 1.1:
                color = "🔴"  # Sobrecargado
            else:
                color = "🟢"  # Balanceado
            
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


def track_ga_evolution(population: List[TaskMapping], 
                      fitness_values: List[float],
                      generation: int):
    """
    Muestra el progreso del GA a través de las generaciones
    
    Args:
        population: Población actual
        fitness_values: Fitness de cada cromosoma
        generation: Número de generación actual
    """
    best_fitness = max(fitness_values)
    avg_fitness = sum(fitness_values) / len(fitness_values)
    worst_fitness = min(fitness_values)
    
    # Crear barra de progreso
    progress = best_fitness
    bar_length = int(progress * 20)
    bar = '█' * bar_length + '░' * (20 - bar_length)
    
    print(f"  Gen {generation:2d}: "
          f"Best={best_fitness:.4f} "
          f"Avg={avg_fitness:.4f} "
          f"Worst={worst_fitness:.4f} "
          f"[{bar}]")


# ============================================================================
# FUNCIÓN PRINCIPAL DE VECTORIZACIÓN
# ============================================================================

def vectorize_with_ga_load_balancing(
    df,
    config: Dict[str, Any] = GA_CONFIG,
    verbose: bool = False,
    train_model: bool = False
) -> Tuple[Any, float, Dict[str, Any]]:
    """
    Vectorización TF-IDF con balanceo de carga basado en GA
    VERSIÓN CON EMPAREJAMIENTO EXPLÍCITO VECTOR-ETIQUETA
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
    
    # Dividir dataset en tareas CON ÍNDICES ORIGINALES
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
    
    # Inicializar estados de procesador
    processor_states = [
        ProcessorState(processor_id=i, current_load=0.0, queue=[])
        for i in range(num_cores)
    ]
    
    # Inicializar GA
    ga = GeneticLoadBalancer(config)
    
    # Estadísticas
    stats: Dict[str, Any] = {
        'total_texts': total_texts,
        'num_tasks': num_tasks_total,
        'num_cores': num_cores,
        'ga_generations': config['num_generations'],
        'ga_population': config['population_size'],
        'chunk_size': chunk_size,
        'ga_time': 0.0,
        'vectorization_time': 0.0,
        'total_time': 0.0
    }
    
    start_total = time.time()
    
    # ========================================================================
    # ⭐ CLAVE: Lista de tuplas (índice_original, vector)
    # ========================================================================
    indexed_vectors = []  # Lista de (idx, vector)
    
    processed_tasks = 0
    chunk_count = 0
    
    while processed_tasks < num_tasks_total:
        window_beginning = chunk_count * chunk_size
        chunk_count += 1
        window_end = min((chunk_count * chunk_size) - 1, num_tasks_total)
        window_tasks = tasks[window_beginning:window_end]
        
        # Ejecutar GA        
        ga_start = time.time()
        best_mapping = ga.evolve(window_tasks, processor_states, verbose=verbose)
        ga_time = time.time() - ga_start
        stats['ga_time'] += ga_time
        
        print(f"(GA: {ga_time:.2f}s, fitness: {best_mapping.fitness_value:.4f})", end=" ")
        
        if verbose:
            print()
            print_distribution_stats(best_mapping, window_tasks, num_cores, show_details=True)
        
        # Ejecutar vectorización
        vec_start = time.time()
        
        # Preparar trabajo para procesadores
        processor_work = [[] for _ in range(num_cores)]
        processor_indices = [[] for _ in range(num_cores)]
        
        for proc_id in range(num_cores):
            task_indices = best_mapping.get_processor_tasks(proc_id)
            for local_tid in task_indices:
                if 0 <= local_tid < len(window_tasks):
                    task = window_tasks[local_tid]
                    processor_work[proc_id].append(task)
                    processor_indices[proc_id].extend(task.original_indices)
        
        work_args = [
            (
                [text for task in proc_tasks for text in task.texts],
                vectorizer,
                proc_indices
            )
            for proc_tasks, proc_indices in zip(processor_work, processor_indices)
            if proc_tasks
        ]
        
        if work_args:
            with Pool(processes=num_cores) as pool:
                chunk_results = pool.map(vectorize_chunk, work_args)
                
                # ⭐ Agregar todas las tuplas (idx, vector) a la lista
                for result_list in chunk_results:
                    if result_list:  # result_list es una lista de tuplas
                        indexed_vectors.extend(result_list)
        
        vec_time = time.time() - vec_start
        stats['vectorization_time'] += vec_time
        
        if not verbose:
            print(f"(Vec: {vec_time:.2f}s)")
        else:
            print(f"\n  Vectorización completada en {vec_time:.2f}s")
        
        # Limpiar estados
        for proc_id in range(num_cores):
            processor_states[proc_id].current_load = 0.0
            processor_states[proc_id].queue.clear()
        
        processed_tasks = window_end
    
    # ========================================================================
    # ⭐ RECONSTRUIR X EN CUALQUIER ORDEN (no importa)
    # ========================================================================
    print(f"  Reconstruyendo matriz...")
    print(f"  - Vectores obtenidos: {len(indexed_vectors)}")
    print(f"  - Vectores esperados: {total_texts}")
    
    # Verificar que tengamos todos los vectores
    obtained_indices = set(idx for idx, _ in indexed_vectors)
    expected_indices = set(range(total_texts))
    
    missing = expected_indices - obtained_indices
    if missing:
        print(f"  ⚠️  Faltan {len(missing)} vectores")
        print(f"      Primeros faltantes: {sorted(list(missing))[:10]}")
    
    duplicates = len(indexed_vectors) - len(obtained_indices)
    if duplicates > 0:
        print(f"  ⚠️  Hay {duplicates} vectores duplicados")
    
    # Construir matriz X (en cualquier orden)
    vectors_list = [vec for _, vec in indexed_vectors]
    X = vstack(vectors_list)
    
    print(f"  ✅ Matriz construida: {X.shape}")
    
    total_time = time.time() - start_total
    stats['total_time'] = total_time
    
    print(f"\n  Resumen de tiempos:")
    print(f"  - Total: {total_time:.2f}s")
    print(f"  - GA: {stats['ga_time']:.2f}s ({stats['ga_time']/total_time*100:.1f}%)")
    print(f"  - Vectorización: {stats['vectorization_time']:.2f}s ({stats['vectorization_time']/total_time*100:.1f}%)")
    
    # ========================================================================
    # ⭐ EMPAREJAMIENTO EXPLÍCITO: Crear y alineado con etiquetas
    # ========================================================================
    if train_model and 'class' in df.columns:
        print(f"\n{'='*70}")
        print(f"🔗 EMPAREJAMIENTO VECTOR-ETIQUETA")
        print(f"{'='*70}")
        
        # Extraer etiquetas originales
        y_original = df['class'].values
        
        # ⭐ Crear nuevo array de etiquetas alineado con los vectores
        y_aligned = np.zeros(len(indexed_vectors), dtype=y_original.dtype)
        
        for i, (original_idx, _) in enumerate(indexed_vectors):
            y_aligned[i] = y_original[original_idx]
        
        print(f"  ✅ Etiquetas emparejadas: {len(y_aligned)}")
        print(f"  - Forma de X: {X.shape}")
        print(f"  - Forma de y: {y_aligned.shape}")
        if X.shape is not None and y_aligned.shape is not None:
            print(f"  - ¿Coinciden?: {'✅ SÍ' if X.shape[0] == y_aligned.shape[0] else '❌ NO'}")
        
        # Verificar distribución de clases
        unique, counts = np.unique(y_aligned, return_counts=True)
        print(f"\n  📊 Distribución de clases:")
        for label, count in zip(unique, counts):
            print(f"     Clase {label}: {count} ({count/len(y_aligned)*100:.1f}%)")
        
        # Verificar algunas muestras
        print(f"\n  🔍 Verificando primeras 5 muestras:")
        for i in range(min(5, len(indexed_vectors))):
            original_idx, _ = indexed_vectors[i]
            text_preview = texts[original_idx][:50] + "..." if len(texts[original_idx]) > 50 else texts[original_idx]
            print(f"     Vector[{i}] -> Original[{original_idx}] -> Clase={y_aligned[i]}")
            print(f"         Texto: {text_preview}")
        
        print(f"{'='*70}\n")
        
        # Entrenar modelo con vectores y etiquetas ALINEADOS
        mlp_stats = train_and_evaluate_mlp(X, y_aligned, method_name="GA-Paralelo")
        stats['mlp_stats'] = mlp_stats
    
    return X, total_time, stats

# ============================================================================
# ENTRENAMIENTO Y EVALUACIÓN DE MODELO
# ============================================================================

def train_and_evaluate_mlp(X, y, method_name: str = "Método") -> Dict[str, Any]:
    """
    Entrena un MLPClassifier y muestra matriz de confusión
    
    Args:
        X: Matriz de características (vectores TF-IDF)
        y: Etiquetas
        method_name: Nombre del método para el título
    
    Returns:
        Diccionario con métricas del modelo
    """
    print(f"\n{'='*70}")
    print(f"🧠 ENTRENAMIENTO DE RED NEURONAL MLP ({method_name})")
    print(f"{'='*70}")
    
    # Dividir datos
    print("  Dividiendo datos (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Train: {X_train.shape[0]} muestras")
    print(f"  Test:  {X_test.shape[0]} muestras")
    
    # Crear y entrenar modelo
    print("\n  Entrenando MLP...")
    mlp_start = time.time()
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # 2 capas ocultas
        max_iter=50,                   # Pocas iteraciones para ser rápido
        random_state=42,
        verbose=False
    )
    
    mlp.fit(X_train, y_train)
    mlp_time = time.time() - mlp_start
    
    print(f"  ✓ Entrenamiento completado en {mlp_time:.2f}s")
    
    # Predecir
    print("\n  Realizando predicciones...")
    y_pred = mlp.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 RESULTADOS:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n  Matriz de Confusión:")
    print(f"  {cm}")
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Suicida', 'Suicida'],
                yticklabels=['No Suicida', 'Suicida'])
    plt.title(f'Matriz de Confusión - {method_name}')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.tight_layout()
    
    # Guardar figura
    filename = f'confusion_matrix_{method_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✓ Matriz de confusión guardada: {filename}")
    
    # Reporte detallado
    print(f"\n  Reporte de Clasificación:")
    report = classification_report(y_test, y_pred, 
                                   target_names=['No Suicida', 'Suicida'])
    print(report)
    
    print(f"{'='*70}\n")
    
    # Retornar métricas
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
    """Pruebas del módulo GA"""
    print("Prueba de GA Load Balancer")
    print(f"Cores disponibles: {AVAILABLE_CORES}")
    
    # Cargar datos de prueba
    df_test = pd.read_csv('Suicide_Detection.csv').head(20000)
    
    # Ejecutar vectorización con GA
    X, tiempo, stats = vectorize_with_ga_load_balancing(df_test)
    
    print(f"\nResultado: {X.shape[0]} textos vectorizados en {tiempo:.2f}s")