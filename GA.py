"""
GA.py

Balanceo de Carga Dinámico Basado en Algoritmos Genéticos para Vectorización de Texto
================================================================================

Basado en: "Observations on Using Genetic Algorithms for Dynamic Load-Balancing"
por Zomaya & Teh (2001)

VERSIÓN MEJORADA: Enfoque en UTILIZACIÓN de cores en lugar de balance de carga
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

STOP_WORDS = {"the", "and", "is", "in", "at", "of", "a", "to", "for", "on",
              "it", "this", "that"}

AVAILABLE_CORES = cpu_count()

GA_CONFIG = {
    'population_size': 20,
    'num_generations': 5,
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
    """Representa una tarea de vectorización (chunk de textos)"""
    texts: List[str]
    size: int
    original_indices: List[int]


@dataclass
class Subtask:
    """Representa una subtarea - subdivisión de una tarea"""
    subtask_id: int
    texts: List[str]
    size: int
    original_indices: List[int]
    parent_task_id: int


@dataclass
class ProcessorState:
    """Estado actual de un procesador en el sistema"""
    processor_id: int
    current_load: float
    queue: List[Task]

    def total_load(self) -> float:
        """Calcula la carga total del procesador"""
        return self.current_load + sum(t.size for t in self.queue)


class TaskMapping:
    """Representa un cromosoma - mapeo completo de tareas a procesadores"""

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
    """Preprocesamiento de texto con expresiones regulares"""
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]
    return tokens


def estimate_task_complexity(texts: List[str]) -> int:
    """Estima el tiempo de procesamiento para un chunk de textos"""
    if not texts:
        return 1
    
    base_cost = len(texts) * 100
    length_cost = sum(len(text) for text in texts)
    
    return max(1, base_cost + length_cost)


def vectorize_chunk(args: Tuple[List[str], TfidfVectorizer, List[int]]) -> List[Tuple[int, Any]]:
    """Worker para vectorizar un chunk"""
    texts, vectorizer, original_indices = args
    
    if not texts or not original_indices:
        return []
    
    if len(texts) != len(original_indices):
        raise ValueError(f"Mismatch: {len(texts)} texts but {len(original_indices)} indices")
    
    X_chunk = vectorizer.transform(texts)
    
    result = []
    for i, original_idx in enumerate(original_indices):
        result.append((original_idx, X_chunk.getrow(i)))
    
    return result


def calculate_optimal_chunk_size(total_texts: int, num_cores: int) -> int:
    chunk_size = total_texts // 10
    return chunk_size


def create_subtasks_from_task(task: Task, num_subtasks: int, task_id: int) -> List[Subtask]:
    """Subdivide una tarea en múltiples subtareas de tamaño aleatorio"""
    total_texts = len(task.texts)
    
    if total_texts < num_subtasks:
        return [Subtask(
            subtask_id=0,
            texts=task.texts,
            size=task.size,
            original_indices=task.original_indices,
            parent_task_id=task_id
        )]
    
    weights = np.random.random(num_subtasks)
    weights = weights / weights.sum()
    
    subtask_sizes = (weights * total_texts).astype(int)
    subtask_sizes[-1] = total_texts - subtask_sizes[:-1].sum()
    
    subtasks = []
    current_idx = 0
    
    for i, size in enumerate(subtask_sizes):
        if size <= 0:
            continue
            
        end_idx = current_idx + size
        subtask_texts = task.texts[current_idx:end_idx]
        subtask_indices = task.original_indices[current_idx:end_idx]
        
        subtask = Subtask(
            subtask_id=i,
            texts=subtask_texts,
            size=estimate_task_complexity(subtask_texts),
            original_indices=subtask_indices,
            parent_task_id=task_id
        )
        subtasks.append(subtask)
        current_idx = end_idx
    
    return subtasks


# ============================================================================
# ALGORITMO GENÉTICO - ENFOCADO EN UTILIZACIÓN
# ============================================================================

class GeneticLoadBalancer:
    """
    Algoritmo Genético para Balanceo de Carga Dinámico
    VERSIÓN MEJORADA: Maximiza utilización uniforme de cores
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

    def initialize_population(self, num_tasks: int, 
                             processor_states: List[ProcessorState]) -> List[TaskMapping]:
        """
        Inicializa población con estrategias conscientes de utilización
        """
        population = []
        
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

        for i in range(self.population_size):
            mapping = TaskMapping(self.num_processors)
            
            if i == 0:
                # ESTRATEGIA 1: GREEDY - Asignar siempre al menos cargado
                for task_idx in range(num_tasks):
                    loads_in_mapping = [
                        current_loads[p] + len(mapping.get_processor_tasks(p))
                        for p in range(self.num_processors)
                    ]
                    least_loaded = np.argmin(loads_in_mapping)
                    mapping.assign_task(least_loaded, task_idx)
                    
            elif i == 1:
                # ESTRATEGIA 2: WEIGHTED RANDOM - Probabilidad inversa a carga
                for task_idx in range(num_tasks):
                    processor = np.random.choice(
                        self.num_processors, 
                        p=probabilities
                    )
                    mapping.assign_task(processor, task_idx)
                    
            elif i == 2:
                # ESTRATEGIA 3: ROUND-ROBIN DESDE MENOS CARGADO
                sorted_procs = np.argsort(current_loads)
                for task_idx in range(num_tasks):
                    processor = sorted_procs[task_idx % self.num_processors]
                    mapping.assign_task(processor, task_idx)
                    
            elif i == 3:
                # ESTRATEGIA 4: BLOQUES BALANCEADOS
                sorted_procs = np.argsort(current_loads)
                tasks_per_proc = num_tasks // self.num_processors
                remainder = num_tasks % self.num_processors
                
                task_idx = 0
                for idx, proc_id in enumerate(sorted_procs):
                    extra = 1 if idx < remainder else 0
                    num_to_assign = tasks_per_proc + extra
                    
                    for _ in range(num_to_assign):
                        if task_idx < num_tasks:
                            mapping.assign_task(proc_id, task_idx)
                            task_idx += 1
            else:
                # ESTRATEGIA 5+: MIXTO
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
        Calcula fitness enfocado en MAXIMIZAR UTILIZACIÓN UNIFORME
        """
        num_tasks = len(tasks)
        mapping.validate_and_fix(num_tasks)

        # Calcular cargas finales
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

        # COMPONENTE 1: UTILIZACIÓN MÍNIMA (queremos que todos trabajen)
        if max_load > 0:
            min_utilization = min_load / max_load
        else:
            min_utilization = 0.0

        # COMPONENTE 2: EFICIENCIA GLOBAL (uso del sistema)
        ideal_load_per_core = total_load / self.num_processors
        if ideal_load_per_core > 0:
            efficiency = avg_load / max_load  # Qué tan cerca está el promedio del máximo
        else:
            efficiency = 0.0

        # COMPONENTE 3: UNIFORMIDAD (coeficiente de variación inverso)
        if avg_load > 0:
            std_dev = (sum((load - avg_load) ** 2 for load in final_loads) / self.num_processors) ** 0.5
            coef_variation = std_dev / avg_load
            uniformity = 1.0 / (1.0 + coef_variation)  # Cercano a 1 si uniformes
        else:
            uniformity = 0.0

        # COMPONENTE 4: THROUGHPUT POTENCIAL (inverso del makespan)
        if max_load > 0:
            throughput_score = avg_load / max_load
        else:
            throughput_score = 0.0

        # FITNESS FINAL: Enfocado en utilización
        fitness = (
            0.35 * min_utilization +    # Que ningún core esté ocioso
            0.25 * efficiency +          # Uso eficiente del sistema
            0.30 * uniformity +          # Distribución uniforme
            0.10 * throughput_score      # Maximizar throughput
        )
        
        # Bonificación si TODOS los cores están bien utilizados
        cores_well_utilized = sum(1 for load in final_loads if load >= 0.8 * avg_load)
        if cores_well_utilized >= 0.85 * self.num_processors:
            fitness *= 1.2
        
        # Penalización severa si algún core está muy subutilizado
        if min_utilization < 0.5:
            fitness *= 0.7

        return max(0.0, min(1.0, fitness))

    def roulette_wheel_selection(self, population: List[TaskMapping],
                                 fitness_values: List[float]) -> TaskMapping:
        """Selección por ruleta"""
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
        """Cruce por ciclo"""
        child1 = TaskMapping(self.num_processors)
        child2 = TaskMapping(self.num_processors)

        for task_id in range(num_tasks):
            parent1_proc = None
            parent2_proc = None
            
            for proc_id in range(self.num_processors):
                if task_id in parent1.get_processor_tasks(proc_id):
                    parent1_proc = proc_id
                if task_id in parent2.get_processor_tasks(proc_id):
                    parent2_proc = proc_id
            
            if parent1_proc is not None and parent2_proc is not None:
                if np.random.random() < 0.5:
                    child1.assign_task(parent1_proc, task_id)
                    child2.assign_task(parent2_proc, task_id)
                else:
                    child1.assign_task(parent2_proc, task_id)
                    child2.assign_task(parent1_proc, task_id)
            elif parent1_proc is not None:
                child1.assign_task(parent1_proc, task_id)
                child2.assign_task(parent1_proc, task_id)
            elif parent2_proc is not None:
                child1.assign_task(parent2_proc, task_id)
                child2.assign_task(parent2_proc, task_id)
            else:
                random_proc = np.random.randint(0, self.num_processors)
                child1.assign_task(random_proc, task_id)
                child2.assign_task(random_proc, task_id)

        return child1, child2

    def swap_mutation(self, mapping: TaskMapping, num_tasks: int) -> TaskMapping:
        """Mutación por intercambio"""
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

    def evolve(self, tasks,
               processor_states: List[ProcessorState],
               verbose: bool = False) -> TaskMapping:
        """Ciclo principal de evolución del Algoritmo Genético"""
        num_tasks = len(tasks)

        if num_tasks == 0:
            return TaskMapping(self.num_processors)

        population = self.initialize_population(num_tasks, processor_states)

        fitness_values = [
            self.calculate_fitness(mapping, tasks, processor_states)
            for mapping in population
        ]

        if verbose:
            print(f"\n🧬 Evolución del Algoritmo Genético:")
            track_ga_evolution(population, fitness_values, 0, tasks, processor_states)

        for gen in range(self.num_generations):
            new_population = []

            sorted_indices = np.argsort(fitness_values)[::-1]
            new_population.append(population[sorted_indices[0]].copy())
            if len(population) > 1:
                new_population.append(population[sorted_indices[1]].copy())

            while len(new_population) < self.population_size:
                parent1 = self.roulette_wheel_selection(population, fitness_values)
                parent2 = self.roulette_wheel_selection(population, fitness_values)

                child1, child2 = self.cycle_crossover(parent1, parent2, num_tasks)

                child1 = self.swap_mutation(child1, num_tasks)
                child2 = self.swap_mutation(child2, num_tasks)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

            fitness_values = [
                self.calculate_fitness(mapping, tasks, processor_states)
                for mapping in population
            ]
            
            if verbose:
                track_ga_evolution(population, fitness_values, gen + 1, tasks, processor_states)

        best_idx = np.argmax(fitness_values)
        best_mapping = population[best_idx]
        best_mapping.fitness_value = fitness_values[best_idx]

        best_mapping.validate_and_fix(num_tasks)

        return best_mapping


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
    print("⚡ ANÁLISIS DE UTILIZACIÓN DE CORES")
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


def track_ga_evolution(population: List[TaskMapping], 
                      fitness_values: List[float],
                      generation: int,
                      tasks,
                      processor_states: List[ProcessorState]):
    """
    Muestra el progreso del GA con métricas de utilización
    """
    best_fitness = max(fitness_values)
    avg_fitness = sum(fitness_values) / len(fitness_values)
    worst_fitness = min(fitness_values)
    
    # Calcular utilización del mejor cromosoma
    best_idx = np.argmax(fitness_values)
    best_mapping = population[best_idx]
    
    final_loads = []
    for proc_id in range(best_mapping.num_processors):
        current_load = processor_states[proc_id].total_load()
        task_ids = best_mapping.get_processor_tasks(proc_id)
        new_load = sum(tasks[tid].size for tid in task_ids if tid < len(tasks))
        final_loads.append(current_load + new_load)
    
    max_load = max(final_loads) if final_loads else 1.0
    avg_load = sum(final_loads) / len(final_loads) if final_loads else 0.0
    min_util = (min(final_loads) / max_load * 100) if max_load > 0 else 0.0
    avg_util = (avg_load / max_load * 100) if max_load > 0 else 0.0
    
    # Barra de progreso
    bar_length = int(best_fitness * 20)
    bar = '█' * bar_length + '░' * (20 - bar_length)
    
    print(f"  Gen {generation:2d}: "
          f"Fitness={best_fitness:.4f} "
          f"AvgUtil={avg_util:.1f}% "
          f"MinUtil={min_util:.1f}% "
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
    VERSIÓN ENFOCADA EN UTILIZACIÓN DE CORES
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
    
    # Dividir dataset en tareas
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
    
    # Subdividir en subtareas
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
        'num_subtasks': num_subtasks_total,
        'num_cores': num_cores,
        'ga_generations': config['num_generations'],
        'ga_population': config['population_size'],
        'chunk_size': chunk_size,
        'subtasks_per_task': num_subtasks_per_task,
        'ga_time': 0.0,
        'vectorization_time': 0.0,
        'total_time': 0.0
    }
    
    start_total = time.time()
    
    indexed_vectors = []
    processed_subtasks = 0
    window_count = 0
    
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
                print(f"  📊 Utilización actual: "
                      f"min={min(utilizations):.1f}%, "
                      f"max={max(utilizations):.1f}%, "
                      f"avg={sum(utilizations)/len(utilizations):.1f}%")
        
        # Ejecutar GA
        ga_start = time.time()
        best_mapping = ga.evolve(window_subtasks, processor_states, verbose=verbose)
        ga_time = time.time() - ga_start
        stats['ga_time'] += ga_time
        
        print(f"  (GA: {ga_time:.2f}s, fitness: {best_mapping.fitness_value:.4f})", 
              end=" ")
        
        if verbose:
            print()
            print_utilization_stats(best_mapping, window_subtasks, 
                                   processor_states, num_cores, show_details=True)
        
        # Ejecutar vectorización
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
            print(f"\n  Vectorización completada en {vec_time:.2f}s")
        
        # Actualizar cargas (NO resetear)
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
                print(f"  📈 Utilización actualizada: "
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
    print(f"  - GA: {stats['ga_time']:.2f}s "
          f"({stats['ga_time']/total_time*100:.1f}%)")
    print(f"  - Vectorización: {stats['vectorization_time']:.2f}s "
          f"({stats['vectorization_time']/total_time*100:.1f}%)")
    
    # Emparejamiento con etiquetas
    if train_model and 'class' in df.columns:
        print(f"\n{'='*70}")
        print(f"🔗 EMPAREJAMIENTO VECTOR-ETIQUETA")
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
        print(f"\n  📊 Distribución de clases:")
        for label, count in zip(unique, counts):
            print(f"     Clase {label}: {count} "
                  f"({count/len(y_aligned)*100:.1f}%)")
        
        print(f"\n  🔍 Verificando primeras 5 muestras:")
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
                                          method_name="GA-Paralelo")
        stats['mlp_stats'] = mlp_stats
    
    return X, total_time, stats


# ============================================================================
# ENTRENAMIENTO Y EVALUACIÓN DE MODELO
# ============================================================================

def train_and_evaluate_mlp(X, y, method_name: str = "Método") -> Dict[str, Any]:
    """Entrena un MLPClassifier y muestra matriz de confusión"""
    print(f"\n{'='*70}")
    print(f"🧠 ENTRENAMIENTO DE RED NEURONAL MLP ({method_name})")
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
    
    print(f"  ✓ Entrenamiento completado en {mlp_time:.2f}s")
    
    print("\n  Realizando predicciones...")
    y_pred = mlp.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 RESULTADOS:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n  Matriz de Confusión:")
    print(f"  {cm}")
    
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
    
    print(f"\n  ✓ Matriz de confusión guardada: {filename}")
    
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
    """Pruebas del módulo GA con enfoque en utilización"""
    print("="*70)
    print("🧬 GA LOAD BALANCER - ENFOQUE EN UTILIZACIÓN DE CORES")
    print("="*70)
    print(f"Cores disponibles: {AVAILABLE_CORES}")
    
    print("\n📂 Cargando datos...")
    df_test = pd.read_csv('Suicide_Detection.csv').head(20000)
    print(f"   Dataset: {len(df_test)} textos")
    
    if 'class' in df_test.columns:
        class_dist = df_test['class'].value_counts()
        print(f"\n📊 Distribución de clases:")
        for label, count in class_dist.items():
            print(f"   Clase {label}: {count} ({count/len(df_test)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("🚀 INICIANDO VECTORIZACIÓN CON GA")
    print("="*70)
    
    X, tiempo, stats = vectorize_with_ga_load_balancing(
        df_test,
        config=GA_CONFIG,
        verbose=True,
        train_model=True
    )
    
    print("\n" + "="*70)
    print("✅ RESULTADO FINAL")
    print("="*70)
    print(f"  Textos procesados:      {X.shape[0]:,}")
    print(f"  Dimensiones del vector: {X.shape[1]:,}")
    print(f"  Tiempo total:           {tiempo:.2f}s")
    print(f"  Tiempo GA:              {stats['ga_time']:.2f}s")
    print(f"  Tiempo vectorización:   {stats['vectorization_time']:.2f}s")
    print(f"  Cores utilizados:       {stats['num_cores']}")
    print(f"  Tareas creadas:         {stats['num_tasks']}")
    print(f"  Subtareas creadas:      {stats['num_subtasks']}")
    
    if 'mlp_stats' in stats:
        print(f"\n🧠 RESULTADOS DEL MODELO:")
        print(f"  Accuracy:               {stats['mlp_stats']['accuracy']:.4f}")
        print(f"  Tiempo entrenamiento:   "
              f"{stats['mlp_stats']['train_time']:.2f}s")
    
    print("="*70)