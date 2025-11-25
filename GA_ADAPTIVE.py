"""
GA.py

Balanceo de Carga Dinámico Basado en Algoritmos Genéticos para Vectorización de Texto
VERSIÓN ADAPTATIVA basada en "An Adaptive Genetic Algorithm" (Kee et al.)
================================================================================
"""

import numpy as np
import pandas as pd
import time
import re
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore

from Task import *

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

AVAILABLE_CORES = cpu_count()

GA_CONFIG_ADAPTIVE = {
    'population_size': 20,
    'num_generations': 5,
    'heavy_multiplier': 1.2,
    'light_multiplier': 0.8,
    'mutation_rate': 0.15,
    'num_cores': AVAILABLE_CORES,
    # Parámetros adaptativos
    'adaptive': True,
    'training_epochs': 100,
    'training_generations_per_epoch': 1
}

# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class ControlVector:
    """Vector de control de parámetros del AG"""
    crossover_prob: float
    mutation_prob: float
    scaling_factor: float
    
    def copy(self) -> 'ControlVector':
        return ControlVector(
            self.crossover_prob,
            self.mutation_prob,
            self.scaling_factor
        )
    
    def randomize(self, base_vector: 'ControlVector', variation: float = 0.2) -> 'ControlVector':
        """Genera variación aleatoria del vector"""
        return ControlVector(
            np.clip(base_vector.crossover_prob + np.random.uniform(-variation, variation), 0.05, 0.99),
            np.clip(base_vector.mutation_prob + np.random.uniform(-variation, variation), 0.001, 0.5),
            np.clip(base_vector.scaling_factor + np.random.uniform(-variation*2, variation*2), 0.5, 3.0)
        )

# ============================================================================
# ALGORITMO GENÉTICO ADAPTATIVO
# ============================================================================

class AdaptiveStateTable:
    """Tabla de estados para mapear vectores de estado a vectores de control"""
    
    def __init__(self):
        self.table: Dict[Tuple[str, str, str], ControlVector] = {}
        self._initialize_table()
    
    def _initialize_table(self):
        """Inicializa la tabla con valores por defecto (Grefenstette)"""
        levels = ['low', 'medium', 'high']
        default_control = ControlVector(
            crossover_prob=0.95,
            mutation_prob=0.01,
            scaling_factor=1.0
        )
        
        for delta in levels:
            for sigma_f in levels:
                for sigma_p in levels:
                    self.table[(delta, sigma_f, sigma_p)] = default_control.copy()
    
    def get_control_vector(self, state: PopulationState) -> ControlVector:
        """Obtiene el vector de control para un estado dado"""
        quantized = state.quantize()
        return self.table[quantized].copy()
    
    def update_control_vector(self, state: PopulationState, control: ControlVector):
        """Actualiza el vector de control para un estado"""
        quantized = state.quantize()
        self.table[quantized] = control.copy()


class GeneticLoadBalancer:
    """
    Algoritmo Genético Adaptativo para Balanceo de Carga Dinámico
    Basado en "An Adaptive Genetic Algorithm" (Kee, Airey, Cyre)
    """

    def __init__(self, config: Dict[str, Any]):
        """Inicializa el algoritmo genético con configuración dada"""
        self.config = config
        self.num_processors = self.config['num_cores']
        self.population_size = self.config['population_size']
        self.num_generations = self.config['num_generations']
        self.H = self.config['heavy_multiplier']
        self.L = self.config['light_multiplier']
        
        # Parámetros adaptativos
        self.adaptive = self.config.get('adaptive', False)
        self.training_epochs = self.config.get('training_epochs', 100)
        self.training_gens_per_epoch = self.config.get('training_generations_per_epoch', 1)
        
        # Estado adaptativo
        self.state_table = AdaptiveStateTable() if self.adaptive else None
        self.is_trained = False
        self.fitness_history: List[float] = []
        self.best_fitness_history: List[float] = []
        
        # Parámetros actuales (pueden ser adaptativos)
        self.current_control = ControlVector(
            crossover_prob=0.95,
            mutation_prob=self.config.get('mutation_rate', 0.15),
            scaling_factor=1.0
        )

    def calculate_population_state(self, population: List[TaskMapping],
                                   fitness_values: List[float]) -> PopulationState:
        """Calcula el vector de estado de la población"""
        
        # ΔFm: Tasa de cambio del mejor fitness
        if len(self.best_fitness_history) >= 2:
            recent_best = self.best_fitness_history[-5:] if len(self.best_fitness_history) >= 5 else self.best_fitness_history
            if len(recent_best) > 1:
                delta_fm = (recent_best[-1] - recent_best[0]) / len(recent_best)
            else:
                delta_fm = 0.0
        else:
            delta_fm = 0.0
        
        # σf²: Varianza de fitness
        if len(fitness_values) > 1:
            sigma_f = np.std(fitness_values)
        else:
            sigma_f = 0.0
        
        # σp²: Diversidad de población (basada en diferencias en asignaciones)
        if len(population) > 1:
            # Calcular diversidad como promedio de diferencias entre individuos
            diversities = []
            for i in range(min(10, len(population))):  # Muestrear para eficiencia
                for j in range(i+1, min(10, len(population))):
                    diversity = self._calculate_hamming_diversity(
                        population[i], population[j]
                    )
                    diversities.append(diversity)
            sigma_p = np.mean(diversities) if diversities else 0.5
        else:
            sigma_p = 0.5
        
        return PopulationState(
            delta_fm=max(0.0, delta_fm),
            sigma_f=sigma_f,
            sigma_p=sigma_p
        )
    
    def _calculate_hamming_diversity(self, mapping1: TaskMapping, 
                                    mapping2: TaskMapping) -> float:
        """Calcula diversidad tipo Hamming entre dos mapeos"""
        total_tasks = sum(len(tasks) for tasks in mapping1.assignment)
        if total_tasks == 0:
            return 0.0
        
        differences = 0
        for proc_id in range(self.num_processors):
            tasks1 = set(mapping1.get_processor_tasks(proc_id))
            tasks2 = set(mapping2.get_processor_tasks(proc_id))
            differences += len(tasks1.symmetric_difference(tasks2))
        
        return differences / (2 * total_tasks)  # Normalizar

    def train_adaptive_table(self, num_tasks: int,
                           processor_states: List[ProcessorState],
                           verbose: bool = False):
        """Fase de entrenamiento del AG adaptativo (Table-based)"""
        if not self.adaptive or self.is_trained:
            return
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"🎓 FASE DE ENTRENAMIENTO ADAPTATIVO")
            print(f"{'='*70}")
            print(f"  Épocas: {self.training_epochs}")
            print(f"  Generaciones por época: {self.training_gens_per_epoch}")
        
        # Crear tareas sintéticas para entrenamiento
        synthetic_tasks = [
            Task(texts=[], size=np.random.randint(100, 1000), original_indices=[])
            for _ in range(num_tasks)
        ]
        
        training_start = time.time()
        
        for epoch in range(self.training_epochs):
            # Crear población inicial
            population_C = self.initialize_population(len(synthetic_tasks), processor_states)
            population_T = [m.copy() for m in population_C]
            
            # Calcular estado de población
            fitness_C = [
                self.calculate_fitness(m, synthetic_tasks, processor_states)
                for m in population_C
            ]
            
            state = self.calculate_population_state(population_C, fitness_C)
            
            # Obtener vector de control actual de la tabla
            control_C = self.state_table.get_control_vector(state)
            
            # Generar variación para población de prueba
            control_T = control_C.randomize(control_C)
            
            # mismo fitness para la poblacion Temporal que la constante antes de la variacion
            fitness_T = fitness_C
            
            # Evolucionar ambas poblaciones
            for gen in range(self.training_gens_per_epoch):
                # Población C con control de tabla
                self.current_control = control_C
                population_C, fitness_C = self._evolve_one_generation(
                    population_C, fitness_C, synthetic_tasks, processor_states
                )
                
                # Población T con control variado
                self.current_control = control_T
                population_T, fitness_T = self._evolve_one_generation(
                    population_T, fitness_T, synthetic_tasks, processor_states
                )
            
            # Comparar resultados
            avg_fitness_C = np.mean(fitness_C)
            avg_fitness_T = np.mean(fitness_T)
            
            # Actualizar tabla si T es mejor
            if avg_fitness_T > avg_fitness_C:
                self.state_table.update_control_vector(state, control_T)
                if verbose and epoch % 20 == 0:
                    improvement = ((avg_fitness_T - avg_fitness_C) / avg_fitness_C * 100)
                    print(f"  Época {epoch:3d}: Mejora {improvement:+.2f}% "
                          f"(Pc={control_T.crossover_prob:.3f}, "
                          f"Pm={control_T.mutation_prob:.3f}, "
                          f"α={control_T.scaling_factor:.2f})")
        
        training_time = time.time() - training_start
        self.is_trained = True
        
        if verbose:
            print(f"\n  ✓ Entrenamiento completado en {training_time:.2f}s")
            print(f"{'='*70}\n")

    def _evolve_one_generation(self, population: List[TaskMapping],
                              fitness_values: List[float],
                              tasks, processor_states: List[ProcessorState]
                              ) -> Tuple[List[TaskMapping], List[float]]:
        """Evoluciona la población una generación"""
        num_tasks = len(tasks)
        new_population = []
        
        # Elitismo
        sorted_indices = np.argsort(fitness_values)[::-1]
        new_population.append(population[sorted_indices[0]].copy())
        if len(population) > 1:
            new_population.append(population[sorted_indices[1]].copy())
        
        # Generar nueva población
        while len(new_population) < self.population_size:
            parent1 = self.roulette_wheel_selection(population, fitness_values)
            parent2 = self.roulette_wheel_selection(population, fitness_values)
            
            # Usar probabilidad de crossover adaptativa
            if np.random.random() < self.current_control.crossover_prob:
                child1, child2 = self.cycle_crossover(parent1, parent2, num_tasks)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Usar probabilidad de mutación adaptativa
            child1 = self.swap_mutation(child1, num_tasks, 
                                       self.current_control.mutation_prob)
            child2 = self.swap_mutation(child2, num_tasks,
                                       self.current_control.mutation_prob)
            
            new_population.extend([child1, child2])
        
        population = new_population[:self.population_size]
        
        # Recalcular fitness
        fitness_values = [
            self.calculate_fitness(mapping, tasks, processor_states)
            for mapping in population
        ]
        
        return population, fitness_values

    def initialize_population(self, num_tasks: int, 
                             processor_states: List[ProcessorState]) -> List[TaskMapping]:
        """Inicializa población con estrategias diversas"""
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
                # Greedy
                for task_idx in range(num_tasks):
                    loads_in_mapping = [
                        current_loads[p] + len(mapping.get_processor_tasks(p))
                        for p in range(self.num_processors)
                    ]
                    least_loaded = np.argmin(loads_in_mapping)
                    mapping.assign_task(least_loaded, task_idx)
                    
            elif i == 1:
                # Weighted random
                for task_idx in range(num_tasks):
                    processor = np.random.choice(self.num_processors, p=probabilities)
                    mapping.assign_task(processor, task_idx)
                    
            elif i == 2:
                # Round-robin
                sorted_procs = np.argsort(current_loads)
                for task_idx in range(num_tasks):
                    processor = sorted_procs[task_idx % self.num_processors]
                    mapping.assign_task(processor, task_idx)
                    
            else:
                # Random mixto
                for task_idx in range(num_tasks):
                    if np.random.random() < 0.7:
                        processor = np.random.choice(self.num_processors, p=probabilities)
                    else:
                        processor = np.random.randint(0, self.num_processors)
                    mapping.assign_task(processor, task_idx)
                    
            population.append(mapping)
            
        return population

    def calculate_fitness(self, mapping: TaskMapping, tasks,
                        processor_states: List[ProcessorState]) -> float:
        """Calcula fitness con power scaling adaptativo"""
        num_tasks = len(tasks)
        mapping.validate_and_fix(num_tasks)

        final_loads = []
        
        for proc_id in range(self.num_processors):
            current_load = processor_states[proc_id].total_load()
            new_task_indices = mapping.get_processor_tasks(proc_id)
            new_load = sum(
                tasks[tid].size for tid in new_task_indices 
                if 0 <= tid < num_tasks
            )
            final_loads.append(current_load + new_load)

        max_load = max(final_loads) if final_loads else 1.0
        min_load = min(final_loads) if final_loads else 0.0
        avg_load = sum(final_loads) / self.num_processors if self.num_processors > 0 else 0.0
        total_load = sum(final_loads)

        # Métricas de utilización
        if max_load > 0:
            min_utilization = min_load / max_load
            efficiency = avg_load / max_load
        else:
            min_utilization = 0.0
            efficiency = 0.0

        if avg_load > 0:
            std_dev = (sum((load - avg_load) ** 2 for load in final_loads) / self.num_processors) ** 0.5
            coef_variation = std_dev / avg_load
            uniformity = 1.0 / (1.0 + coef_variation)
        else:
            uniformity = 0.0

        if max_load > 0:
            throughput_score = avg_load / max_load
        else:
            throughput_score = 0.0

        # Fitness base
        fitness = (
            0.35 * min_utilization +
            0.25 * efficiency +
            0.30 * uniformity +
            0.10 * throughput_score
        )
        
        # Aplicar power scaling si es adaptativo
        if self.adaptive and self.current_control.scaling_factor != 1.0:
            alpha = self.current_control.scaling_factor
            fitness = fitness ** alpha
        
        # Bonificaciones y penalizaciones
        cores_well_utilized = sum(1 for load in final_loads if load >= 0.8 * avg_load)
        if cores_well_utilized >= 0.85 * self.num_processors:
            fitness *= 1.2
        
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

    def swap_mutation(self, mapping: TaskMapping, num_tasks: int,
                     mutation_rate: Optional[float] = None) -> TaskMapping:
        """Mutación por intercambio con tasa adaptativa"""
        mutated = mapping.copy()
        
        if mutation_rate is None:
            mutation_rate = self.current_control.mutation_prob

        if np.random.random() > mutation_rate:
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
        """Ciclo principal de evolución con adaptación"""
        num_tasks = len(tasks)

        if num_tasks == 0:
            return TaskMapping(self.num_processors)

        # Fase de entrenamiento si es adaptativo y no está entrenado
        if self.adaptive and not self.is_trained:
            self.train_adaptive_table(num_tasks, processor_states, verbose)

        population = self.initialize_population(num_tasks, processor_states)

        fitness_values = [
            self.calculate_fitness(mapping, tasks, processor_states)
            for mapping in population
        ]
        
        self.best_fitness_history.append(max(fitness_values))

        if verbose:
            print(f"\n🧬 Evolución del Algoritmo Genético {'ADAPTATIVO' if self.adaptive else 'ESTÁTICO'}:")
            if self.adaptive:
                state = self.calculate_population_state(population, fitness_values)
                control = self.state_table.get_control_vector(state)
                print(f"  Estado inicial: ΔFm={state.delta_fm:.4f}, "
                      f"σf={state.sigma_f:.4f}, σp={state.sigma_p:.4f}")
                print(f"  Control inicial: Pc={control.crossover_prob:.3f}, "
                      f"Pm={control.mutation_prob:.3f}, α={control.scaling_factor:.2f}")

        for gen in range(self.num_generations):
            # Actualizar parámetros adaptativamente si corresponde
            if self.adaptive and self.is_trained:
                state = self.calculate_population_state(population, fitness_values)
                self.current_control = self.state_table.get_control_vector(state)
            
            new_population = []

            sorted_indices = np.argsort(fitness_values)[::-1]
            new_population.append(population[sorted_indices[0]].copy())
            if len(population) > 1:
                new_population.append(population[sorted_indices[1]].copy())

            while len(new_population) < self.population_size:
                parent1 = self.roulette_wheel_selection(population, fitness_values)
                parent2 = self.roulette_wheel_selection(population, fitness_values)

                if np.random.random() < self.current_control.crossover_prob:
                    child1, child2 = self.cycle_crossover(parent1, parent2, num_tasks)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                child1 = self.swap_mutation(child1, num_tasks)
                child2 = self.swap_mutation(child2, num_tasks)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

            fitness_values = [
                self.calculate_fitness(mapping, tasks, processor_states)
                for mapping in population
            ]
            
            self.best_fitness_history.append(max(fitness_values))
            
            if verbose and gen % 5 == 0:
                if self.adaptive:
                    state = self.calculate_population_state(population, fitness_values)
                    q_state = state.quantize()
                    print(f"  Gen {gen:2d}: fitness={max(fitness_values):.4f}, "
                          f"estado={q_state}, "
                          f"Pc={self.current_control.crossover_prob:.2f}, "
                          f"Pm={self.current_control.mutation_prob:.3f}")
                else:
                    print(f"  Gen {gen:2d}: fitness={max(fitness_values):.4f}")

        best_idx = np.argmax(fitness_values)
        best_mapping = population[best_idx]
        best_mapping.fitness_value = fitness_values[best_idx]

        best_mapping.validate_and_fix(num_tasks)

        return best_mapping


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def print_utilization_stats(mapping: TaskMapping, tasks, 
                           processor_states: List[ProcessorState],
                           num_cores: int, show_details: bool = True):
    """Muestra estadísticas de UTILIZACIÓN de cores"""
    print("\n" + "="*80)
    print("⚡ ANÁLISIS DE UTILIZACIÓN DE CORES")
    print("="*80)
    
    final_loads = []
    task_counts = []
    
    for proc_id in range(num_cores):
        current_load = processor_states[proc_id].total_load()
        task_ids = mapping.get_processor_tasks(proc_id)
        new_load = sum(tasks[tid].size for tid in task_ids if tid < len(tasks))
        final_loads.append(current_load + new_load)
        task_counts.append(len(task_ids))
    
    max_load = max(final_loads) if final_loads else 1.0
    min_load = min(final_loads) if final_loads else 0.0
    avg_load = sum(final_loads) / num_cores if num_cores > 0 else 0.0
    total_load = sum(final_loads)
    
    utilizations = [(load / max_load * 100) if max_load > 0 else 0.0 
                    for load in final_loads]
    
    min_util = min(utilizations)
    max_util = max(utilizations)
    avg_util = sum(utilizations) / num_cores
    
    efficiency = (avg_load / max_load * 100) if max_load > 0 else 0.0
    
    std_dev = (sum((u - avg_util) ** 2 for u in utilizations) / num_cores) ** 0.5
    coef_variation = (std_dev / avg_util) if avg_util > 0 else 0.0
    
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
    
    if optimal_cores >= num_cores * 0.8:
        status = "✅ EXCELENTE - Utilización óptima"
    elif optimal_cores >= num_cores * 0.6:
        status = "🟢 BUENO - Mayoría bien utilizada"
    elif efficiency > 70:
        status = "🟡 ACEPTABLE - Mejorable"
    else:
        status = "🔴 POBRE - Requiere optimización"
    
    print(f"\n💯 EVALUACIÓN:          {status}")
    
    if show_details:
        print(f"\n📋 DETALLE POR CORE:")
        print(f"  {'Core':<6} {'Tareas':<8} {'Carga':<12} {'Utilización':<15} {'Barra Visual':<30}")
        print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*15} {'-'*30}")
        
        for proc_id in range(num_cores):
            load = final_loads[proc_id]
            tasks_count = task_counts[proc_id]
            util = utilizations[proc_id]
            
            bar_length = int(min(util, 100) / 5)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            
            if util < 50:
                color = "🔴"
            elif util < 80:
                color = "🟡"
            elif util <= 100:
                color = "🟢"
            else:
                color = "🔥"
            
            print(f"  {color} {proc_id:<4} {tasks_count:<8} {load:<12,} "
                  f"{util:>6.1f}%          {bar}")
    
    print("="*80 + "\n")


# ============================================================================
# FUNCIÓN PRINCIPAL DE VECTORIZACIÓN
# ============================================================================

def vectorize_with_ga_adaptive_load_balancing(
    df,
    config: Dict[str, Any] = GA_CONFIG_ADAPTIVE,
    verbose: bool = False,
    train_model: bool = False
) -> Tuple[Any, float, Dict[str, Any]]:
    """
    Vectorización TF-IDF con balanceo de carga basado en GA Adaptativo
    """
    
    num_cores = config['num_cores']
    texts = df["text"].tolist()
    total_texts = len(texts)
    
    adaptive_str = "ADAPTATIVO" if config.get('adaptive', False) else "ESTÁTICO"
    print(f"  Usando {num_cores} cores con AG {adaptive_str}")
    
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
    
    processor_states = [
        ProcessorState(processor_id=i, current_load=0.0, queue=[])
        for i in range(num_cores)
    ]
    
    ga = GeneticLoadBalancer(config)
    
    stats: Dict[str, Any] = {
        'total_texts': total_texts,
        'num_tasks': num_tasks_total,
        'num_subtasks': num_subtasks_total,
        'num_cores': num_cores,
        'adaptive': config.get('adaptive', False),
        'ga_generations': config['num_generations'],
        'ga_population': config['population_size'],
        'chunk_size': chunk_size,
        'subtasks_per_task': num_subtasks_per_task,
        'aga_time': 0.0,
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
        
        ga_start = time.time()
        best_mapping = ga.evolve(window_subtasks, processor_states, verbose=verbose)
        aga_time = time.time() - ga_start
        stats['aga_time'] += aga_time
        
        print(f"  (GA: {aga_time:.2f}s, fitness: {best_mapping.fitness_value:.4f})", 
              end=" ")
        
        if verbose:
            print()
            print_utilization_stats(best_mapping, window_subtasks, 
                                   processor_states, num_cores, show_details=True)
        
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
        
        for proc_id in range(num_cores):
            subtask_indices = best_mapping.get_processor_tasks(proc_id)
            added_load = sum(
                window_subtasks[sid].size 
                for sid in subtask_indices 
                if sid < len(window_subtasks)
            )
            processor_states[proc_id].current_load += added_load
        
        processed_subtasks = window_end
        window_count += 1
    
    print(f"  Reconstruyendo matriz...")
    
    vectors_list = [vec for _, vec in indexed_vectors]
    X = vstack(vectors_list)
    
    print(f"  OK: Matriz construida: {X.shape} ({len(indexed_vectors)} vectores)")
    
    total_time = time.time() - start_total
    stats['total_time'] = total_time
    
    print(f"\n  Resumen de tiempos:")
    print(f"  - Total: {total_time:.2f}s")
    print(f"  - GA: {stats['aga_time']:.2f}s "
          f"({stats['aga_time']/total_time*100:.1f}%)")
    print(f"  - Vectorización: {stats['vectorization_time']:.2f}s "
          f"({stats['vectorization_time']/total_time*100:.1f}%)")
    
    if train_model and 'class' in df.columns:
        print(f"\n{'='*70}")
        print(f"🔗 EMPAREJAMIENTO VECTOR-ETIQUETA")
        print(f"{'='*70}")
        
        y_original = df['class'].values
        y_aligned = np.zeros(len(indexed_vectors), dtype=y_original.dtype)
        
        for i, (original_idx, _) in enumerate(indexed_vectors):
            y_aligned[i] = y_original[original_idx]
        
        print(f"  OK: Etiquetas emparejadas: {len(y_aligned)}")
        
        mlp_stats = train_and_evaluate_mlp(X, y_aligned, 
                                          method_name=f"GA-{'Adaptativo' if stats['adaptive'] else 'Estatico'}")
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
    
    filename = f'confusion_matrix_{method_name.lower().replace(" ", "_").replace("-", "_")}.png'
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
    """Pruebas del módulo GA con comportamiento adaptativo"""
    print("="*70)
    print("🧬 GA LOAD BALANCER ADAPTATIVO")
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
    print("🚀 INICIANDO VECTORIZACIÓN CON GA ADAPTATIVO")
    print("="*70)
    
    X, tiempo, stats = vectorize_with_ga_adaptive_load_balancing(
        df_test,
        config=GA_CONFIG_ADAPTIVE,
        verbose=True,
        train_model=True
    )
    
    print("\n" + "="*70)
    print("✅ RESULTADO FINAL")
    print("="*70)
    print(f"  Modo:                   {'ADAPTATIVO' if stats['adaptive'] else 'ESTÁTICO'}")
    print(f"  Textos procesados:      {X.shape[0]:,}")
    print(f"  Dimensiones del vector: {X.shape[1]:,}")
    print(f"  Tiempo total:           {tiempo:.2f}s")
    print(f"  Tiempo GA:              {stats['aga_time']:.2f}s")
    print(f"  Tiempo vectorización:   {stats['vectorization_time']:.2f}s")
    print(f"  Cores utilizados:       {stats['num_cores']}")
    
    if 'mlp_stats' in stats:
        print(f"\n🧠 RESULTADOS DEL MODELO:")
        print(f"  Accuracy:               {stats['mlp_stats']['accuracy']:.4f}")
        print(f"  Tiempo entrenamiento:   {stats['mlp_stats']['train_time']:.2f}s")
    
    print("="*70)