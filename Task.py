from dataclasses import dataclass
import numpy as np
import re
from multiprocessing import Pool, cpu_count 
from typing import List, Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns # type: ignore
from scipy.sparse import vstack

# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

STOP_WORDS = {"the", "and", "is", "in", "at", "of", "a", "to", "for", "on",
              "it", "this", "that"}

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

    def total_load(self) -> float:
        """Calcula la carga total del procesador"""
        return self.current_load


@dataclass
class PopulationState:
    """Estado de la población para el AG adaptativo"""
    delta_fm: float  # Tasa de cambio del mejor fitness
    sigma_f: float   # Varianza de fitness
    sigma_p: float   # Varianza de población (diversidad)
    
    def quantize(self) -> Tuple[str, str, str]:
        """Cuantifica el estado en niveles (high/medium/low)"""
        # Rangos basados en observaciones empíricas
        def quantize_value(value: float, low_thresh: float, high_thresh: float) -> str:
            if value < low_thresh:
                return 'low'
            elif value < high_thresh:
                return 'medium'
            else:
                return 'high'
        
        # ΔFm: cambio en fitness (0.0 a ~0.1+)
        delta_level = quantize_value(self.delta_fm, 0.01, 0.05)
        
        # σf: varianza de fitness (0.0 a ~0.3+)
        sigma_f_level = quantize_value(self.sigma_f, 0.05, 0.15)
        
        # σp: diversidad (0.0 a 1.0)
        sigma_p_level = quantize_value(self.sigma_p, 0.3, 0.6)
        
        return (delta_level, sigma_f_level, sigma_p_level)

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
    chunk_size = total_texts // 2
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