"""
secuencial.py

Vectorización TF-IDF Secuencial (Baseline para Comparación)
===========================================================

Este módulo implementa la vectorización TF-IDF de forma secuencial
(sin paralelismo) para servir como baseline en las comparaciones de
rendimiento contra el enfoque con Algoritmo Genético.

PROPÓSITO:
----------
Medir el tiempo que toma vectorizar textos sin ninguna optimización
de balanceo de carga, ejecutando todo en un solo hilo/proceso.

IMPORTANCIA:
-----------
La comparación justa requiere:
1. Mismo preprocesamiento de texto
2. Misma configuración de TF-IDF
3. Mismo dataset
Solo difiere en: secuencial vs. paralelo con GA
"""

import pandas as pd
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Dict, Any
from scipy.sparse import spmatrix, vstack
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Palabras vacías para preprocesamiento de texto
# DEBE ser idéntico al usado en GA.py para comparación justa
STOP_WORDS = {"the", "and", "is", "in", "at", "of", "a", "to", "for", "on",
              "it", "this", "that"}


# ============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# ============================================================================

def procesar_texto(texto: str) -> List[str]:
    """
    Preprocesamiento de texto con expresiones regulares
    
    IMPORTANTE: Esta función es IDÉNTICA a process_text() en GA.py
    para asegurar una comparación justa.
    
    PASOS:
    1. Convertir a minúsculas
    2. Eliminar URLs (http://, www.)
    3. Eliminar caracteres no alfabéticos
    4. Tokenizar por espacios
    5. Filtrar stopwords y tokens muy cortos
    
    Args:
        texto: Texto crudo a procesar
        
    Returns:
        Lista de tokens limpios
    
    Example:
        >>> procesar_texto("Check out http://example.com for more!")
        ['check', 'example']
    """
    # Paso 1: Normalizar a minúsculas
    texto = texto.lower()
    
    # Paso 2: Eliminar URLs completas
    texto = re.sub(r'http\S+|www\.\S+', '', texto)    
    # Paso 3: Eliminar todo excepto letras y espacios
    texto = re.sub(r"[^a-z\s]", "", texto)
    
    # Paso 4: Tokenización simple por espacios
    tokens = texto.split()
    
    # Paso 5: Filtrar stopwords y tokens muy cortos (≤2 caracteres)
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]
    
    return tokens


# ============================================================================
# FUNCIÓN SIMPLE DE VECTORIZACIÓN
# ============================================================================

def vectorizar_secuencialmente(df: pd.DataFrame) -> float:
    """
    Vectorización TF-IDF secuencial simple
    
    Esta es la función más básica: toma un DataFrame, vectoriza todo
    en un solo paso, y retorna el tiempo que tardó.
    
    PROCESO:
    1. Crear vectorizador TF-IDF
    2. Ajustar y transformar en un solo paso
    3. Medir tiempo total
    
    Args:
        df: DataFrame con columna 'text'
    
    Returns:
        Tiempo de ejecución en segundos
    
    Note:
        No retorna la matriz X para mantener la función simple.
        Para obtener X, usar sequential_vectorize().
    """
    print("  Iniciando vectorización secuencial...")
    
    # Iniciar temporizador
    inicio = time.time()
    
    # Crear vectorizador TF-IDF con misma configuración que GA.py
    vectorizer = TfidfVectorizer(
        tokenizer=procesar_texto,  # Usar nuestra función de preprocesamiento
        lowercase=False,            # Ya normalizamos en procesar_texto
        max_features=1000           # Vocabulario de 1000 palabras más frecuentes
    )
    
    # fit_transform: Ajustar vocabulario Y transformar textos en un solo paso
    # Este es el paso que consume la mayor parte del tiempo
    x = vectorizer.fit_transform(df["text"])
    
    # Detener temporizador
    fin = time.time()
    
    tiempo_total = fin - inicio
    print(f"  Vectorización secuencial completada en {tiempo_total:.4f}s")
    
    return tiempo_total


# ============================================================================
# FUNCIÓN AVANZADA PARA COMPARACIÓN
# ============================================================================

def sequential_vectorize(
    df: pd.DataFrame,
    intervalo: int = 20000,
    train_model: bool = False
) -> Tuple[spmatrix, float, Dict[str, Any]]:
    """
    Vectorización TF-IDF secuencial con logging detallado
    
    Esta función es compatible con la interfaz de GA.py para permitir
    comparaciones justas entre el enfoque secuencial y paralelo.
    
    DIFERENCIAS CON vectorizar_secuencialmente():
    1. Retorna la matriz X (no solo el tiempo)
    2. Retorna estadísticas detalladas
    3. Muestra progreso durante la ejecución
    4. Divide la transformación en lotes para mostrar progreso
    
    PROCESO DETALLADO:
    1. Extraer textos del DataFrame
    2. Ajustar vocabulario en todo el dataset
    3. Transformar por lotes (mostrando progreso)
    4. Combinar todos los lotes
    5. Retornar matriz, tiempo y estadísticas
    
    Args:
        df: DataFrame con columna 'text'
        intervalo: Cada cuántos tweets mostrar progreso (default: 20,000)
    
    Returns:
        Tupla (X, tiempo_total, stats):
        - X: Matriz dispersa de características TF-IDF (forma: num_textos × 1000)
        - tiempo_total: Tiempo total de ejecución en segundos
        - stats: Diccionario con estadísticas detalladas
    
    Example:
        >>> df = pd.read_csv('datos.csv')
        >>> X, tiempo, stats = sequential_vectorize(df, intervalo=10000)
        >>> print(f"Vectorizados {X.shape[0]} textos en {tiempo:.2f}s")
    """
    # Extraer lista de textos del DataFrame
    texts = df["text"].tolist()
    total_texts = len(texts)
    
    print("  Ajustando vocabulario (secuencial)...")
    
    # ========================================================================
    # PASO 1: INICIAR TEMPORIZADOR TOTAL
    # ========================================================================
    inicio_total = time.time()
    
    # ========================================================================
    # PASO 2: CREAR Y AJUSTAR VECTORIZADOR
    # ========================================================================
    # TF-IDF: Term Frequency - Inverse Document Frequency
    # - TF: frecuencia de palabra en documento
    # - IDF: penaliza palabras muy comunes en todos los documentos
    # Resultado: palabras distintivas tienen valores más altos
    
    vectorizer = TfidfVectorizer(
        tokenizer=procesar_texto,
        lowercase=False,
        max_features=1000  # Limitar vocabulario a 1000 palabras más importantes
    )
    
    # Ajustar: aprender el vocabulario de todo el dataset
    # Esto identifica las 1000 palabras más frecuentes (después de preprocesar)
    inicio_fit = time.time()
    vectorizer.fit(texts)
    tiempo_fit = time.time() - inicio_fit
    
    # ========================================================================
    # PASO 3: TRANSFORMAR POR LOTES (CON PROGRESO)
    # ========================================================================
    inicio_transform = time.time()
    
    if total_texts <= intervalo:
        # CASO 1: Dataset pequeño
        # Transformar todo de una vez (más eficiente)
        X = vectorizer.transform(texts)
        tiempo_transcurrido = time.time() - inicio_total
        print(f"  Lote {total_texts}/{total_texts} - Tiempo acum: {tiempo_transcurrido:.2f}s")
        
    else:
        # CASO 2: Dataset grande
        # Transformar por lotes para mostrar progreso al usuario
        X_chunks = []
        
        # Procesar en lotes de tamaño 'intervalo'
        for i in range(0, total_texts, intervalo):
            # Calcular índice final del lote actual
            end_idx = min(i + intervalo, total_texts)
            
            # Extraer textos del lote actual
            chunk_texts = texts[i:end_idx]
            
            # Transformar lote a vectores TF-IDF
            X_chunk = vectorizer.transform(chunk_texts)
            X_chunks.append(X_chunk)
            
            # Mostrar progreso al usuario
            tiempo_acumulado = time.time() - inicio_total
            print(f"  Lote {end_idx}/{total_texts} - Tiempo acum: {tiempo_acumulado:.2f}s")
        
        # Concatenar todos los lotes verticalmente
        # vstack: apilar matrices dispersas verticalmente (filas)
        X = vstack(X_chunks)
    
    tiempo_transform = time.time() - inicio_transform
    tiempo_total = time.time() - inicio_total
    
    # ========================================================================
    # PASO 4: PREPARAR ESTADÍSTICAS
    # ========================================================================
    # Estadísticas compatibles con las retornadas por GA.py
    # Esto permite comparar ambos enfoques usando las mismas métricas
    
    stats = {
        'total_texts': total_texts,      # Número de textos procesados
        'num_cores': 1,                  # Secuencial = 1 core
        'fit_time': tiempo_fit,          # Tiempo de ajuste del vocabulario
        'transform_time': tiempo_transform,  # Tiempo de transformación
        'total_time': tiempo_total       # Tiempo total de ejecución
    }
    
    if train_model and 'class' in df.columns:
        y = df['class'].values
        mlp_stats = train_and_evaluate_mlp(X, y, method_name="Secuencial")
        stats['mlp_stats'] = mlp_stats
    
    return X, tiempo_total, stats

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
# PUNTO DE ENTRADA (solo para pruebas)
# ============================================================================

if __name__ == '__main__':
    """
    Pruebas del módulo secuencial
    
    Este bloque solo se ejecuta cuando se corre el script directamente
    (no cuando se importa como módulo).
    
    Propósito: Verificar que ambas funciones funcionan correctamente.
    """
    print("=" * 60)
    print("PRUEBAS DE VECTORIZACIÓN SECUENCIAL")
    print("=" * 60)
    
    # Cargar datos de prueba (primeros 10,000 tweets)
    print("\nCargando datos de prueba...")
    df_prueba = pd.read_csv('Suicide_Detection.csv').head(10000)
    print(f"Cargados {len(df_prueba)} tweets")
    
    # ========================================================================
    # PRUEBA 1: Función simple
    # ========================================================================
    print("\n" + "-" * 60)
    print("PRUEBA 1: vectorizar_secuencialmente (función simple)")
    print("-" * 60)
    
    tiempo1 = vectorizar_secuencialmente(df_prueba)
    print(f"✓ Completado en {tiempo1:.4f}s")
    
    # ========================================================================
    # PRUEBA 2: Función avanzada
    # ========================================================================
    print("\n" + "-" * 60)
    print("PRUEBA 2: sequential_vectorize (función para comparación)")
    print("-" * 60)
    
    X, tiempo2, stats = sequential_vectorize(df_prueba, intervalo=5000)
    
    print(f"\n✓ Completado en {tiempo2:.4f}s")
    print(f"  Forma de matriz: {X.shape}")
    print(f"  Estadísticas:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"    - {key}: {value:.4f}s")
        else:
            print(f"    - {key}: {value}")
    
    # ========================================================================
    # VERIFICACIÓN
    # ========================================================================
    print("\n" + "=" * 60)
    print("VERIFICACIÓN")
    print("=" * 60)
    
    # Ambas funciones deberían tomar tiempo similar
    diferencia = abs(tiempo1 - tiempo2)
    print(f"Diferencia de tiempo: {diferencia:.4f}s")
    
    if diferencia < 1.0:
        print("✓ Ambas funciones tienen rendimiento similar")
    else:
        print("⚠ Las funciones tienen diferencia significativa")
    
    print("\n" + "=" * 60)
    print("PRUEBAS COMPLETADAS")
    print("=" * 60)