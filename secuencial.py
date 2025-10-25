# librerias para guardar el csv
import numpy as np
import pandas as pd

# librerias para limpieza y vectorizacion
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer

# concatenar todos los vectores de cada 20k tweets
from scipy.sparse import vstack

# importamos modelo de red neuronal
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =====================================================================================================================
"""funcion para limpieza"""
# =====================================================================================================================
stop_words = {"the","and","is","in","at","of","a","to","for","on","it","this","that"} # set básico de stopwords

def procesar_texto(texto: str) -> list[str]:
    # minúsculas
    texto = texto.lower()
    # quitar URLs, menciones, hashtags
    texto = re.sub(r'http\S+|www.\S+', '', texto)
    # quitar números y signos
    texto = re.sub(r"[^a-z\s]", "", texto)
    # tokenización simple por espacios
    tokens = texto.split()
    # quitar stopwords y tokens cortos
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return tokens

# =====================================================================================================================
"""leyendo dataset"""
# =====================================================================================================================
df = pd.read_csv('Suicide_Detection.csv')
df = df.drop(df.columns[0], axis=1)

# =====================================================================================================================
"""vectorizando con mediciones cada 20k tweets"""
# =====================================================================================================================
vectorizer = TfidfVectorizer(
    tokenizer=procesar_texto,
    lowercase=False,
    max_features=1000
)

# Lista para almacenar resultados intermedios
tiempos_parciales = []
intervalo = 20000  # cada 20k tweets
total_tweets = len(df)

# Primero fit del vocabulario con todo el dataset
print("Ajustando vocabulario...")
vectorizer.fit(df["text"])

# Ahora transformar por lotes y medir tiempo
inicio_total = time.time()
vectores_parciales = []

for i in range(0, total_tweets, intervalo):
    fin_lote = min(i + intervalo, total_tweets)
    df_lote = df.iloc[i:fin_lote]  # solo este lote
    
    # Transformar solo este lote
    x_lote = vectorizer.transform(df_lote["text"])
    vectores_parciales.append(x_lote)
    
    tiempo_transcurrido = time.time() - inicio_total
    tweets_procesados = fin_lote
    
    # Guardar información del lote
    tiempos_parciales.append({
        "tweets_procesados": tweets_procesados,
        "tiempo_acumulado_s": tiempo_transcurrido,
        "tiempo_promedio_s": tiempo_transcurrido / tweets_procesados
    })
    
    print(f"Tweets procesados: {tweets_procesados}/{total_tweets} - Tiempo acumulado: {tiempo_transcurrido:.2f}s")

x = vstack(vectores_parciales)

tiempo_total = time.time() - inicio_total

# ===========================LO QUE NOS INTERESA===========================
tiempo_promedio = tiempo_total / total_tweets
print(f"\nTiempo total: {tiempo_total:.2f} s")
print(f"Tweets procesados: {total_tweets}")
print(f"Tiempo promedio por tweet: {tiempo_promedio:.6f} s")

# Guardar resultados parciales en CSV
df_tiempos = pd.DataFrame(tiempos_parciales)
df_tiempos.to_csv("tiempo.csv", index=False)
print("\nTiempos parciales guardados en 'tiempo.csv'")

# =====================================================================================================================
"""entrenando modelo"""
# =====================================================================================================================
y = df["class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# crear y configurar la red neuronal
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=20,
    random_state=42
)

# Entrenar el modelo
mlp.fit(x_train, y_train)

# Predecir
y_pred = mlp.predict(x_test)

# Evaluar
print("Exactitud:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
