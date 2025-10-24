import numpy as np
import pandas as pd

import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

stop_words = {"the","and","is","in","at","of","a","to","for","on","it","this","that"}  # set básico de stopwords

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


df = pd.read_csv('Suicide_Detection.csv')
df = df.drop(df.columns[0], axis=1)

vectorizer = TfidfVectorizer(
    tokenizer=procesar_texto,  # usamos regex
    lowercase=False,            # ya convertimos a minúsculas
    max_features=1000           # límite de vocabulario
)

inicio = time.time()                        # Aqui esta lo bueno: iniciamos el timer para saber cuanto tardamos en el proceso de limpiar y vectorizar el texto
x = vectorizer.fit_transform(df["text"])
fin = time.time()                           # Termina el proceso de vectorizacion
# ===========================LO QUE NOS INTERESA===========================
total_tweets = len(df)
tiempo_total = fin - inicio
tiempo_promedio = tiempo_total / total_tweets

print(f"Tiempo total: {tiempo_total:.2f} s")
print(f"Tweets procesados: {total_tweets}")
print(f"Tiempo promedio por tweet: {tiempo_promedio:.6f} s") # sera nuestra carga de los algos. de PSO y GA
# =========================================================================

y = df["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Crear y configurar la red neuronal
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # dos capas ocultas
    activation='relu',             # función de activación
    solver='adam',                 # optimizador
    max_iter=20,                   # número de épocas
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

"""
OUTPUT ACTUAL: 

Tiempo total: 14.83 s
Tweets procesados: 232074
Tiempo promedio por tweet: 0.000064 s

Exactitud: 0.9043706821021789

Reporte de clasificación:
              precision    recall  f1-score   support

 non-suicide       0.90      0.91      0.90     34824
     suicide       0.91      0.90      0.90     34799

    accuracy                           0.90     69623
   macro avg       0.90      0.90      0.90     69623
weighted avg       0.90      0.90      0.90     69623
"""