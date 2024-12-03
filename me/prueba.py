import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import pipeline
import time
import torch

# Prueba de CPU/GPU con Machine Learning
def ml_test():
    print("Generando datos sintéticos para clasificación...")
    X, y = make_classification(n_samples=100000, n_features=100, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Entrenando modelo de regresión logística...")
    start_time = time.time()
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    print(f"Entrenamiento completado en {elapsed_time:.2f} segundos")
    print("Evaluando modelo...")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("Reporte de clasificación:\n", report)

# Prueba de inferencia con un modelo de lenguaje
def llm_test():
    print("\nCargando modelo de lenguaje (esto puede tardar)...")
    device = 0 if torch.cuda.is_available() else -1  # Usa GPU si está disponible
    text_generator = pipeline("text-generation", model="gpt2", device=device)
    print("Modelo cargado. Generando texto...")

    input_text = "Once upon a time in a faraway land"
    start_time = time.time()
    output = text_generator(input_text, max_length=50, num_return_sequences=1)
    elapsed_time = time.time() - start_time

    print(f"Inferencia completada en {elapsed_time:.2f} segundos")
    print("Texto generado:\n", output[0]['generated_text'])

if __name__ == "__main__":
    print("==== Prueba de Machine Learning ====")
    ml_test()

    print("\n==== Prueba de Modelos de Lenguaje ====")
    llm_test()
