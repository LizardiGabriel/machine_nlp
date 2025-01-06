import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import json
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score


def leer_jsonl(ruta_archivo):
    datos = []
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        for linea in f:
            datos.append(json.loads(linea))
    return datos


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


# --- Código principal ---

# Iniciar el temporizador
start_time = time.time()

# los datos incluyen del 0 al 4.

ruta_test = "./data_span/test.jsonl"

datos_test = leer_jsonl(ruta_test)

# Convertir las listas de diccionarios a DataFrames de pandas
df_emotion_corpus_test = pd.DataFrame(datos_test)

label_name = df_emotion_corpus_test["label"].unique()
label_name = [str(label) for label in label_name]
print('label name: ', label_name)


# Definir el dispositivo a utilizar
device = torch.device("mps")

# Cargar el modelo y el tokenizer guardados
model_path = "./model_span"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=5)

# Mover el modelo al dispositivo
model.to(device)

# Crear el conjunto de datos de prueba
test_dataset = Dataset(df_emotion_corpus_test["text"].tolist(), df_emotion_corpus_test["label"].tolist())

# Definir los argumentos de entrenamiento (solo para la evaluación)
training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=44,
)

# Crear el entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
)

# Evaluar el modelo en el conjunto de prueba
print("Evaluando el modelo en el conjunto de prueba...")
eval_results = trainer.evaluate(test_dataset)

# Imprimir los resultados de la evaluación
print(f"**Resultados de la evaluación en el conjunto de prueba:**\n{eval_results}")

# Obtener las predicciones para el conjunto de prueba
print("Prediciendo las etiquetas del conjunto de prueba...")
predictions = trainer.predict(test_dataset)

# Obtener las etiquetas predichas
predicted_labels = np.argmax(predictions.predictions, axis=1)

# Imprimir el informe de clasificación
print(classification_report(test_dataset.labels, predicted_labels, target_names=label_name))

# Calcular la matriz de confusión
cm = confusion_matrix(test_dataset.labels, predicted_labels)

# Mostrar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_name)
disp.plot()
plt.show()

# Detener el temporizador
end_time = time.time()

# Calcular el tiempo transcurrido
elapsed_time = end_time - start_time

# Imprimir el tiempo de ejecución
print(f"Tiempo de ejecución del programa: {elapsed_time:.2f} segundos")