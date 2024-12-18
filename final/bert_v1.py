import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizer, TrainingArguments, BertForSequenceClassification, Trainer
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

# los datos incluyen: sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5).

ruta_train = "train.jsonl"
ruta_validation = "validation.jsonl"
ruta_test = "test.jsonl"

datos_train = leer_jsonl(ruta_train)
datos_validation = leer_jsonl(ruta_validation)
datos_test = leer_jsonl(ruta_test)

# Convertir las listas de diccionarios a DataFrames de pandas
df_emotion_corpus_train = pd.DataFrame(datos_train)
df_emotion_corpus_validation = pd.DataFrame(datos_validation)
df_emotion_corpus_test = pd.DataFrame(datos_test)

print(df_emotion_corpus_validation)

df_emotion_corpus_validation["label_name"] = df_emotion_corpus_validation["label"]
print(df_emotion_corpus_validation[["text", "label_name"]])
print(df_emotion_corpus_validation["label_name"])
print(type(df_emotion_corpus_validation["label_name"]))

label_name = df_emotion_corpus_validation["label_name"].unique()
label_name = [str(label) for label in label_name]
print('label name: ', label_name)


device = torch.device("mps")


#https://huggingface.co/bert-base-uncased
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

#Training using pretrained model weights
model_ckpt = "bert-base-uncased"

model = (BertForSequenceClassification.from_pretrained(model_ckpt, num_labels=6))

# Mover el modelo a la GPU
model.to(device)

# Crear los conjuntos de datos
train_dataset = Dataset(df_emotion_corpus_train["text"].tolist(), df_emotion_corpus_train["label"].tolist())
validation_dataset = Dataset(df_emotion_corpus_validation["text"].tolist(), df_emotion_corpus_validation["label"].tolist())
test_dataset = Dataset(df_emotion_corpus_test["text"].tolist(), df_emotion_corpus_test["label"].tolist())

# Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Crear el entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
)

# Entrenar el modelo
print("Entrenando el modelo...")
train_results = trainer.train()

# Imprimir los resultados del entrenamiento
print(f"**Resultados del entrenamiento:**\n{train_results}")

# Guardar el modelo y el tokenizer necesarios para usarlo después
print("Guardando el modelo y el tokenizer...")
trainer.save_model("./model")  # Guarda el modelo y los pesos
tokenizer.save_pretrained("./model")  # Guarda el tokenizer
print("Modelo y tokenizer guardados correctamente.")

# Evaluar el modelo
print("Evaluando el modelo...")
eval_results = trainer.evaluate()

# Imprimir los resultados de la evaluación
print(f"**Resultados de la evaluación:**\n{eval_results}")

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