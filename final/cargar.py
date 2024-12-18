import torch
from transformers import BertForSequenceClassification, BertTokenizer
import time

# Ruta del modelo guardado
model_dir = "./model"

# Cargar el modelo y el tokenizer
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)

# Función para hacer predicciones
def predict(texts, device):
    model.eval()  # Modo de evaluación
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    encodings = {key: val.to(device) for key, val in encodings.items()}
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    return predictions.cpu().numpy()

# Nuevos textos para predecir
new_texts = [
    "I am feeling very excited about my vacation!",
    "This is the worst service I have ever experienced.",
    "The weather today is so pleasant and sunny.",
    "I am furious that my order was delivered late!",
    "I can't believe how delicious this food tastes!",
    "i feel a little stunned but can t imagine what the folks who were working in the studio up until this morning are feeling.",
    "i feel shocked and sad at the fact that there are so many sick people"
]

# --- Usando MPS ---
device = torch.device("mps")
model.to(device)

start_time = time.time()
predictions_mps = predict(new_texts, device)
end_time = time.time()

print("Predicciones usando MPS:")
for text, pred in zip(new_texts, predictions_mps):
    print(f"- Texto: '{text}' | Predicción: {pred}")

print(f"Tiempo de ejecución con MPS: {end_time - start_time:.4f} segundos")

# --- Usando CPU ---
device = torch.device("cpu")
model.to(device)

start_time = time.time()
predictions_cpu = predict(new_texts, device)
end_time = time.time()

print("\nPredicciones usando CPU:")
for text, pred in zip(new_texts, predictions_cpu):
    print(f"- Texto: '{text}' | Predicción: {pred}")

print(f"Tiempo de ejecución con CPU: {end_time - start_time:.4f} segundos")