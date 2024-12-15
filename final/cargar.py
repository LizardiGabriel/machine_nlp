import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Ruta del modelo guardado
model_dir = "./model"

# Cargar el modelo y el tokenizer
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)

# Asegurarte de que el modelo esté en el dispositivo correcto (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Función para hacer predicciones
def predict(texts):
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

# Realizar las predicciones
predictions = predict(new_texts)

# Imprimir los resultados
print("Predicciones para los textos:")
for text, pred in zip(new_texts, predictions):
    print(f"- Texto: '{text}' | Predicción: {pred}")
