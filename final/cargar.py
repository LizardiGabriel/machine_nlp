import torch
from transformers import BertForSequenceClassification, BertTokenizer



test_texts = [
    "I am very happy to have passed the exam.",
    "I feel sad and disappointed with the results.",
    "This movie scares me.",
    "I am angry because my phone has been stolen.",
    "I am surprised that there are so many people here.",
    "I feel disgusted when I see the garbage on the street."
]


# Ruta del modelo guardado
model_dir = "./model"

# Cargar el modelo y el tokenizer
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)

# Asegurarte de que el modelo esté en el dispositivo correcto (CPU o GPU)
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
model.to(device)

# Crear una función para hacer predicciones
def predict(texts):
    model.eval()  # Modo de evaluación
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    encodings = {key: val.to(device) for key, val in encodings.items()}
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    return predictions.cpu().numpy()


# Hacer predicciones
predictions = predict(test_texts)
print(predictions)

