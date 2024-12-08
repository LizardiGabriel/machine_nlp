from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("vapogore/clasificador-poemas")
model = AutoModelForSequenceClassification.from_pretrained("vapogore/clasificador-poemas")

config = AutoConfig.from_pretrained("vapogore/clasificador-poemas")
print(config.id2label)  # Esto mostrará las etiquetas asociadas a cada índice

def clasificar_poema(poema):
    inputs = tokenizer(poema, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1)
    return probs[0].tolist()

poema = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme"
probs = clasificar_poema(poema)
print(probs)

