from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("vapogore/clasificador-poemas")
model = AutoModelForSequenceClassification.from_pretrained("vapogore/clasificador-poemas")

def clasificar_poema(poema):
    inputs = tokenizer(poema, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1)
    return probs[0].tolist()

poema = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme"
probs = clasificar_poema(poema)
print(probs)

