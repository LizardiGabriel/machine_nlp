import csv
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

def guardar_resultados(nombre_csv, resultados, n):
    with open(nombre_csv, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        if n == 2:
            writer.writerow(["Term1", "Term2", "Frec Bigrama", "Frec Contexto (Term1)", "Probabilidad Condicional"])
        elif n == 3:
            writer.writerow(["Term1", "Term2", "Term3", "Frec Trigrama", "Frec Contexto (Term1 Term2)", "Probabilidad Condicional"])
        writer.writerows(resultados)

def sacar_ngramas(nombre_archivo, n):
    with open(nombre_archivo, "r", encoding="utf-8") as archivo:
        contenido = archivo.readlines()

    vectorizer = CountVectorizer(ngram_range=(n, n), token_pattern=r'\S+', lowercase=False)

    # token_pattern=r'<s>(.*?)</s>|https?://\S+|\S+'

    X = vectorizer.fit_transform(contenido)
    vocabulario = vectorizer.get_feature_names_out()
    ngram_frecs = X.toarray().sum(axis=0)

    contexto_frecs = defaultdict(int)
    resultados = []

    for ngram, freq in zip(vocabulario, ngram_frecs):
        terms = ngram.split()
        if n == 2:
            term1, term2 = terms
            contexto_frecs[term1] += freq
        elif n == 3:
            term1_term2 = ' '.join(terms[:2])
            contexto_frecs[term1_term2] += freq

    for ngram, freq in zip(vocabulario, ngram_frecs):
        terms = ngram.split()
        if n == 2:
            term1, term2 = terms
            contexto_freq = contexto_frecs[term1]
            probabilidad_condicional = freq / contexto_freq
            resultados.append([term1, term2, freq, contexto_freq, probabilidad_condicional])
        elif n == 3:
            term1, term2, term3 = terms
            contexto_freq = contexto_frecs[f"{term1} {term2}"]
            probabilidad_condicional = freq / contexto_freq
            resultados.append([term1, term2, term3, freq, contexto_freq, probabilidad_condicional])

    resultados.sort(key=lambda x: x[-3], reverse=True)

    nombre_extraido = nombre_archivo.split("/")[-1].split(".")[0]
    nombre_csv = f"./ngramas/{nombre_extraido}_{n}.csv"
    guardar_resultados(nombre_csv, resultados, n)
    return nombre_csv
