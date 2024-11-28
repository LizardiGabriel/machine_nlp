import csv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import stanza
import spacy
import re

# Configuración de pipelines
stanza.download('es')
nlp_stanza = stanza.Pipeline('es', processors='tokenize,mwt,pos,lemma', tokenize_no_ssplit=True)
nlp_spacy = spacy.load('es_core_news_md')


# Funciones de normalización de texto
def normalize_text(text, tokenize=True, stopwords_removal=True, text_cleaning=True, lemmatization=True):
    if not isinstance(text, str):
        return ''

    tokens = []  # Lista de tokens procesados

    # Tokenización y procesamiento con Stanza si se requiere
    if tokenize or stopwords_removal or lemmatization:
        doc = nlp_stanza(text)
        tokens = [word for sentence in doc.sentences for word in sentence.words]

        # Eliminación de stopwords
        if stopwords_removal:
            tokens = [token for token in tokens if token.upos not in {'PRON', 'DET', 'ADP', 'CCONJ', 'AUX'}]

        # Lemmatización
        if lemmatization:
            tokens = [token.lemma for token in tokens]
        else:
            tokens = [token.text for token in tokens]

    # Limpieza de texto adicional
    if text_cleaning:
        tokens = [token.lower() for token in tokens if re.match(r'^[a-záéíóúñ]+$', token)]

    return ' '.join(tokens)


# Generar representaciones vectoriales
def generate_representations(texts, vector_type, ngram_range=(1, 1)):
    if vector_type == 'Frecuencia':
        vect = CountVectorizer(ngram_range=ngram_range)
    elif vector_type == 'Binarización':
        vect = CountVectorizer(ngram_range=ngram_range, binary=True)
    elif vector_type == 'TF-IDF':
        vect = TfidfVectorizer(ngram_range=ngram_range)
    else:
        raise ValueError("Tipo de vectorización no válida.")
    matrix = vect.fit_transform(texts)
    return matrix, vect


def generate_embeddings(texts):
    return np.array([nlp_spacy(text).vector for text in texts])


# Entrenamiento y evaluación
def train_and_evaluate_model(X_train, y_train, X_test, y_test, model, model_name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    text_report = classification_report(y_test, predictions)
    macro_f1 = report['macro avg']['f1-score']
    return macro_f1, f"Modelo: {model_name}\n{text_report}"


def main():
    filepath = 'news.csv'
    data = pd.read_csv(filepath)
    data['Features'] = data['Title'] + " " + data['Content']
    X = data['Features']
    y = data['Section']
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    normalizations = [
        {"tokenize": True, "stopwords_removal": True, "text_cleaning": True, "lemmatization": True},
        {"tokenize": True, "stopwords_removal": False, "text_cleaning": True, "lemmatization": True},
        {"tokenize": True, "stopwords_removal": True, "text_cleaning": False, "lemmatization": False},
    ]

    models_config = {
        "Logistic Regression": [
            LogisticRegression(max_iter=1000),
            LogisticRegression(max_iter=2000, solver='liblinear', C=0.5)
        ],
        "Naive Bayes": [
            MultinomialNB(),
            MultinomialNB(alpha=0.5)
        ],
        "SVM": [
            SVC(kernel='linear', C=1.0),
            SVC(kernel='rbf', C=1.0, gamma=0.1)
        ],
        "MLP": [
            MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),
            MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=2000)
        ]
    }

    vector_types = ['Frecuencia', 'Binarización', 'TF-IDF', 'Embeddings']
    resultados = []
    csv_rows = []  # Aquí se almacenan las filas del CSV

    for norm in normalizations:
        X_train = X_train_raw.apply(lambda x: normalize_text(x, **norm))
        X_test = X_test_raw.apply(lambda x: normalize_text(x, **norm))

        for vector_type in vector_types:
            for use_svd in [False, True]:
                if vector_type == 'Embeddings':
                    X_train_vect = generate_embeddings(X_train)
                    X_test_vect = generate_embeddings(X_test)
                else:
                    X_train_vect, vectorizer = generate_representations(X_train, vector_type)
                    X_test_vect = vectorizer.transform(X_test)

                if use_svd:
                    svd = TruncatedSVD(n_components=300)
                    X_train_vect = svd.fit_transform(X_train_vect)
                    X_test_vect = svd.transform(X_test_vect)

                for model_name, model_variants in models_config.items():
                    for model in model_variants:
                        if model_name == "Naive Bayes" and vector_type != 'Embeddings' and not use_svd:
                            macro_f1, reporte = train_and_evaluate_model(
                                X_train_vect, y_train, X_test_vect, y_test, model, f"{model_name} - {model}"
                            )
                        elif model_name != "Naive Bayes":
                            macro_f1, reporte = train_and_evaluate_model(
                                X_train_vect, y_train, X_test_vect, y_test, model, f"{model_name} - {model}"
                            )

                        resultados.append((
                            macro_f1,
                            f"Normalización: {norm}\nRepresentación: {vector_type} - SVD: {use_svd}\n{reporte}"
                        ))

                        # Agregar fila al CSV
                        csv_rows.append([
                            model_name,
                            str(model),
                            str(norm),
                            f"{vector_type} - SVD: {use_svd}",
                            macro_f1
                        ])

    # Ordenar resultados por macro_f1 (global)
    resultados.sort(key=lambda x: x[0], reverse=True)

    # Guardar archivo detallado
    with open("resultados_ordenados.txt", "w") as f:
        f.writelines([res[1] + '\n\n' for res in resultados])

    # Guardar archivo CSV resumido
    with open("resultados_resumidos.csv", mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Machine Learning Method", "ML Method Parameters", "Text Normalization", "Text Representation", "Avg F-Score"])
        csv_writer.writerows(csv_rows)

    print(f"Archivos 'resultados_ordenados.txt' y 'resultados_resumidos.csv' creados con éxito.")


if __name__ == '__main__':
    main()
