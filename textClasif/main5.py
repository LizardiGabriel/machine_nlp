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
import re  # Para extraer macro avg


# Carga de datos
def load_data(filepath):
    data = pd.read_csv(filepath)
    data['Features'] = data['Title'] + " " + data['Content']
    return data


# Normalización de texto
nlp_stanza = stanza.Pipeline('es', processors='tokenize,mwt,pos,lemma', tokenize_no_ssplit=True)


def normalize_text(text):
    if isinstance(text, str):
        doc = nlp_stanza(text)
        normalized_words = [
            word.lemma for sentence in doc.sentences for word in sentence.words
            if not word.upos in {'PRON', 'DET', 'ADP', 'CCONJ', 'AUX'}
        ]
        return ' '.join(normalized_words)
    return ''


# Representaciones vectoriales
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


# Generar embeddings con spaCy
# python -m spacy download es_core_news_md
nlp_spacy = spacy.load('es_core_news_md')


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


# Flujo principal
def main():
    # Ruta al archivo CSV
    filepath = 'news.csv'

    # Cargar datos
    data = load_data(filepath)
    X = data['Features']
    y = data['Section']
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalización
    X_train = X_train_raw.apply(normalize_text)
    X_test = X_test_raw.apply(normalize_text)

    # Tipos de representaciones
    vector_types = ['Frecuencia', 'Binarización', 'TF-IDF', 'Embeddings']

    # Modelos y parámetros
    models_config = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "SVM": SVC(),
        "MLP": MLPClassifier(max_iter=1000)
    }

    resultados = []

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

            for model_name, model in models_config.items():
                print(f"\nModel: {model_name} - Text Representation: {vector_type} - SVD: {use_svd}")

                if model_name == "Naive Bayes" and vector_type != 'Embeddings' and not use_svd:
                    macro_f1, reporte = train_and_evaluate_model(X_train_vect, y_train, X_test_vect, y_test, model, model_name)
                    resultados.append((macro_f1, f"Representación: {vector_type} - SVD: {use_svd} - {reporte}"))

                elif model_name != "Naive Bayes":
                    macro_f1, reporte = train_and_evaluate_model(X_train_vect, y_train, X_test_vect, y_test, model, model_name)
                    resultados.append((macro_f1, f"Representación: {vector_type} - SVD: {use_svd} - {reporte}"))

    # Ordenar los resultados por macro avg (F1-score) en orden descendente
    resultados.sort(key=lambda x: x[0], reverse=True)

    # Guardar reportes
    with open("reportes_ordenados.txt", "w") as f:
        # primero guardar la config de los modelos
        f.write('Modelos:\n')
        for model_name, model in models_config.items():
            f.write(f"{model_name}: {model}\n")
        f.write('\n\n')
        # luego guardar los resultados ordenados
        f.writelines([res[1] + '\n\n' for res in resultados])

    print("Resultados: en orden")
    print('---' * 20)
    print('\n'.join([res[1] for res in resultados]))
    print('---' * 20)


if __name__ == '__main__':
    main()
