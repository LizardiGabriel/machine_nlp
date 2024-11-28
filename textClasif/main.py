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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import stanza

# Carga de datos
def load_data(filepath):
    data = pd.read_csv(filepath)
    data['Features'] = data['Title'] + " " + data['Content']
    return data

# Normalización de texto
nlp = stanza.Pipeline('es', processors='tokenize,mwt,pos,lemma', tokenize_no_ssplit=True)

def normalize_text(text):
    if isinstance(text, str):
        doc = nlp(text)
        normalized_words = [
            word.lemma for sentence in doc.sentences for word in sentence.words
            if not word.upos in {'PRON', 'DET', 'ADP', 'CCONJ', 'AUX'}
        ]
        return ' '.join(normalized_words)
    return ''

# Generación de representaciones vectoriales
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

# Entrenamiento y evaluación de modelos
def train_and_evaluate_model(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

# Flujo principal
def main():
    # 1. Cargar datos
    filepath = 'news.csv'
    data = load_data(filepath)

    # 2. División en conjuntos de entrenamiento y prueba
    X = data['Features']
    y = data['Section']
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Normalización
    X_train = X_train_raw.apply(normalize_text)
    X_test = X_test_raw.apply(normalize_text)

    # 4. Representaciones vectoriales
    vector_types = ['Frecuencia', 'Binarización', 'TF-IDF']
    for vector_type in vector_types:
        print(f"\nRepresentación: {vector_type}")
        X_train_vect, vectorizer = generate_representations(X_train, vector_type)
        X_test_vect = vectorizer.transform(X_test)

        # Reducción de dimensionalidad (opcional)
        svd = TruncatedSVD(n_components=100, random_state=42)
        X_train_reduced = svd.fit_transform(X_train_vect)
        X_test_reduced = svd.transform(X_test_vect)

        # 5. Evaluación de modelos
        print("\nLogistic Regression")
        train_and_evaluate_model(X_train_reduced, y_train, X_test_reduced, y_test, LogisticRegression())

        print("\nNaive Bayes")
        train_and_evaluate_model(X_train_reduced, y_train, X_test_reduced, y_test, MultinomialNB())

        print("\nSVM")
        train_and_evaluate_model(X_train_reduced, y_train, X_test_reduced, y_test, SVC())

        print("\nMLP")
        train_and_evaluate_model(X_train_reduced, y_train, X_test_reduced, y_test, MLPClassifier())



main()
