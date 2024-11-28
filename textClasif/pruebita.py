import pandas as pd
import stanza
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib
import os

# Configuración de normalización
nlp = stanza.Pipeline('es', processors='tokenize,mwt,pos,lemma', tokenize_no_ssplit=True)

# Normalización de texto
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

# Guardar representaciones
def save_representation(matrix, filename):
    joblib.dump(matrix, filename)
    print(f"Representación guardada en {filename}")

# Cargar representaciones
def load_representation(filename):
    if os.path.exists(filename):
        print(f"Cargando representación desde {filename}")
        return joblib.load(filename)
    return None

# Flujo principal
def main():
    # 1. Cargar datos
    filepath = 'news.csv'
    data = pd.read_csv(filepath)
    data['Features'] = data['Title'] + " " + data['Content']

    # Normalizar texto
    print("Normalizando texto...")
    data['Normalized'] = data['Features'].apply(normalize_text)

    # 2. Generar y guardar representaciones
    vector_types = ['Frecuencia', 'Binarización', 'TF-IDF']
    for vector_type in vector_types:
        filename = f"{vector_type}_representation.pkl"
        matrix = load_representation(filename)

        if matrix is None:  # Si no existe, calcular y guardar
            print(f"Generando representación: {vector_type}")
            matrix, _ = generate_representations(data['Normalized'], vector_type)
            save_representation(matrix, filename)
        else:
            print(f"Representación {vector_type} cargada correctamente.")

    # Embeddings (placeholder: ajusta según modelo de embeddings que uses)
    embeddings_file = "Embeddings_representation.pkl"
    if not os.path.exists(embeddings_file):
        print("Generando representación de embeddings (opcional)")
        # Aquí debes integrar la generación de embeddings según tu modelo (Ej: FastText, Word2Vec)
        embeddings_matrix = None  # Sustituir con la matriz de embeddings generada
        save_representation(embeddings_matrix, embeddings_file)
    else:
        embeddings_matrix = load_representation(embeddings_file)

if __name__ == "__main__":
    main()
