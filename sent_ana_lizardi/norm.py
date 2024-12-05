import pandas as pd
from spacy.lang.es.stop_words import STOP_WORDS
import spacy
import pickle


corpus = pd.read_excel('Rest_Mex_2022.xlsx')
corpus.head()
corpus['Title'] = corpus['Title'].astype(str)
corpus['Opinion'] = corpus['Opinion'].astype(str)
corpus['texto'] = corpus['Title'] + ' ' + corpus['Opinion']
corpus = corpus.drop(['Title', 'Opinion'], axis=1)
corpus.head()


nlp = spacy.load('es_core_news_sm')


def procesar_texto(texto):
    try:
        # Tokenización
        doc = nlp(texto)
        tokens = [token.text for token in doc]
        texto_tokenizado = " ".join(tokens)

        # Lematización
        doc = nlp(texto_tokenizado)
        lemas = [token.lemma_ for token in doc]
        texto_lematizado = " ".join(lemas)

        # Eliminación de stopwords
        doc = nlp(texto_lematizado)
        tokens_filtrados = [token.text for token in doc if token.pos_ not in ["DET", "ADP", "CCONJ", "PRON"]]
        texto_filtrado = " ".join(tokens_filtrados)
    except:
        # Ignorar contenido si no se puede procesar
        texto_filtrado = ""

    return texto_filtrado


# Normalizar contenido
lista_contenido = []

for i, contenido in enumerate(corpus['texto']):
    lista_contenido.append(procesar_texto(contenido))
    print(f"\rProgreso: {i/len(corpus):.1%}", end="")


print(corpus['texto'][0])
print("--------------------------------------------------")
print(lista_contenido[0])

# Guardar lista de contenido normalizado
with open('normalizado.pkl', 'wb') as archivo:
    pickle.dump(lista_contenido, archivo)

