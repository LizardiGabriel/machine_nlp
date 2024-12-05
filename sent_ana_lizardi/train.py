import os
import re
import pickle
import statistics
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

# Importar los modelos de clasificación
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import spacy
nlp_spacy = spacy.load('es_core_news_md')


def generate_embeddings(texts):
    return np.array([nlp_spacy(text).vector for text in texts])


# representacion de los datos
def representacion_datos(method, x_train, x_test):
    if method == 'tfidf':
        vectorizador = TfidfVectorizer(token_pattern=r'(?u)\w\w+|\w\w+\n|\.')
    elif method == 'binario':
        vectorizador = CountVectorizer(binary=True, token_pattern=r'(?u)\w\w+|\w\w+\n|\.')
    elif method == 'frecuencia':
        vectorizador = CountVectorizer(token_pattern=r'(?u)\w\w+|\w\w+\n|\.')

    X_train_vector = vectorizador.fit_transform(x_train)
    X_test_vector = vectorizador.transform(x_test)

    print(X_train_vector[0].shape)

    return vectorizador, X_train_vector, X_test_vector


def load_sel():
    lexicon_sel = {}
    input_file = open('SEL_full.txt', 'r')

    # Para cada línea del archivo
    for line in input_file:
        # formato lexicon:
        #abundancia	0	0	50	50	0.83	Alegría

        # Separar la línea en una lista de palabras
        palabras = line.split("\t")
        # Eliminar el salto de línea
        palabras[6]= re.sub('\n', '', palabras[6])

        # guarde la palabra y su valor de polaridad
        pair = (palabras[6], palabras[5])

        if lexicon_sel:
            if palabras[0] not in lexicon_sel:
                # se crea una lista con la palabra y su valor de polaridad
                lista = [pair]
                # se agrega la lista al diccionario
                lexicon_sel[palabras[0]] = lista
            else:
                lexicon_sel[palabras[0]].append (pair)
        else:
            lista = [pair]
            lexicon_sel[palabras[0]] = lista

    input_file.close()
    del lexicon_sel['Palabra'];

    #Estructura resultante
    #'hastiar': [('Enojo\n', '0.629'), ('Repulsi\xf3n\n', '0.596')]

    return lexicon_sel


def getSELFeatures(cadenas, lexicon_sel):

    #'hastiar': [('Enojo\n', '0.629'), ('Repulsi\xf3n\n', '0.596')]
    features = []
    for cadena in cadenas:
        valor_alegria = 0.0
        valor_enojo = 0.0
        valor_miedo = 0.0
        valor_repulsion = 0.0
        valor_sorpresa = 0.0
        valor_tristeza = 0.0
        cadena_palabras = re.split('\s+', cadena)
        dic = {}
        for palabra in cadena_palabras:
            if palabra in lexicon_sel:
                caracteristicas = lexicon_sel[palabra]
                for emocion, valor in caracteristicas:
                    if emocion == 'Alegría':
                        valor_alegria = valor_alegria + float(valor)
                    elif emocion == 'Tristeza':
                        valor_tristeza = valor_tristeza + float(valor)
                    elif emocion == 'Enojo':
                        valor_enojo = valor_enojo + float(valor)
                    elif emocion == 'Repulsión':
                        valor_repulsion = valor_repulsion + float(valor)
                    elif emocion == 'Miedo':
                        valor_miedo = valor_miedo + float(valor)
                    elif emocion == 'Sorpresa':
                        valor_sorpresa = valor_sorpresa + float(valor)
        dic['__alegria__'] = valor_alegria
        dic['__tristeza__'] = valor_tristeza
        dic['__enojo__'] = valor_enojo
        dic['__repulsion__'] = valor_repulsion
        dic['__miedo__'] = valor_miedo
        dic['__sorpresa__'] = valor_sorpresa

        #Esto es para los valores acumulados del mapeo a positivo (alegría + sorpresa) y negativo (enojo + miedo + repulsión + tristeza)
        dic['acumuladopositivo'] = dic['__alegria__'] + dic['__sorpresa__']
        dic['acumuladonegative'] = dic['__enojo__'] + dic['__miedo__'] + dic['__repulsion__'] + dic['__tristeza__']

        features.append (dic)
    return features


def polaridad_cadenas(polaridades):
    polaridad_cadenas = []
    for x in range(len(polaridades)):
        polaridad_cadena_pos = np.array([polaridades[x]['acumuladopositivo']])
        polaridad_cadena_neg = np.array([polaridades[x]['acumuladonegative']])
        polaridad_cadena = np.concatenate((polaridad_cadena_pos, polaridad_cadena_neg), axis=0)
        polaridad_cadenas.append(polaridad_cadena)
    return polaridad_cadenas


def select_model(model):
    if model == 'multinomial':
        clf = MultinomialNB(alpha=0.1, fit_prior=True)
    elif model == 'logistic':
        clf = LogisticRegression(solver='lbfgs', max_iter=2000)
    elif model == 'logistic1000':
        clf = LogisticRegression(max_iter=1000)
    elif model == 'logistic05':
        clf = LogisticRegression(solver='liblinear', max_iter=2000, C=0.5)
    elif model == 'randomforest':
        clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    elif model == 'knn':
        clf = KNeighborsClassifier(n_neighbors=3)
    elif model == 'mlp':
        clf = MLPClassifier(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', alpha=0.0001, learning_rate='constant', max_iter=200, batch_size='auto', early_stopping=False)
    elif model == 'gradientboosting':
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, min_samples_leaf=1, subsample=1.0)
    elif model == "mlp-cordova":
        clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', alpha=0.001, max_iter=500, random_state=0)

    return clf




def train_parameters(metodo, modelo, balanceo):
    print(f"Entrenando modelo {modelo} con método {metodo} y balanceo {balanceo}")

    # Cargar lexicon
    if os.path.exists('lexicon_sel.pkl'):
        with open('lexicon_sel.pkl', 'rb') as lexicon_sel_file:
            lexicon_sel = pickle.load(lexicon_sel_file)
    else:
        lexicon_sel = load_sel()
        with open('lexicon_sel.pkl', 'wb') as lexicon_sel_file:
            pickle.dump(lexicon_sel, lexicon_sel_file)

    # Cargar corpus
    with open("normalizado.pkl", "rb") as f:
        corpus = pickle.load(f)

    data = pd.read_excel('Rest_Mex_2022.xlsx')
    y = data['Polarity'].values

    x_train, x_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2, random_state=0)

    # Crear polaridades
    polaridades_train = getSELFeatures(x_train, lexicon_sel)
    polaridades_test = getSELFeatures(x_test, lexicon_sel)

    # Crear matriz de polaridades
    polaridad_cadenas_train = polaridad_cadenas(polaridades_train)
    polaridad_cadenas_test = polaridad_cadenas(polaridades_test)

    matriz_polaridades_train = csr_matrix(polaridad_cadenas_train)
    matriz_polaridades_test = csr_matrix(polaridad_cadenas_test)

    # Representación de los datos

    vectorizador, X_train_vector, X_test_vector = representacion_datos(metodo, x_train, x_test)

    X_train_vector = hstack((X_train_vector, matriz_polaridades_train))
    X_test_vector = hstack((X_test_vector, matriz_polaridades_test))

    # Balanceo de datos en el conjunto de entrenamiento
    if balanceo == "oversampling":
        ros = RandomOverSampler(random_state=0)
        X_train_vector, y_train = ros.fit_resample(X_train_vector, y_train)
    elif balanceo == "subsampling":
        rus = RandomUnderSampler(random_state=0)
        X_train_vector, y_train = rus.fit_resample(X_train_vector, y_train)
    elif balanceo == "smote":
        smote = SMOTE(random_state=0)
        X_train_vector, y_train = smote.fit_resample(X_train_vector, y_train)

    # Modelo seleccionado
    clf = select_model(modelo)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    f1_macro = []

    # Validación cruzada
    for entrenamiento, prueba in skf.split(X_train_vector, y_train):
        rasgos_entrenamiento = X_train_vector[entrenamiento]
        rasgos_prueba = X_train_vector[prueba]

        clases_entrenamiento = y_train[entrenamiento]
        clases_prueba = y_train[prueba]

        clf.fit(rasgos_entrenamiento, clases_entrenamiento)
        clase_predicha = clf.predict(rasgos_prueba)

        f1_macro.append(f1_score(clases_prueba, clase_predicha, average='macro'))
        print(f"\rProgreso: {len(f1_macro)/skf.get_n_splits():.1%}", end="")

    print(f"\nModelo: {modelo}")
    print(f"F1 macro: {round(statistics.mean(f1_macro), 2)*100}%")

    print(classification_report(clases_prueba, clase_predicha))

    # Matriz de confusión
    mat = confusion_matrix(clases_prueba, clase_predicha)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('Etiquetas verdaderas')
    plt.ylabel('Etiquetas predichas')
    plt.show()

    # Guardar reporte en txt
    with open(f"reportes/{modelo}-{metodo}-{balanceo}-rasgos.txt", "a") as f:
        f.write(f"{classification_report(clases_prueba, clase_predicha)}\n")
        f.write("Matriz de confusión:\n")
        f.write(f"{mat}\n")

    # Guardar modelo y vectorizador
    with open(f'./pkl/mod-{modelo}-{metodo}-{balanceo}.pkl', 'wb') as f:
        pickle.dump(clf, f)

    with open(f'./pkl/vec-{modelo}-{metodo}-{balanceo}.pkl', 'wb') as f:
        pickle.dump(vectorizador, f)




def train_and_evaluate_best_model():
    # Define the models and balance options
    modelos = ['multinomial', 'logistic', 'logistic1000']
    metodos = ['binario', 'frecuencia', 'tfidf']

    balanceos = ['', 'smote']

    best_f1_score = 0
    best_model = None
    best_method = None
    best_balance = None

    # Diccionario para almacenar los resultados de F1 Macro
    f1_scores = {}

    # Entrenar y evaluar todos los modelos
    for metodo in metodos:
        for modelo in modelos:
            for balanceo in balanceos:
                print(f"Evaluando modelo {modelo} con método {metodo} y balanceo {balanceo}")

                # Entrenar el modelo y obtener el F1 Macro
                f1_macro = []

                # Cargar lexicon
                if os.path.exists('lexicon_sel.pkl'):
                    with open('lexicon_sel.pkl', 'rb') as lexicon_sel_file:
                        lexicon_sel = pickle.load(lexicon_sel_file)
                else:
                    lexicon_sel = load_sel()
                    with open('lexicon_sel.pkl', 'wb') as lexicon_sel_file:
                        pickle.dump(lexicon_sel, lexicon_sel_file)

                # Cargar corpus
                with open("normalizado.pkl", "rb") as f:
                    corpus = pickle.load(f)

                data = pd.read_excel('Rest_Mex_2022.xlsx')
                y = data['Polarity'].values

                x_train, x_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2, random_state=0)

                # Crear polaridades
                polaridades_train = getSELFeatures(x_train, lexicon_sel)
                polaridades_test = getSELFeatures(x_test, lexicon_sel)

                # Crear matriz de polaridades
                polaridad_cadenas_train = polaridad_cadenas(polaridades_train)
                polaridad_cadenas_test = polaridad_cadenas(polaridades_test)

                matriz_polaridades_train = csr_matrix(polaridad_cadenas_train)
                matriz_polaridades_test = csr_matrix(polaridad_cadenas_test)

                # Representación de los datos
                vectorizador, X_train_vector, X_test_vector = representacion_datos(metodo, x_train, x_test)

                X_train_vector = hstack((X_train_vector, matriz_polaridades_train))
                X_test_vector = hstack((X_test_vector, matriz_polaridades_test))

                # Balanceo de datos en el conjunto de entrenamiento
                if balanceo == "oversampling":
                    ros = RandomOverSampler(random_state=0)
                    X_train_vector, y_train = ros.fit_resample(X_train_vector, y_train)
                elif balanceo == "subsampling":
                    rus = RandomUnderSampler(random_state=0)
                    X_train_vector, y_train = rus.fit_resample(X_train_vector, y_train)
                elif balanceo == "smote":
                    smote = SMOTE(random_state=0)
                    X_train_vector, y_train = smote.fit_resample(X_train_vector, y_train)

                # Modelo seleccionado
                clf = select_model(modelo)

                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

                # Validación cruzada
                for entrenamiento, prueba in skf.split(X_train_vector, y_train):
                    rasgos_entrenamiento = X_train_vector[entrenamiento]
                    rasgos_prueba = X_train_vector[prueba]

                    clases_entrenamiento = y_train[entrenamiento]
                    clases_prueba = y_train[prueba]

                    clf.fit(rasgos_entrenamiento, clases_entrenamiento)
                    clase_predicha = clf.predict(rasgos_prueba)

                    f1_macro.append(f1_score(clases_prueba, clase_predicha, average='macro'))

                # Promedio F1 Macro
                promedio_f1 = statistics.mean(f1_macro)
                print(f"F1 Macro promedio para {modelo}-{metodo}-{balanceo}: {round(promedio_f1, 2)*100}%")

                # Almacenar el mejor modelo
                if promedio_f1 > best_f1_score:
                    best_f1_score = promedio_f1
                    best_model = modelo
                    best_method = metodo
                    best_balance = balanceo

    print(f"Mejor modelo: {best_model} con {best_method} y {best_balance} - F1 Macro: {round(best_f1_score, 2)*100}%")

    # Ahora entrenamos el mejor modelo en el conjunto completo de datos usando SMOTE
    print(f"Entrenando el mejor modelo en el conjunto completo con balanceo SMOTE...")

    # Cargar corpus
    with open("normalizado.pkl", "rb") as f:
        corpus = pickle.load(f)

    data = pd.read_excel('Rest_Mex_2022.xlsx')
    y = data['Polarity'].values

    x_train, x_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2, random_state=0)

    # Cargar lexicon
    if os.path.exists('lexicon_sel.pkl'):
        with open('lexicon_sel.pkl', 'rb') as lexicon_sel_file:
            lexicon_sel = pickle.load(lexicon_sel_file)
    else:
        lexicon_sel = load_sel()
        with open('lexicon_sel.pkl', 'wb') as lexicon_sel_file:
            pickle.dump(lexicon_sel, lexicon_sel_file)

    # Crear polaridades
    polaridades_train = getSELFeatures(x_train, lexicon_sel)
    polaridades_test = getSELFeatures(x_test, lexicon_sel)

    # Crear matriz de polaridades
    polaridad_cadenas_train = polaridad_cadenas(polaridades_train)
    polaridad_cadenas_test = polaridad_cadenas(polaridades_test)

    matriz_polaridades_train = csr_matrix(polaridad_cadenas_train)
    matriz_polaridades_test = csr_matrix(polaridad_cadenas_test)

    # Representación de los datos
    vectorizador, X_train_vector, X_test_vector = representacion_datos(best_method, x_train, x_test)

    X_train_vector = hstack((X_train_vector, matriz_polaridades_train))
    X_test_vector = hstack((X_test_vector, matriz_polaridades_test))

    # Aplicar SMOTE
    smote = SMOTE(random_state=0)
    X_train_vector, y_train = smote.fit_resample(X_train_vector, y_train)

    # Modelo seleccionado
    clf = select_model(best_model)

    clf.fit(X_train_vector, y_train)

    # Predecir en el conjunto de prueba
    clase_predicha = clf.predict(X_test_vector)

    # Evaluación del modelo
    print(f"Evaluando el modelo en el conjunto de prueba...")

    print(f"F1 Macro: {f1_score(y_test, clase_predicha, average='macro')*100}%")
    print(classification_report(y_test, clase_predicha))

    # Matriz de confusión
    mat = confusion_matrix(y_test, clase_predicha)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('Etiquetas verdaderas')
    plt.ylabel('Etiquetas predichas')
    plt.show()

    # Guardar el mejor modelo y vectorizador
    with open(f'./pkl/best-mod-{best_model}-{best_method}-{best_balance}.pkl', 'wb') as f:
        pickle.dump(clf, f)

    with open(f'./pkl/best-vec-{best_model}-{best_method}-{best_balance}.pkl', 'wb') as f:
        pickle.dump(vectorizador, f)

# Llamar a la función principal
if __name__ == "__main__":
    train_and_evaluate_best_model()










