import os
import re
import pickle
import numpy as np
from scipy.sparse import hstack, csr_matrix

with open("./pkl/mod-logistic-binario-.pkl", "rb") as f:
    modelo = pickle.load(f)

with open("./pkl/vec-logistic-binario-.pkl", "rb") as f:
    vectorizador = pickle.load(f)


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





lexicon_sel_file = open ('./lexicon_sel.pkl','rb')
lexicon_sel = pickle.load(lexicon_sel_file)

texto_prueba = "Es un lugar muy poco agradable, no me gusta para nada"

# --> Procesar texto
texto_prueba = [texto_prueba]

vectorizador_prueba = vectorizador.transform(texto_prueba)

caracteristicas_prueba = getSELFeatures(texto_prueba, lexicon_sel)
polaridad_cadenas_prueba = polaridad_cadenas(caracteristicas_prueba)
matriz_prueba = csr_matrix(polaridad_cadenas_prueba)

vectorizador_prueba = hstack((vectorizador_prueba, matriz_prueba))



prediccion_prueba = modelo.predict(vectorizador_prueba)

print("Predicción: ", prediccion_prueba)

probabilidades_prueba = modelo.predict_proba(vectorizador_prueba)
print("Probabilidades: ", probabilidades_prueba)