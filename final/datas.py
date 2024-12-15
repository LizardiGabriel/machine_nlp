
import json


def leer_jsonl(ruta_archivo):
    datos = []
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        for linea in f:
            datos.append(json.loads(linea))
    return datos

ruta_train = "./data/train.jsonl"
ruta_validation = "./data/validation.jsonl"
ruta_test = "./data/test.jsonl"

datos_train = leer_jsonl(ruta_train)
datos_validation = leer_jsonl(ruta_validation)
datos_test = leer_jsonl(ruta_test)


# imprimir 10datos de train que tengan en la columna label el valor 0
print('datos de train con label 0: ', [dato for dato in datos_train if dato['label'] == 0][:10])

# imprimir 10datos de validation que tengan en la columna label el valor 1
print('datos de validation con label 1: ', [dato for dato in datos_validation if dato['label'] == 1][:10])

# imprimir 10datos de test que tengan en la columna label el valor 2
print('datos de test con label 2: ', [dato for dato in datos_test if dato['label'] == 2][:10])

# imprimir 10datos de train que tengan en la columna label el valor 3
print('datos de train con label 3: ', [dato for dato in datos_train if dato['label'] == 3][:10])

# imprimir 10datos de validation que tengan en la columna label el valor 4
print('datos de validation con label 4: ', [dato for dato in datos_validation if dato['label'] == 4][:10])

# imprimir 10datos de test que tengan en la columna label el valor 5
print('datos de test con label 5: ', [dato for dato in datos_test if dato['label'] == 5][:10])



