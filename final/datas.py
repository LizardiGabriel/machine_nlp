
import json


def leer_jsonl(ruta_archivo):
    datos = []
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        for linea in f:
            datos.append(json.loads(linea))
    return datos


ruta_train = "./data_span/train.jsonl"
ruta_validation = "./data_span/validation.jsonl"
ruta_test = "./data_span/test.jsonl"

datos_train = leer_jsonl(ruta_train)
datos_validation = leer_jsonl(ruta_validation)
datos_test = leer_jsonl(ruta_test)


def contar_labels(datos):
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for dato in datos:
        label = dato.get('label')
        if label in label_counts:
            label_counts[label] += 1
    return label_counts

# Contar los labels en los datos de entrenamiento
label_counts_train = contar_labels(datos_train)
print('Label counts in training data:', label_counts_train)

# Contar los labels en los datos de validación
label_counts_validation = contar_labels(datos_validation)
print('Label counts in validation data:', label_counts_validation)

# Contar los labels en los datos de prueba
label_counts_test = contar_labels(datos_test)
print('Label counts in test data:', label_counts_test)


def contar_total_labels(label_counts_train):
    total_count = sum(label_counts_train.values())
    return total_count

# Contar el total de labels en los datos de entrenamiento
total_label_count_train = contar_total_labels(label_counts_train)
print('Total label count in training data:', total_label_count_train)

# Contar el total de labels en los datos de validación
total_label_count_validation = contar_total_labels(label_counts_validation)
print('Total label count in validation data:', total_label_count_validation)

# Contar el total de labels en los datos de prueba
total_label_count_test = contar_total_labels(label_counts_test)
print('Total label count in test data:', total_label_count_test)