import pandas as pd
import json


def procesar_corpus(ruta_archivo, ruta_salida_train, ruta_salida_validation, ruta_salida_test):

    df = pd.read_excel(ruta_archivo)

    df['Title'] = df['Title'].astype(str)
    df['Opinion'] = df['Opinion'].astype(str)
    df['text'] = df['Title'] + ' ' + df['Opinion']
    df = df.rename(columns={'Polarity': 'label'})


    df = df.drop(['Title', 'Opinion'], axis=1)


    # Dividir el DataFrame en conjuntos de entrenamiento, validación y prueba (70%, 15%, 15%)
    train_df = df.sample(frac=0.70, random_state=0)
    rest_df = df.drop(train_df.index)
    validation_df = rest_df.sample(frac=0.5, random_state=0)
    test_df = rest_df.drop(validation_df.index)

    # Guardar los DataFrames en archivos JSONL
    guardar_jsonl(train_df[['text', 'label']], ruta_salida_train)
    guardar_jsonl(validation_df[['text', 'label']], ruta_salida_validation)
    guardar_jsonl(test_df[['text', 'label']], ruta_salida_test)


def guardar_jsonl(df, ruta_archivo):
    with open(ruta_archivo, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            json.dump({'text': row['text'], 'label': int(row['label'])}, f, ensure_ascii=False)
            f.write('\n')



# Rutas de los archivos
ruta_archivo = 'Rest_Mex_2022.xlsx'
ruta_salida_train = './data_span/train.jsonl'
ruta_salida_validation = './data_span/validation.jsonl'
ruta_salida_test = './data_span/test.jsonl'

procesar_corpus(ruta_archivo, ruta_salida_train, ruta_salida_validation, ruta_salida_test)