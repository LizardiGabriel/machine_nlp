import pandas as pd
import numpy as np
import nltk
import string
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline  # Importar Pipeline de imblearn

# Descargar stopwords en español
nltk.download('stopwords')

# Cargar el corpus
df = pd.read_excel('Rest_Mex_2022.xlsx', engine='openpyxl')

# Preprocesamiento del texto
# Concatenar 'Title' y 'Opinion' y manejar valores faltantes
df['Title'] = df['Title'].astype(str).fillna('')
df['Opinion'] = df['Opinion'].astype(str).fillna('')
df['texto'] = df['Title'] + ' ' + df['Opinion']

# Función para limpiar el texto
spanish_stopwords = stopwords.words('spanish')

def limpiar_texto(texto):
    # Convertir a minúsculas
    texto = texto.lower()
    # Eliminar signos de puntuación
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    # Eliminar dígitos
    texto = re.sub(r'\d+', '', texto)
    # Eliminar stopwords
    tokens = texto.split()
    tokens = [word for word in tokens if word not in spanish_stopwords]
    texto = ' '.join(tokens)
    # Eliminar espacios en blanco adicionales
    texto = texto.strip()
    return texto

# Aplicar la función de limpieza
df['texto'] = df['texto'].apply(limpiar_texto)

# División de los datos en conjuntos de entrenamiento y prueba
X = df['texto']
y = df['Polarity']
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Codificar las etiquetas

X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=0)

# Vectorización del texto usando TF-IDF con n-gramas
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Definir modelos y parámetros para GridSearchCV usando Pipeline
from imblearn.pipeline import Pipeline  # Importar Pipeline de imblearn

# Logistic Regression
pipeline_logistic = Pipeline([
    ('smote', SMOTE(random_state=0)),
    ('clf', LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000))
])
param_grid_logistic = {
    'clf__C': [10],
    'clf__penalty': ['l1']
}

# MLPClassifier
pipeline_mlp = Pipeline([
    ('smote', SMOTE(random_state=0)),
    ('clf', MLPClassifier(max_iter=500, random_state=0))
])
param_grid_mlp = {
    'clf__hidden_layer_sizes': [(100,)],
    'clf__activation': ['relu'],
    'clf__alpha': [0.001]
}

# Configurar la validación cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Lista de modelos y sus grids
modelos = [
    ('Logistic Regression', pipeline_logistic, param_grid_logistic),
    ('MLPClassifier', pipeline_mlp, param_grid_mlp)
]

# Variable para almacenar los mejores modelos y sus scores
mejores_modelos = {}

# Iterar sobre los modelos y realizar GridSearchCV
for nombre_modelo, pipeline, param_grid in modelos:
    print(f'Entrenando y validando {nombre_modelo}...')
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train_tfidf, y_train_encoded)
    print(f'Mejores parámetros para {nombre_modelo}: {grid_search.best_params_}')
    print(f'Mejor F1 Macro promedio: {grid_search.best_score_}\n')
    mejores_modelos[nombre_modelo] = {
        'modelo': grid_search.best_estimator_,
        'score': grid_search.best_score_
    }

# Seleccionar el mejor modelo basado en F1 Macro promedio
mejor_modelo_nombre = max(mejores_modelos, key=lambda x: mejores_modelos[x]['score'])
print(f'El mejor modelo es: {mejor_modelo_nombre}')

# Entrenar el mejor modelo en el conjunto completo de entrenamiento (aplicando SMOTE)
mejor_modelo = mejores_modelos[mejor_modelo_nombre]['modelo']
mejor_modelo.fit(X_train_tfidf, y_train_encoded)

# Predecir en el conjunto de prueba
y_pred_encoded = mejor_modelo.predict(X_test_tfidf)
y_pred = le.inverse_transform(y_pred_encoded)
y_test = le.inverse_transform(y_test_encoded)

# Evaluar el modelo
f1_macro = f1_score(y_test, y_pred, average='macro')
print(f'F1 Macro en el conjunto de prueba: {f1_macro}')
print('Reporte de clasificación:')
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()
