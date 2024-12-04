import pandas as pd
import numpy as np
import stanza
import spacy
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Descargar modelos de Stanza y spaCy
stanza.download('es')
nlp_stanza = stanza.Pipeline('es')
nlp_spacy = spacy.load('es_core_news_md')


# Cargar datos
df = pd.read_excel('Rest_Mex_2022.xlsx', engine='openpyxl')

# Concatenar 'Title' y 'Opinion', manejar valores faltantes
df['Title'] = df['Title'].astype(str).fillna('')
df['Opinion'] = df['Opinion'].astype(str).fillna('')
df['texto'] = df['Title'] + ' ' + df['Opinion']


# Función para preprocesar texto usando spaCy
def preprocess_text(text, stopwords_removal=True):
    # Procesar el texto con spaCy
    doc = nlp_spacy(text)

    # Extraer tokens
    tokens = [token.text for token in doc]

    # Eliminación de stopwords (opcional)
    if stopwords_removal:
        tokens = [token.text for token in doc if not token.is_stop]
    else:
        tokens = [token.text for token in doc]

    return ' '.join(tokens)


print('preprocesar texto')
# Aplicar preprocesamiento
df['texto'] = df['texto'].apply(lambda x: preprocess_text(x, stopwords_removal=True))


# Generar embeddings usando spaCy
def generate_embeddings(texts):
    return np.array([nlp_spacy(text).vector for text in texts])


print('generar embeddings')
X = generate_embeddings(df['texto'])
y = df['Polarity']

# Codificar etiquetas
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# División de datos en entrenamiento y prueba
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=0
)


# Configurar SMOTE opcional
use_smote = True

# Pipeline con LogisticRegression
pipeline_steps = []
if use_smote:
    pipeline_steps.append(('smote', SMOTE(random_state=0)))

pipeline_steps.append(('clf', LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000)))
pipeline = Pipeline(pipeline_steps)


# Configurar validación cruzada y grid search
param_grid = {
    # 'smote__sampling_strategy': [0.5, 0.7, 1.0],  # Parametros de SMOTE
    'clf__C': [0.1, 1, 10],
    'clf__penalty': ['l1', 'l2']
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
grid_search = GridSearchCV(
    pipeline, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1
)

# Entrenar modelo
grid_search.fit(X_train, y_train_encoded)

# Evaluar resultados
print(f'Mejores parámetros: {grid_search.best_params_}')
print(f'Mejor F1 Macro promedio: {grid_search.best_score_}')

# Entrenar mejor modelo en conjunto de prueba
best_model = grid_search.best_estimator_
y_pred_encoded = best_model.predict(X_test)
y_pred = le.inverse_transform(y_pred_encoded)
y_test = le.inverse_transform(y_test_encoded)

# Reporte de resultados
f1_macro = f1_score(y_test, y_pred, average='macro')
print(f'F1 Macro en el conjunto de prueba: {f1_macro}')
print('Reporte de clasificación:')
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()
