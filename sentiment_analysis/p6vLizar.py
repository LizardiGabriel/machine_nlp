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


with open("../normalizacion_texto/lista_contenido_normalizado.pkl", "rb") as f:
    corpus = pickle.load(f)

data = pd.read_excel('../apoyo/Rest_Mex_2022.xlsx')
y = data['Polarity'].values

x_train, x_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2, random_state=0)

