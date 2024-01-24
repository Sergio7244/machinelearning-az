# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Plantilla preprocesado

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Datanuevo.csv')
X = dataset.iloc[:, :-1].values
# .values es para decir que solo valores y no posiciones
Y = dataset.iloc[:, 3].values

#tratamiento de los na
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean", verbose = 0)
# verbose 0 es columna, verbose 1 es fila, en este caso es la media de la respectiva columna en donde está el nan
imputer = imputer.fit(X[:, 1:3])
#aquí arriba tengo que decirle que me impute las columnas 1 y 2 (o sea la segunda y tercera) pero toca decir que termine en la 3 para que me coja hasta la 2.
X[:,1:3] = imputer.transform(X[:,1:3])

#Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Variables Dummy
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)

X = np.array(ct.fit_transform(X), dtype=np.float)
# Para la Y no hace falta el One encoder porque es solo Yes y No, o sea solo 1 y 0:
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Dividir el dataset entre conjunto de entrenamiento y testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1310)
# el 20% se va para base de prueba y el 80% para entrenamiento
#random state es la semilla, el número lo pone uno

# Escalar Estandarizar variables:
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Y no se estandariza ya que es categórica
# acá se estandarizaron las variables dummyficadas, pero no se aconseja porque no tiene sentido