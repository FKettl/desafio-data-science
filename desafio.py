import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics

dataset = pd.read_csv(
    '/home/kettl/Principal/Codigos/Python/desafio-data-science/wage_train.csv',
                       index_col = 0, skipinitialspace = True)

dataset_test = pd.read_csv(
    '/home/kettl/Principal/Codigos/Python/desafio-data-science/wage_test.csv',
                       index_col = 0, skipinitialspace = True)

# Features a serem desconsideradas
dataset.drop('race', inplace=True, axis=1)
dataset.drop('native_country', inplace=True, axis=1)
dataset.drop('education', inplace=True, axis=1)
dataset.drop('capital_gain', inplace=True, axis=1)
dataset.drop('capital_loss', inplace=True, axis=1)
dataset.drop('fnlwgt', inplace=True, axis=1)
dataset.drop('relationship', inplace=True, axis=1)
dataset.drop('marital_status', inplace=True, axis=1)

dataset_test.drop('race', inplace=True, axis=1)
dataset_test.drop('native_country', inplace=True, axis=1)
dataset_test.drop('education', inplace=True, axis=1)
dataset_test.drop('capital_gain', inplace=True, axis=1)
dataset_test.drop('capital_loss', inplace=True, axis=1)
dataset_test.drop('fnlwgt', inplace=True, axis=1)
dataset_test.drop('relationship', inplace=True, axis=1)
dataset_test.drop('marital_status', inplace=True, axis=1)

# Checagem e troca dos valores faltando
for col in dataset.columns:
    dataset[col].replace('?', np.nan, inplace=True)
    #Para checar quantos valores faltam originalmente, comentar linha abaixo
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)

for col in dataset_test.columns:
    dataset_test[col].replace('?', np.nan, inplace=True)
    #Para checar quantos valores faltam originalmente, comentar linha abaixo
    dataset_test[col].fillna(dataset_test[col].mode()[0], inplace=True)

# Codificação das variáveis em labels

labelencoder = LabelEncoder()
need_label = []
need_label_test = []
types = dataset.dtypes
types_test = dataset_test.dtypes
# Procura quais colunas possuem objects
for x in types:
  if x == 'object':
    need_label.append(True)
  else:
    need_label.append(False)

for x in types_test:
  if x == 'object':
    need_label_test.append(True)
  else:
    need_label_test.append(False)

columns_need_label = dataset.columns[need_label]
columns_need_label_test = dataset_test.columns[need_label_test]
# Lista para salvar as colunas que não vão precisar de encoder
columns_0_1 = []
columns_0_1_test = []
# Transforma as colunas de objetos em labels
for x in columns_need_label:
  dataset[x] = labelencoder.fit_transform(dataset[x])
  # Verifica o valor maximo da label de cada coluna
  max = dataset[x].max()
  # Se a label for igual a 1, existem apenas dois valores possíveis (0 e 1)
  # Portanto, não sera necessário usar o encoder nessas colunas
  if max == 1:
    columns_0_1.append(x)

for x in columns_need_label_test:
  dataset_test[x] = labelencoder.fit_transform(dataset_test[x])
  # Verifica o valor maximo da label de cada coluna
  max = dataset_test[x].max()
  # Se a label for igual a 1, existem apenas dois valores possíveis (0 e 1)
  # Portanto, não sera necessário usar o encoder nessas colunas
  if max == 1:
    columns_0_1_test.append(x)

# Codificação das labels em valores

# A codificação escolhida é o OneHotEncoder,
# se cria uma nova coluna com valor 1 ou 0 para cada valor de label
# Evita que o valor seja considerado em labels não hierarquicas
encoder = OneHotEncoder(sparse=False)

# Removo as colunas que possuem máximo igual a 1 da lista
columns_need_encoder = columns_need_label.difference(columns_0_1)
columns_need_encoder_test = columns_need_label_test.difference(columns_0_1_test)
# Um novo dataset com apenas as colunas codificadas é criado
dataset_encoded = pd.DataFrame(
    encoder.fit_transform(dataset[columns_need_encoder]))
encoded_columns = encoder.get_feature_names_out()
dataset_encoded.columns = encoded_columns

dataset_encoded_test = pd.DataFrame(
    encoder.fit_transform(dataset_test[columns_need_encoder_test]))
encoded_columns_test = encoder.get_feature_names_out()
dataset_encoded_test.columns = encoded_columns_test

# Retiro as colunas originais que foram codificadas do dataset principal
dataset.drop(dataset[columns_need_encoder], axis=1, inplace=True)
dataset_test.drop(dataset_test[columns_need_encoder_test], axis=1, inplace=True)

# Concateno o dataset codificado com o principal
dataset = pd.concat([dataset, dataset_encoded], axis=1)
dataset_test = pd.concat([dataset_test, dataset_encoded_test], axis=1)

# Normalização dos valores

# As colunas que não foram codificadas precisam ser normalizadas
columns_need_normalize = []
columns_need_normalize_test = []

for x in dataset.columns:
  if x in encoded_columns:
    columns_need_normalize.append(False)
  else:
    columns_need_normalize.append(True)

for x in dataset_test.columns:
  if x in encoded_columns_test:
    columns_need_normalize_test.append(False)
  else:
    columns_need_normalize_test.append(True)

columns_need_normalize = dataset.columns[columns_need_normalize]
dataset[columns_need_normalize] = MinMaxScaler().fit_transform(
    dataset[columns_need_normalize])

columns_need_normalize_test = dataset_test.columns[columns_need_normalize_test]
dataset_test[columns_need_normalize_test] = MinMaxScaler().fit_transform(
    dataset_test[columns_need_normalize_test])

# Aplicação dos testes
first_column = dataset.pop('yearly_wage')
dataset.insert(0, 'yearly_wage', first_column)

X_train = dataset.iloc[:,1:27].values
Y_train = dataset.iloc[:,0].values

classifier = RandomForestClassifier(n_estimators=200, oob_score=False)
classifier.fit(X_train,Y_train)
prediction = classifier.predict(dataset_test)
    
results = pd.DataFrame(prediction, columns = ['predictedValues'])
results.to_csv('predicted.csv')