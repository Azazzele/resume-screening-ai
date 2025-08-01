import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression  


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



#* Ищем подходящие столбцы и загружаем в модель отфилоьтрованные данные 
dataExeleTable = pd.read_excel(r'I:\Python\Jusu\since.xlsx')
dataExeleTable.columns = dataExeleTable.columns.str.strip()
dataExeleTable['Suitable'] = (
    (dataExeleTable['Experience(years)'] > 5) &
    (dataExeleTable['Age'] < 35) &
    (dataExeleTable['Skills'].str.contains('Python', case=False, na=False))
).astype(int)


label_enc = LabelEncoder()
dataExeleTable['Marital Status'] = label_enc.fit_transform(dataExeleTable['Marital Status'])
dataExeleTable['Specialization'] = label_enc.fit_transform(dataExeleTable['Specialization'])
dataExeleTable['Has_Python'] = dataExeleTable['Skills'].str.contains('Python', case=False, na=False).astype(int)
dataExeleTable['Has_SQL'] = dataExeleTable['Skills'].str.contains('SQL', case=False, na=False).astype(int)

#! ИСключение из модели ненужных столбцов и целевая переменная { y }, то что модель будет пытаться предсказать (подходит ли кандидат).
X = dataExeleTable.drop(columns=['Full Name', 'Phone', 'Email', 'Suitable', 'Expected Salary','Skills'])
y = dataExeleTable['Suitable']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(16, input_dim=X.shape[1], activation='relu'))
model.add(Dense(8, activation='tanh'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

test_indicues = X_test.index
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
print(classification_report(y_test, y_pred))

X_test_with_preds = X_test.copy()
X_test_with_preds['Predicted_Suitable'] = y_pred

test_data = dataExeleTable.loc[test_indicues].copy()
test_data['Predicted_Suitable'] = y_pred
suitable_candidates = X_test_with_preds[X_test_with_preds['Predicted_Suitable'] == 1]
suitable_candidates = test_data[test_data['Predicted_Suitable'] == 1]


merage = pd.concat([test_data, dataExeleTable.loc[X_test.index][['Full Name', 'Phone', 'Email', 'Expected Salary', 'Skills']]], axis=1)
final_candidation = merage[merage['Predicted_Suitable'] == 1]
print(final_candidation)